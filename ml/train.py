"""Training script for seismic feature tracking U-Net.

Usage:
    python -m ml.train
    python -m ml.train --label_type faults --epochs 100
    python -m ml.train --cubes_dir path/to/cubes --batch_size 4 --tile_size 512
"""

from __future__ import annotations

import argparse
import csv
import random
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.config import Config
from ml.dataset import SeismicSliceDataset, get_cube_stems
from ml.losses import BCEDiceLoss
from ml.unet import UNet
from ml.visualize import save_sample_grid


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_entity_split(
    stems: list[str],
    cubes_dir: Path,
    label_type: str,
    n_val_per_cube: int,
    seed: int,
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Per-cube entity-id split: hold out n_val_per_cube entities per cube as val.

    Returns (train_entities, val_entities) — each cube → set of entity_ids.
    The same seismic appears in both splits but with different target horizons,
    which is exactly the inference scenario: model must follow the hint, not
    memorise the geology.
    """
    rng = random.Random(seed)
    train_ent: dict[str, set[int]] = {}
    val_ent: dict[str, set[int]] = {}
    for stem in sorted(stems):
        labels = np.load(cubes_dir / f"{stem}_{label_type}.npz")["labels"]
        ents = sorted(int(e) for e in np.unique(labels) if e != 0)
        if len(ents) == 0:
            continue
        if len(ents) == 1:
            # Can't split — give the only entity to train, no val from this cube
            train_ent[stem] = set(ents)
            continue
        n_val = max(1, min(n_val_per_cube, len(ents) - 1))
        shuffled = list(ents)
        rng.shuffle(shuffled)
        val_ent[stem] = set(shuffled[:n_val])
        train_ent[stem] = set(shuffled[n_val:])
    return train_ent, val_ent


def _init_log(
    path: Path,
    cfg: "Config",
    train_stems: set[str],
    val_stems: set[str],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(
            f"# data: label_type={cfg.data.label_type}, tile_size={cfg.data.tile_size}, "
            f"val_fraction={cfg.data.val_fraction}, seed={cfg.data.seed}, "
            f"min_labels={cfg.data.min_labels_per_slice}, min_tile_labels={cfg.data.min_tile_labels}\n"
        )
        f.write(f"# train_cubes: {sorted(train_stems)}\n")
        f.write(f"# val_cubes:   {sorted(val_stems)}\n")
        f.write(
            f"# train_cfg: epochs={cfg.train.epochs}, batch={cfg.train.batch_size}, "
            f"lr={cfg.train.lr}, wd={cfg.train.weight_decay}, "
            f"bce={cfg.train.bce_weight}, dice={cfg.train.dice_weight}\n"
        )
        f.write(
            f"# model: UNet in_ch={cfg.model.in_channels}, "
            f"base_filters={cfg.model.base_filters}, depth={cfg.model.depth}\n"
        )
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s", "cumul_time_s", "is_best"])


def _append_epoch(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    epoch_time: float,
    cumul_time: float,
    is_best: bool,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{lr:.8f}",
            f"{epoch_time:.1f}",
            f"{cumul_time:.1f}",
            int(is_best),
        ])


def train(
    cfg: Config,
    val_cubes_override: list[str] | None = None,
    split_mode: Literal["cube", "entity"] = "cube",
    n_val_entities_per_cube: int = 1,
) -> None:
    _set_seed(cfg.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_stems = get_cube_stems(cfg.data.cubes_dir, cfg.data.label_type)
    if not all_stems:
        raise FileNotFoundError(f"No cubes found in {cfg.data.cubes_dir}")

    ds_kwargs: dict = dict(
        cubes_dir=cfg.data.cubes_dir,
        label_type=cfg.data.label_type,
        tile_size=cfg.data.tile_size,
        min_labels=cfg.data.min_labels_per_slice,
        min_tile_labels=cfg.data.min_tile_labels,
        n_seeds=cfg.data.n_seeds,
        seed_sigma=cfg.data.seed_sigma,
        axes=cfg.data.axes,
    )

    if split_mode == "slice":
        # Random per-slice split. NOTE: adjacent slices in the same cube are
        # nearly identical, so this is an in-distribution metric (not a true
        # generalisation test). Useful to demonstrate the model fits the
        # training distribution well; honest about the limitation in writeup.
        from torch.utils.data import Subset

        train_full = SeismicSliceDataset(**ds_kwargs, augment=True)
        val_full = SeismicSliceDataset(**ds_kwargs, augment=False)
        n = len(train_full)
        rng = np.random.RandomState(cfg.data.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * cfg.data.val_fraction))
        val_idx_set = set(int(i) for i in perm[:n_val])
        train_idx = sorted(set(range(n)) - val_idx_set)
        val_idx = sorted(val_idx_set)

        train_ds = Subset(train_full, train_idx)
        val_ds = Subset(val_full, val_idx)
        train_stems = set(train_full._labels.keys())
        val_stems = set(val_full._labels.keys())
        print(f"Split mode: slice (random {len(val_idx)}/{n} = {cfg.data.val_fraction:.0%} val)")
    elif split_mode == "entity":
        # Per-cube entity-id split. Both splits use the same cubes' seismic
        # but target different horizons — directly mirrors inference behaviour.
        train_ent, val_ent = _build_entity_split(
            all_stems, cfg.data.cubes_dir, cfg.data.label_type,
            n_val_per_cube=n_val_entities_per_cube, seed=cfg.data.seed,
        )
        train_ds = SeismicSliceDataset(**ds_kwargs, entity_filter=train_ent, augment=True)
        val_ds = SeismicSliceDataset(**ds_kwargs, entity_filter=val_ent, augment=False)
        print(f"Split mode: entity (n_val_per_cube={n_val_entities_per_cube})")
        for stem in sorted(set(train_ent) | set(val_ent)):
            print(f"  {stem}: train={sorted(train_ent.get(stem, set()))}  val={sorted(val_ent.get(stem, set()))}")
        train_stems = set(train_ent)
        val_stems = set(val_ent)
    else:
        # Cube-level split: holds out an entire cube as val (true generalisation test).
        if val_cubes_override:
            unknown = [c for c in val_cubes_override if c not in all_stems]
            if unknown:
                raise ValueError(f"Unknown val cubes: {unknown}. Available: {all_stems}")
            val_stems = set(val_cubes_override)
            train_stems = set(all_stems) - val_stems
            if not train_stems:
                raise ValueError("No training cubes left after applying val override.")
        elif len(all_stems) == 1:
            train_stems = set(all_stems)
            val_stems = set(all_stems)
        else:
            random.shuffle(all_stems)
            n_val = max(1, int(len(all_stems) * cfg.data.val_fraction))
            val_stems = set(all_stems[:n_val])
            train_stems = set(all_stems[n_val:])
        train_ds = SeismicSliceDataset(**ds_kwargs, cube_filter=train_stems, augment=True)
        val_ds = SeismicSliceDataset(**ds_kwargs, cube_filter=val_stems, augment=False)

    print(f"Train: {len(train_ds)} samples from {sorted(train_stems)}")
    print(f"Val  : {len(val_ds)} samples from {sorted(val_stems)}")

    # num_workers=0 required — dataset holds numpy arrays that don't pickle well
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    model = UNet(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        depth=cfg.model.depth,
        dropout=cfg.model.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # Linear warmup → cosine decay. Warmup avoids early-epoch divergence at higher
    # effective LRs and lets the model find a smoother trajectory toward the minimum.
    if cfg.train.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=cfg.train.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.train.epochs - cfg.train.warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[cfg.train.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    criterion = BCEDiceLoss(cfg.train.bce_weight, cfg.train.dice_weight)

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ckpt_dir / "training_log.csv"
    _init_log(log_path, cfg, train_stems, val_stems)

    best_val_loss = float("inf")
    cumul_time = 0.0
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = _time.time()
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % cfg.train.log_every == 0:
                print(f"  [{epoch}/{cfg.train.epochs}] step {step}/{len(train_loader)}  loss={running/step:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_train = running / len(train_loader)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                val_running += criterion(model(x), y).item()
        avg_val = val_running / len(val_loader)

        epoch_time = _time.time() - epoch_start
        cumul_time += epoch_time
        print(f"Epoch {epoch}/{cfg.train.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  lr={current_lr:.2e}  time={epoch_time:.0f}s")

        if epoch % cfg.train.save_samples_every == 0 or epoch == cfg.train.epochs:
            save_sample_grid(
                model, val_loader, device,
                ckpt_dir / "samples" / f"epoch_{epoch:04d}.png",
            )
            model.train()

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": avg_val,
            "cfg_model": cfg.model,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  → best.pt saved (val={avg_val:.4f})")
        else:
            epochs_no_improve += 1

        _append_epoch(log_path, epoch, avg_train, avg_val, current_lr, epoch_time, cumul_time, is_best)

        if epochs_no_improve >= cfg.train.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val improvement for {epochs_no_improve} epochs (best={best_val_loss:.4f})"
            )
            break

    print(f"Done. Best val loss: {best_val_loss:.4f}  Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train seismic U-Net")
    parser.add_argument("--label_type", choices=["horizons", "faults"], default="horizons")
    parser.add_argument("--cubes_dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument(
        "--val_cubes",
        nargs="+",
        default=None,
        help="Cube stems to use as validation set (e.g. --val_cubes teapot kerry). "
        "Overrides random split. Remaining cubes go to training. (split_mode=cube only)",
    )
    parser.add_argument(
        "--split_mode",
        choices=["cube", "entity", "slice"],
        default="cube",
        help="cube: hold out whole cubes as val (true generalisation test). "
        "entity: per-cube hold out N entity_ids as val (mirrors inference). "
        "slice: random per-slice split (in-distribution; has leakage from "
        "adjacent slices but gives smoothest curves).",
    )
    parser.add_argument(
        "--n_val_entities_per_cube",
        type=int,
        default=1,
        help="When split_mode=entity, how many entity_ids per cube go to val.",
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.data.label_type = args.label_type
    # Both horizons and faults train on vertical sections (inline + crossline) —
    # time slices through layered media are too low-contrast for faults.
    # Separate checkpoint dir so a faults run doesn't clobber horizon weights.
    if args.label_type == "faults":
        cfg.train.checkpoint_dir = Path("ml/checkpoints_faults")
    if args.cubes_dir:
        cfg.data.cubes_dir = args.cubes_dir
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.tile_size:
        cfg.data.tile_size = args.tile_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir

    train(
        cfg,
        val_cubes_override=args.val_cubes,
        split_mode=args.split_mode,
        n_val_entities_per_cube=args.n_val_entities_per_cube,
    )


if __name__ == "__main__":
    main()
