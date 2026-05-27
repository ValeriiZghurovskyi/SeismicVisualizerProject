"""Export trained PyTorch U-Net to ONNX for use with OnnxTracker.

The exported model accepts dynamic batch size and spatial dimensions, so the
inference code can pass tiles of any size without re-exporting.

Usage:
    python -m ml.export_onnx
    python -m ml.export_onnx --checkpoint ml/checkpoints/best.pt --output models/horizon_tracker.onnx
    python -m ml.export_onnx --label_type faults --output models/fault_tracker.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ml.config import Config
from ml.unet import UNet


def export(
    checkpoint_path: Path,
    output_path: Path,
    tile_size: int = 256,
    cfg: Config | None = None,
) -> None:
    if cfg is None:
        cfg = Config()

    device = torch.device("cpu")  # export on CPU for maximum portability

    model = UNet(
        in_channels=cfg.model.in_channels,
        base_filters=cfg.model.base_filters,
        depth=cfg.model.depth,
        dropout=cfg.model.dropout,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros(1, cfg.model.in_channels, tile_size, tile_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Exported  → {output_path}  ({size_mb:.1f} MB)")
    print(f"Input     : (batch, {cfg.model.in_channels}, H, W)  — seismic + hint")
    print(f"Output    : (batch, 1, H, W)  — raw logits, apply sigmoid for probs")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(output_path)))
        print("ONNX check: OK")
    except ImportError:
        print("Skipping ONNX check — run: pip install onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export U-Net checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=Path("ml/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/horizon_tracker.onnx"))
    parser.add_argument("--tile_size", type=int, default=256)
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.tile_size)


if __name__ == "__main__":
    main()
