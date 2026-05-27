"""Tests for AppConfig persistence and recent files."""

from __future__ import annotations

from pathlib import Path

import pytest

from seismic_visualizer.infrastructure.config import AppConfig


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = AppConfig.load(tmp_path / "nonexistent.json")
    assert cfg.recent_files == []
    assert cfg.theme == "light"


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    cfg = AppConfig(theme="light", recent_files=[Path("/a/b.npz")])
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert loaded.theme == "light"
    assert loaded.recent_files == [Path("/a/b.npz")]


def test_load_missing_recent_files_field(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"theme": "dark"}', encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.recent_files == []


def test_load_corrupt_json_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not valid json", encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.recent_files == []


def test_add_recent_prepends(tmp_path: Path) -> None:
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    a.touch()
    b.touch()
    cfg = AppConfig(recent_files=[a])
    cfg.add_recent(b)
    assert cfg.recent_files[0].resolve() == b.resolve()
    assert cfg.recent_files[1].resolve() == a.resolve()


def test_add_recent_deduplicates(tmp_path: Path) -> None:
    a = tmp_path / "a.npz"
    a.touch()
    cfg = AppConfig(recent_files=[a])
    cfg.add_recent(a)
    assert len(cfg.recent_files) == 1


def test_add_recent_trims_to_max(tmp_path: Path) -> None:
    files = [tmp_path / f"{i}.npz" for i in range(10)]
    for f in files:
        f.touch()
    cfg = AppConfig(recent_files=files)
    new = tmp_path / "new.npz"
    new.touch()
    cfg.add_recent(new)
    assert len(cfg.recent_files) == 10
    assert cfg.recent_files[0].resolve() == new.resolve()


def test_get_recent_filters_nonexistent(tmp_path: Path) -> None:
    existing = tmp_path / "exists.npz"
    existing.touch()
    missing = tmp_path / "gone.npz"
    cfg = AppConfig(recent_files=[missing, existing])
    result = cfg.get_recent()
    assert result == [existing]


def test_atomic_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "dir" / "config.json"
    cfg = AppConfig()
    cfg.save(path)
    assert path.exists()
    loaded = AppConfig.load(path)
    assert loaded.theme == cfg.theme


def test_clear_recent_empties_list(tmp_path: Path) -> None:
    a = tmp_path / "a.npz"
    a.touch()
    cfg = AppConfig(recent_files=[a])
    cfg.clear_recent()
    assert cfg.recent_files == []


def test_config_path_returns_path_object() -> None:
    result = AppConfig.config_path()
    assert isinstance(result, Path)


def test_config_path_ends_with_config_json() -> None:
    result = AppConfig.config_path()
    assert result.name == "config.json"


def test_config_path_contains_seismic_visualizer() -> None:
    result = AppConfig.config_path()
    assert "SeismicVisualizer" in str(result)


def test_save_error_removes_tmp_file(tmp_path: Path, monkeypatch) -> None:
    import os

    path = tmp_path / "config.json"
    cfg = AppConfig()

    original_replace = os.replace

    def fail_replace(src, dst):
        raise OSError("simulated write failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        cfg.save(path)
    # tmp file should have been cleaned up
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_config_path_on_non_win32() -> None:
    """Line 36: else branch returns ~/.config/SeismicVisualizer/config.json."""
    from unittest.mock import patch

    with patch("seismic_visualizer.infrastructure.config.sys.platform", "linux"):
        result = AppConfig.config_path()
    assert ".config" in str(result).replace("\\", "/")


def test_save_cleanup_unlink_failure_is_silent(tmp_path: Path) -> None:
    """Lines 76-77: if os.unlink raises during cleanup, the error is swallowed."""
    import os

    path = tmp_path / "cfg.json"
    cfg = AppConfig()

    def fail_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    def fail_unlink(f: str) -> None:
        raise OSError("unlink failed")

    with pytest.MonkeyPatch().context() as m:
        m.setattr(os, "replace", fail_replace)
        m.setattr(os, "unlink", fail_unlink)
        with pytest.raises(OSError, match="disk full"):
            cfg.save(path)


def test_load_preserves_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    import logging
    cfg = AppConfig(
        log_level=logging.WARNING,
        max_log_size=5 * 1024 * 1024,
        backup_count=3,
        theme="dark",
        colormap="gray",
        recent_files=[],
    )
    cfg.save(path)
    loaded = AppConfig.load(path)
    assert loaded.log_level == logging.WARNING
    assert loaded.max_log_size == 5 * 1024 * 1024
    assert loaded.backup_count == 3
    assert loaded.theme == "dark"
    assert loaded.colormap == "gray"
