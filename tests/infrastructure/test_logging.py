"""Tests for infrastructure.logging — setup_logging and get_logger."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

import seismic_visualizer.infrastructure.logging as _log_module
from seismic_visualizer.infrastructure.logging import get_logger, setup_logging


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset the module-level _configured flag and root logger handlers after each test."""
    original = _log_module._configured
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    yield
    _log_module._configured = original
    # Remove any handlers added by setup_logging during the test
    for h in list(root.handlers):
        if h not in original_handlers:
            h.close()
            root.removeHandler(h)


def test_setup_logging_creates_log_file(tmp_path: Path) -> None:
    _log_module._configured = False
    log_file = tmp_path / "test.log"
    setup_logging(log_file=log_file, max_bytes=1024, backup_count=1)
    assert log_file.exists()


def test_setup_logging_adds_handler_to_root(tmp_path: Path) -> None:
    _log_module._configured = False
    root = logging.getLogger()
    count_before = len(root.handlers)
    setup_logging(log_file=tmp_path / "t.log", max_bytes=1024, backup_count=1)
    assert len(root.handlers) > count_before


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    _log_module._configured = False
    log_file = tmp_path / "t.log"
    setup_logging(log_file=log_file, max_bytes=1024, backup_count=1)
    count_after_first = len(logging.getLogger().handlers)
    setup_logging(log_file=log_file, max_bytes=1024, backup_count=1)
    assert len(logging.getLogger().handlers) == count_after_first


def test_setup_logging_creates_parent_dirs(tmp_path: Path) -> None:
    _log_module._configured = False
    log_file = tmp_path / "sub" / "dir" / "app.log"
    setup_logging(log_file=log_file, max_bytes=1024, backup_count=1)
    assert log_file.parent.exists()


def test_setup_logging_sets_configured_flag(tmp_path: Path) -> None:
    _log_module._configured = False
    setup_logging(log_file=tmp_path / "t.log", max_bytes=1024, backup_count=1)
    assert _log_module._configured is True


def test_get_logger_returns_logger_with_correct_name(tmp_path: Path) -> None:
    _log_module._configured = False
    _log_module._configured = True  # skip actual file creation
    logger = get_logger("seismic_test_module")
    assert logger.name == "seismic_test_module"


def test_get_logger_triggers_setup_when_not_configured(tmp_path: Path) -> None:
    _log_module._configured = False
    # Point setup_logging to a writable tmp path via monkeypatching the module constant
    original = _log_module._LOG_FILE
    _log_module._LOG_FILE = tmp_path / "fallback.log"
    try:
        logger = get_logger("any")
        assert isinstance(logger, logging.Logger)
        assert _log_module._configured is True
    finally:
        _log_module._LOG_FILE = original


def test_get_logger_returns_logging_logger_instance(tmp_path: Path) -> None:
    _log_module._configured = True
    result = get_logger("my.module")
    assert isinstance(result, logging.Logger)
