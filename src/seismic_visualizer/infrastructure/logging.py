"""Logging setup — rotating file handler, get_logger() factory.

Usage: from seismic_visualizer.infrastructure.logging import get_logger
       logger = get_logger(__name__)
"""

import logging
import logging.handlers
from pathlib import Path

_LOGS_DIR = Path("logs")
_LOG_FILE = _LOGS_DIR / "seismic_visualizer.log"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_configured = False


def setup_logging(
    level: int = logging.DEBUG,
    log_file: Path = _LOG_FILE,
    max_bytes: int = _MAX_BYTES,
    backup_count: int = _BACKUP_COUNT,
) -> None:
    """Configure root logger once; subsequent calls are no-ops."""
    global _configured
    if _configured:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising logging with defaults if needed."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
