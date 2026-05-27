"""AppConfig — application configuration with JSON persistence."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

_MAX_RECENT = 10


@dataclass
class AppConfig:
    log_level: int = logging.DEBUG
    max_log_size: int = 10 * 1024 * 1024
    backup_count: int = 5
    theme: str = "light"
    colormap: str = "seismic"
    language: str = "en"
    recent_files: list[Path] = field(default_factory=list)


    @staticmethod
    def config_path() -> Path:
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home()))
        else:
            base = Path.home() / ".config"
        return base / "SeismicVisualizer" / "config.json"

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _log.warning("Could not read config %s: %s", path, exc)
            return cls()
        recent = [Path(p) for p in data.get("recent_files", [])]
        return cls(
            log_level=data.get("log_level", logging.DEBUG),
            max_log_size=data.get("max_log_size", 10 * 1024 * 1024),
            backup_count=data.get("backup_count", 5),
            theme=data.get("theme", "light"),
            colormap=data.get("colormap", "seismic"),
            language=data.get("language", "en"),
            recent_files=recent,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "log_level": self.log_level,
            "max_log_size": self.max_log_size,
            "backup_count": self.backup_count,
            "theme": self.theme,
            "colormap": self.colormap,
            "language": self.language,
            "recent_files": [str(p) for p in self.recent_files],
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


    def add_recent(self, path: Path) -> None:
        resolved = path.resolve()
        self.recent_files = [resolved] + [
            p for p in self.recent_files if p.resolve() != resolved
        ]
        self.recent_files = self.recent_files[:_MAX_RECENT]

    def get_recent(self) -> list[Path]:
        return [p for p in self.recent_files if p.exists()]

    def clear_recent(self) -> None:
        self.recent_files = []
