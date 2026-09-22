"""
T0.2 — Structured logging setup.
Call setup_logging() once at process start (entry points only).
All other modules use: logger = logging.getLogger(__name__)
Emits key=value pairs for easy grep/parsing.
"""
from __future__ import annotations

import logging
import time


class KVFormatter(logging.Formatter):
    """Formatter that emits: LEVEL t=<monotonic_s> name=<logger> key=value ..."""

    _start = time.monotonic()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = time.monotonic() - self._start
        base = (
            f"{record.levelname:<8} "
            f"t={elapsed:.3f} "
            f"logger={record.name} "
            f"msg={record.getMessage()}"
        )
        return base


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with KVFormatter to stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(KVFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
