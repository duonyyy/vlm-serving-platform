"""Structured JSON logger with request ID context."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .context import get_request_id


class StructuredJsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects with standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or get_request_id() or None,
        }

        # Include exception info if available
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra payload data if passed in record.__dict__
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with StructuredJsonFormatter."""
    root_logger = logging.getLogger()
    # Avoid duplicate handlers if already configured
    if not any(isinstance(h.formatter, StructuredJsonFormatter) for h in root_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        root_logger.handlers = [handler]
        root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Obtain a logger instance configured for structured JSON output."""
    setup_logging()
    return logging.getLogger(name)
