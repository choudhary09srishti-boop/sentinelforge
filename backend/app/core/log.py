"""Structured JSON logs: one line per event, with request IDs and audit events."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

LOGGER_NAME = "sentinelforge"
MAX_VALUE_CHARS = 300


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_VALUE_CHARS else value[:MAX_VALUE_CHARS] + "..."
    if value is None or isinstance(value, bool | int | float):
        return value
    return _clip(str(value))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {}
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                entry[str(key)] = _clip(value)
        created = datetime.fromtimestamp(record.created, tz=UTC)
        entry["ts"] = created.isoformat(timespec="milliseconds")
        entry["level"] = record.levelname.lower()
        entry["logger"] = record.name
        entry["message"] = _clip(record.getMessage())
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc_type"] = record.exc_info[0].__name__
        return json.dumps(entry, ensure_ascii=True)


def configure_logging() -> None:
    """Attach one JSON handler to the app logger. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    if any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def audit(event: str, **fields: Any) -> None:
    """Record a security-relevant event (sandbox runs, patch decisions) as structured JSON."""
    payload = {"event": event, **fields}
    logging.getLogger(f"{LOGGER_NAME}.audit").info(event, extra={"fields": payload})
