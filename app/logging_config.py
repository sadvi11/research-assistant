"""Structured logging that never emits secrets.

A redacting filter is applied to the root logger rather than trusting every
call site to remember. Anything resembling an API key is masked on its way out,
including inside exception text where it is easiest to leak accidentally.
"""
from __future__ import annotations

import json
import logging
import re
import sys

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|auth[_-]?token|password|secret)"
               r"\s*[=:]\s*['\"]?([^\s'\"]{6,})"),
    re.compile(r"postgres(?:ql)?://[^:]+:([^@]+)@"),
]

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
}


def redact(text: str) -> str:
    if not text:
        return text
    out = _SECRET_PATTERNS[0].sub("sk-ant-***REDACTED***", text)
    out = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}=***REDACTED***", out)
    out = _SECRET_PATTERNS[2].sub(lambda m: m.group(0).replace(m.group(1), "***"), out)
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(
                redact(a) if isinstance(a, str) else a for a in record.args
            ) if isinstance(record.args, tuple) else record.args
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
