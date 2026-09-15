"""Load a project .env file into os.environ.

The README instructs `cp .env.example .env`, so something must actually read it.
Deliberately dependency-free and deliberately non-overriding: a variable already
set in the real environment always wins, so a CI secret is never silently
replaced by a stale local file.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_value(raw: str) -> str:
    """Shell rules: in an unquoted value, ` #` starts a comment; quotes keep `#` literal.

    `.env.example` documents options inline (`RA_EFFORT=high   # low | medium ...`),
    so without this the comment became part of the value and the API rejected it.
    """
    value = raw.strip()
    if value[:1] in ('"', "'"):
        end = value.find(value[0], 1)
        return value[1:end] if end != -1 else value[1:]
    return re.split(r"\s#", raw, maxsplit=1)[0].strip()


def load_env(path: str | Path | None = None, *, override: bool = False) -> int:
    """Read KEY=VALUE lines from `.env`. Returns the number of variables set."""
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return 0

    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = _parse_value(value)
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded += 1

    if loaded:
        # Never log names or values - a key name is a hint, a value is a breach.
        logger.info("loaded environment file", extra={"variables": loaded})
    return loaded
