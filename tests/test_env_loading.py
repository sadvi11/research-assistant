"""The README says `cp .env.example .env`. Something must read it."""
from __future__ import annotations

import os

from config.env import load_env


def test_env_file_is_loaded(tmp_path):
    env = tmp_path / ".env"
    env.write_text("RA_TEST_VALUE=from_file\n# comment\n\nRA_TEST_QUOTED=\"quoted value\"\n")
    os.environ.pop("RA_TEST_VALUE", None)
    os.environ.pop("RA_TEST_QUOTED", None)
    assert load_env(env) == 2
    assert os.environ["RA_TEST_VALUE"] == "from_file"
    assert os.environ["RA_TEST_QUOTED"] == "quoted value"
    os.environ.pop("RA_TEST_VALUE"); os.environ.pop("RA_TEST_QUOTED")


def test_real_environment_wins_over_the_file(tmp_path):
    """A CI secret must never be silently replaced by a stale local file."""
    env = tmp_path / ".env"
    env.write_text("RA_TEST_PRECEDENCE=from_file\n")
    os.environ["RA_TEST_PRECEDENCE"] = "from_environment"
    load_env(env)
    assert os.environ["RA_TEST_PRECEDENCE"] == "from_environment"
    os.environ.pop("RA_TEST_PRECEDENCE")


def test_export_prefix_is_tolerated(tmp_path):
    env = tmp_path / ".env"
    env.write_text("export RA_TEST_EXPORTED=yes\n")
    os.environ.pop("RA_TEST_EXPORTED", None)
    load_env(env)
    assert os.environ["RA_TEST_EXPORTED"] == "yes"
    os.environ.pop("RA_TEST_EXPORTED")


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == 0


def test_malformed_lines_are_skipped(tmp_path):
    env = tmp_path / ".env"
    env.write_text("no equals sign here\n=missing_key\nRA_TEST_OK=fine\n")
    os.environ.pop("RA_TEST_OK", None)
    assert load_env(env) == 1
    os.environ.pop("RA_TEST_OK")


def test_inline_comments_are_stripped(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "RA_TEST_INLINE=high    # low | medium\n"
        "RA_TEST_HASH=a#b\n"
        'RA_TEST_QHASH="keep # this"  # but not this\n'
        "RA_TEST_EMPTY= # nothing set\n"
    )
    loaded: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", loaded)
    load_env(env)
    assert loaded == {
        "RA_TEST_INLINE": "high",
        "RA_TEST_HASH": "a#b",
        "RA_TEST_QHASH": "keep # this",
        "RA_TEST_EMPTY": "",
    }


def test_shipped_template_loads_without_comments_in_values(monkeypatch):
    """The first live run would have sent effort='high   # low | medium ...' to the API."""
    from pathlib import Path

    loaded: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", loaded)
    load_env(Path(__file__).resolve().parent.parent / ".env.example")
    assert loaded, "template produced no variables"
    assert {k: v for k, v in loaded.items() if "#" in v} == {}
