"""Redact credentials from run evidence before it is saved or committed.

A run with the shell tool can print its environment (L1 run 4 did, in turn 5), and this
session's environment holds API keys. ``redact`` replaces the value of every credential-like
environment variable with ``<redacted:NAME>``; ``secret_hits`` lists what is still present.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|ACCESS|CREDENTIAL)", re.I)
NOT_SECRET = {"MAX_THINKING_TOKENS", "BUN_TLS_EXTRA_DISTRUSTED_KEYS",
              "CCR_AUTO_MODE_USER_ENV_KEYS_FACT"}


def secrets(env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if env is None else env)
    return {n: v for n, v in env.items()
            if SECRET_NAME.search(n) and n not in NOT_SECRET and len(v) >= 12}


def redact(text: str, env: dict[str, str] | None = None) -> str:
    for name, value in sorted(secrets(env).items(), key=lambda kv: -len(kv[1])):
        text = text.replace(value, f"<redacted:{name}>")
    return text


def redact_tree(root: Path, env: dict[str, str] | None = None) -> list[str]:
    """Redact every text file under ``root`` in place; returns the files changed."""
    changed = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        new = redact(text, env)
        if new != text:
            f.write_text(new)
            changed.append(str(f))
    return changed


def secret_hits(root: Path, env: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """(file, variable name) for every credential value still present under ``root``."""
    found = secrets(env)
    hits = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and not f.is_symlink():
            data = f.read_bytes()
            hits += [(str(f), n) for n, v in found.items() if v.encode() in data]
    return hits
