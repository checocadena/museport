"""Run the demo's terminal commands for real and record what they print, for the terminal shots.

usage: python3 demo/capture.py <llm.c parent dir> > demo/sessions.json
(run from the repository root, with the toolkit's toolchain on PATH)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: str, cwd: Path, keep: int = 18,
        hide: dict[str, str] | None = None) -> dict[str, object]:
    out = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True,
                         timeout=900)
    text = out.stdout + out.stderr
    for path, shown in (hide or {}).items():  # this machine's paths are not part of the demo
        text, cmd = text.replace(path, shown), cmd.replace(path, shown)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {"cmd": cmd, "lines": lines[:keep] + (["…"] if len(lines) > keep else []),
            "exit": out.returncode}


def main(ext: Path) -> dict[str, list[dict[str, object]]]:
    site = Path(tempfile.mkdtemp())
    try:
        return {
            "scan": [run("museport scan llm.c/dev/cuda | sed -n '1,16p'", ext,
                         hide={f"{ext}/": ""})],
            "bug": [run("demo/planted_bug.sh", ROOT, keep=10)],
            "report": [run(f"museport report evidence --out {site}", ROOT, keep=3,
                           hide={str(site): "site"})],
        }
    finally:
        shutil.rmtree(site, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
