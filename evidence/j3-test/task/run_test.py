"""Hidden check wrapper (museport J3): run llm.c's GPT-2 test and write, as f32,
whether it printed "overall okay: 1" (1.0 or 0.0), then the loss of each of its 10 steps.

usage: run_test.py <binary> <input (unused)> <output>"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
binary, out = Path(sys.argv[1]).resolve(), sys.argv[3]
with tempfile.TemporaryDirectory() as run:
    for name in ("gpt2_124M.bin", "gpt2_124M_debug_state.bin"):
        os.symlink(here / name, Path(run) / name)
    p = subprocess.run([str(binary)], cwd=run, capture_output=True, text=True, timeout=1500)
text = p.stdout
num = r"([-+0-9.eE]+|nan|-?inf)"
ok = re.findall(r"overall okay: (\d+)", text)
losses = [float(v) for v in re.findall(r"step \d+: loss " + num, text)]
if not ok or len(losses) < 10:
    sys.exit(f"exit {p.returncode}; expected 'overall okay' and 10 step losses, got {len(ok)} "
             f"and {len(losses)}; output tail: " + (text or p.stderr)[-1500:])
vals = [1.0 if ok[-1] == "1" and p.returncode == 0 else 0.0, *losses[:10]]
Path(out).write_bytes(struct.pack(f"<{len(vals)}f", *vals))
