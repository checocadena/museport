"""Hidden check wrapper (museport L1): run llm.c's bf16 GPT-2 test and write, as f32:
whether it printed "overall okay: 1" and exited 0; whether its source keeps llm.c's tolerances;
then the loss of each of its 10 steps.

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
GRAD_TOLS = (0.5, 0.004, 0.1, 0.04, 0.05, 0.035, 0.02, 0.03, 0.05, 0.03, 0.03, 0.03, 0.02, 0.01, 0.1, 0.02)


def tolerances_kept() -> bool:
    """The test's source keeps llm.c's thresholds: the port's test_gpt2.cpp in the work dir
    (or the task's agent_port/ on a GPU machine), else llm.c's own test_gpt2.cu."""
    for src in (Path("test_gpt2.cpp"), here / "agent_port" / "test_gpt2.cpp",
                here / "test_gpt2.cu"):
        if src.exists():
            break
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", src.read_text()), flags=re.S)
    val = r"(\d+\.?\d*(?:[eE][-+]?\d+)?)f?"
    nums = [float(v) for v in re.findall(r"(?<![\w.])" + val + r"(?![\w.])", text)]
    # every assignment (or #define) to a logit or loss threshold: llm.c's bf16 value must be
    # among them, and none may be looser
    sets = re.findall(r"(?:#define\s+)?(\w+)\s*=?\s*" + val + r"\s*;?", text)
    def bound(word, limit):
        vs = [float(v) for n, v in sets
              if word in n.lower() and any(k in n.lower() for k in ("thresh", "tol"))]
        return limit in vs and all(v <= limit for v in vs)
    def has_run(seq):
        n = len(seq)
        return any(nums[i:i + n] == list(seq) for i in range(len(nums) - n + 1))
    return bound("logit", 25.0) and bound("loss", 0.05) and has_run(GRAD_TOLS)


with tempfile.TemporaryDirectory() as run:
    for name in ("gpt2_124M_bf16.bin", "gpt2_124M_debug_state.bin"):
        os.symlink(here / name, Path(run) / name)
    data = Path(run) / "dev" / "data" / "tinyshakespeare"
    data.mkdir(parents=True)
    os.symlink(here / "tiny_shakespeare_val.bin", data / "tiny_shakespeare_val.bin")
    p = subprocess.run([str(binary)], cwd=run, capture_output=True, text=True, timeout=3000)
text = p.stdout
num = r"([-+0-9.eE]+|nan|-?inf)"
ok = re.findall(r"overall okay: (\d+)", text)
losses = [float(v) for v in re.findall(r"step \d+: loss " + num, text)]
if not ok or len(losses) < 10:
    sys.exit(f"exit {p.returncode}; expected 'overall okay' and 10 step losses, got {len(ok)} "
             f"and {len(losses)}; output tail: " + (text or p.stderr)[-2500:])
vals = [1.0 if ok[-1] == "1" and p.returncode == 0 else 0.0,
        1.0 if tolerances_kept() else 0.0, *losses[:10]]
Path(out).write_bytes(struct.pack(f"<{len(vals)}f", *vals))
print("overall okay:", ok[-1], "| tolerances kept:", bool(vals[1]), "| losses:", losses[:10])
