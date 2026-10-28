"""Hidden check wrapper (museport J3): run an llm.c GPT-2 trainer on the short
tiny-shakespeare run and write its printed losses as f32: the first val loss, the first 10
train losses, the second val loss.

usage: run_train.py [--reference] <binary> <args file> <output>
  --reference: llm.c's train_gpt2.c (fixed settings, no arguments)"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
argv = sys.argv[1:]
reference = argv[:1] == ["--reference"]
if reference:
    argv = argv[1:]
binary, args_file, out = Path(argv[0]).resolve(), argv[1], argv[2]
args = [] if reference else Path(args_file).read_text().split()
with tempfile.TemporaryDirectory() as run:
    run = Path(run)
    data = run / "dev" / "data" / "tinyshakespeare"
    data.mkdir(parents=True)
    for name in ("gpt2_124M.bin", "gpt2_tokenizer.bin"):
        os.symlink(here / name, run / name)
    for name in ("tiny_shakespeare_train.bin", "tiny_shakespeare_val.bin"):
        os.symlink(here / "data" / name, data / name)
    p = subprocess.run([str(binary), *args], cwd=run, capture_output=True, text=True,
                       timeout=1500)
text = p.stdout
if p.returncode != 0:
    sys.exit(f"exit {p.returncode}: " + (p.stderr or text)[-2000:])
num = r"([-+0-9.eE]+|nan|-?inf)"
val = [float(v) for v in re.findall(r"val loss " + num, text)]
train = [float(v) for v in re.findall(r"step\s+\d+(?:/\d+)?: train loss " + num, text)]
if len(val) < 2 or len(train) < 10:
    sys.exit(f"expected 2 val losses and 10 train losses, got {len(val)} and {len(train)}; "
             "output tail: " + text[-1500:])
vals = [val[0], *train[:10], val[1]]
Path(out).write_bytes(struct.pack(f"<{len(vals)}f", *vals))
