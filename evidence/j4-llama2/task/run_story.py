"""Hidden check wrapper (museport J4): run a llama2 program (llama2.cu,
llama2.c's run.c, or a port) on one case and write the story it prints as f32, one value per
byte; print it as JSON.

usage: run_story.py <binary> <case file: temperature, steps, prompt on three lines> <output>"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
binary, case, out = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
temperature, steps, prompt = Path(case).read_text().splitlines()[:3]
with tempfile.TemporaryDirectory() as run:
    for name in ("stories15M.bin", "tokenizer.bin"):
        os.symlink(here / name, Path(run) / name)
    p = subprocess.run([str(binary), "stories15M.bin", temperature, steps, prompt], cwd=run,
                       capture_output=True, timeout=1200)
text = p.stdout.decode("utf-8", "replace")
start, end = text.find("<s>\n"), text.rfind("\nachieved tok/s")
if p.returncode != 0 or start < 0 or end < start:
    sys.exit(f"exit {p.returncode}; expected '<s>' then the story then 'achieved tok/s'; "
             "output tail: " + (text or p.stderr.decode("utf-8", "replace"))[-1500:])
story = text[start + 4:end].encode("utf-8")
Path(out).write_bytes(struct.pack(f"<{len(story)}f", *story))
print(json.dumps({"story": story.decode("utf-8", "replace")}))
