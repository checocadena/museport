"""Serve the recorded reference (museport J4): copy the recording of <input> (by
its position in the task's inputs) to <output>.

usage: recorded.py <input> <output>"""
import json
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
inputs = json.loads((here / "task.json").read_text())["inputs"]
i = [Path(x).name for x in inputs].index(Path(sys.argv[1]).name)
shutil.copyfile(here / "recorded" / f"case{i}.out", sys.argv[2])
