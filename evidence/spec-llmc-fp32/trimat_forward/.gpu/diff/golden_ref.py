"""Hidden reference of a differential check: copy the original program's
recorded output for this input. usage: golden_ref.py <input> <output>"""
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
index = __import__("json").loads((here / "golden" / "index.json").read_text())
name = Path(sys.argv[1]).name
if name not in index:
    sys.exit(f"no recorded output for input {name!r}")
shutil.copyfile(here / "golden" / index[name], sys.argv[2])
