"""J4: a second, unseen CUDA repo, end to end: ankan-ban/llama2.cu (Llama 2 inference, MIT).

The hidden check is the whole program's output. Greedy decoding (temperature 0) with Karpathy's
stories15M model and fixed prompts, compared byte for byte with llama2.cu's own CUDA program,
recorded on an NVIDIA GPU (J1's ``record_native``).

- ``run_story.py`` (the wrapper) runs a llama2 program in a scratch dir next to the model and
  ``tokenizer.bin``, and cuts the story from its printout: everything between the ``<s>`` line
  and the final ``achieved tok/s`` timing line, which both llama2.cu and llama2.c print. It
  writes the story's bytes as f32 (one value per byte), so ``verify`` with atol 0 means the
  same text, and prints the story as JSON.
- ``recorded.py`` serves the recording: each input's recorded output, by input order.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

MODEL = "stories15M.bin"
MODEL_URL = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
MODEL_SHA256 = "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
# (temperature, steps, prompt): greedy, so the story is fixed by the model and the code
CASES = (("0.0", "128", "Once upon a time"), ("0.0", "128", "Tom and his dog went to the park"))

RUN_STORY = '''"""Hidden check wrapper (museport J4): run a llama2 program (llama2.cu,
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
start, end = text.find("<s>\\n"), text.rfind("\\nachieved tok/s")
if p.returncode != 0 or start < 0 or end < start:
    sys.exit(f"exit {p.returncode}; expected '<s>' then the story then 'achieved tok/s'; "
             "output tail: " + (text or p.stderr.decode("utf-8", "replace"))[-1500:])
story = text[start + 4:end].encode("utf-8")
Path(out).write_bytes(struct.pack(f"<{len(story)}f", *story))
print(json.dumps({"story": story.decode("utf-8", "replace")}))
'''

RECORDED = '''"""Serve the recorded reference (museport J4): copy the recording of <input> (by
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
'''

GOAL = (
    "Port llama2.cu (Llama 2 inference: a CUDA port of llama2.c, one file) to one SYCL 2020 "
    "program, llama2.cpp, with zero dpct helpers and the model computed on the SYCL device. "
    "Keep its command line (<checkpoint> [temperature] [steps] [prompt]), its loading of the "
    "checkpoint and of tokenizer.bin from the working directory, and its printout: '<s>' on its "
    "own line, then each token's text as it is generated, then the '\\nachieved tok/s: ...' line. "
    "Keep its fp16 storage (CUDA half -> sycl::half) and its fp32 accumulation; CUB block and "
    "warp reductions become SYCL group algorithms (sycl::reduce_over_group). A hidden check runs "
    "your program with temperature 0 (greedy) on fixed prompts with the stories15M model and "
    "compares the generated story byte for byte with llama2.cu's own output on an NVIDIA GPU. "
    "Compile with sources [llama2.cpp]. Success requires compile, verify and a zero-helper "
    "scan_helpers of the whole work directory (path '.') on the same sources; delete migrate's "
    "output (delete_file) once your port is done.")


def _case_files(dest: Path) -> list[str]:
    names = []
    for i, case in enumerate(CASES):
        name = f"case{i}.txt"
        (dest / name).write_text("\n".join(case) + "\n")
        names.append(name)
    return names


def make_record_task(repo: Path, dest: Path) -> Path:
    """llama2.cu, built with nvcc and run twice per case on an NVIDIA GPU: the recording that
    becomes the hidden reference. No SYCL port is built."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("llama2.cu", "tokenizer.bin"):
        shutil.copyfile(repo / name, dest / name)
    (dest / "run_story.py").write_text(RUN_STORY)
    task = {"name": "llama2_story_record", "cuda_path": "llama2.cu", "include_dirs": ["."],
            "reference_build": ["true"], "reference_cmd": "true",
            "candidate_cmd": "python3 {task_dir}/run_story.py build/candidate {input} {output}",
            "inputs": _case_files(dest), "mode": "array", "output_format": "f32",
            "tolerance": 0.0, "rtol": 0.0, "record_native": True, "benchmark_runs": 0,
            "fetch": [{"path": MODEL, "url": MODEL_URL + "?download=true",
                       "sha256": MODEL_SHA256}],
            "museport": {"item": "J4", "repo": "ankan-ban/llama2.cu",
                         "note": "greedy stories15M stories, byte for byte"}}
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"


def make_loop_task(repo: Path, recorded: Path, model: Path, dest: Path) -> Path:
    """The task Muse ports against: llama2.cu, the wrapper, and the recording (``recorded``: the
    record task's results/recorded/ directory) as the hidden reference. ``model`` is the local
    stories15M.bin; it is linked, not copied."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("llama2.cu", "tokenizer.bin"):
        shutil.copyfile(repo / name, dest / name)
    (dest / MODEL).symlink_to(model.resolve())
    shutil.copytree(recorded, dest / "recorded")
    (dest / "run_story.py").write_text(RUN_STORY)
    (dest / "recorded.py").write_text(RECORDED)
    task = {"name": "llama2_story", "cuda_path": "llama2.cu", "include_dirs": ["."],
            "reference_build": ["true"],
            "reference_cmd": "python3 {task_dir}/recorded.py {input} {output}",
            "candidate_cmd": "python3 {task_dir}/run_story.py build/candidate {input} {output}",
            "inputs": _case_files(dest), "mode": "array", "output_format": "f32",
            "tolerance": 0.0, "rtol": 0.0, "goal": GOAL,
            "museport": {"item": "J4", "repo": "ankan-ban/llama2.cu",
                         "reference": "llama2.cu's own CUDA output, recorded on an NVIDIA GPU"}}
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"
