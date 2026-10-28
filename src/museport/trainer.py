"""J3 step 2: Muse turns its proven GPT-2 port into llm.c's real fp32 trainer.

The GPT-2 stage-3 port (``evidence/gpt2-stage3``) holds the model, forward, backward and
AdamW, behind a test ``main``. llm.c's ``train_gpt2_fp32.cu`` also has a real ``main``:
- its command line;
- the train and val DataLoaders;
- validation loss, sampling with the tokenizer, and timing.

This task asks Muse to wrap its proven model code in that ``main``. The work dir is seeded with
the proven port.

**Hidden check:** a short training run on llm.c's tiny-shakespeare data, cut to 10 training and
5 validation batches in llm.c's own token format. The values checked are:
- the validation loss before training;
- the 10 training losses;
- the validation loss after training.

They must match llm.c's own CPU trainer, ``train_gpt2.c``, within 1e-2 (llm.c's test tolerance).
That trainer uses the same dataloader, shuffle and AdamW; its fixed 40-step loop is cut to 10 when
the reference is built. Both programs run through ``run_train.py``, which lives in the task dir,
out of the model's reach. It runs a program in a scratch dir laid out the way llm.c expects, and
writes the printed losses as f32.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path

from .gpt2 import LLMC_HEADERS, WEIGHTS
from .llmc import ONEMATH_FLAGS

TOKENIZER = "gpt2_tokenizer.bin"
TRAIN = "tiny_shakespeare_train.bin"
VAL = "tiny_shakespeare_val.bin"
B, T = 4, 64
TRAIN_BATCHES, VAL_BATCHES = 10, 5
# llm.c's train_gpt2.c settings, passed to the candidate on its command line
TRAIN_ARGS = f"-b {B} -t {T} -l 0.0001 -v {TRAIN_BATCHES} -m {VAL_BATCHES} -s 20 -g 16"
HEADER_MAGIC, HEADER_VERSION = 20240520, 1

RUN_TRAIN = '''"""Hidden check wrapper (museport J3): run an llm.c GPT-2 trainer on the short
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
train = [float(v) for v in re.findall(r"step\\s+\\d+(?:/\\d+)?: train loss " + num, text)]
if len(val) < 2 or len(train) < 10:
    sys.exit(f"expected 2 val losses and 10 train losses, got {len(val)} and {len(train)}; "
             "output tail: " + text[-1500:])
vals = [val[0], *train[:10], val[1]]
Path(out).write_bytes(struct.pack(f"<{len(vals)}f", *vals))
'''

GOAL = (
    "Port llm.c's fp32 GPT-2 trainer, train_gpt2_fp32.cu, to one SYCL 2020 program, "
    "train_gpt2_fp32.cpp, with zero dpct helpers and the model computed on the SYCL device. "
    "Start from your own proven port: gpt2_train_proven.cpp in the work dir is the model of "
    "this file (forward, backward, AdamW, with oneMath) that you ported earlier. It passed "
    "llm.c's own training test on a CPU, an NVIDIA GPU and an AMD GPU, behind a test main. Keep "
    "that model code; replace its main with llm.c's real main and host code from "
    "train_gpt2_fp32.cu: the same command line (-i -j -o -b -t -l -v -m -s -g), "
    "gpt2_build_from_checkpoint on \"gpt2_124M.bin\", the train and val DataLoaders, the "
    "validation loss every -v steps and at the last step, sampling with the tokenizer "
    "(\"gpt2_tokenizer.bin\") every -s steps, and the same printout ('val loss %f' and "
    "'step %4d/%d: train loss %f ...'). Use llm.c's host headers unchanged, as the CUDA file "
    "does (#include \"llmc/utils.h\", \"llmc/tokenizer.h\", \"llmc/dataloader.h\"; they are "
    "plain C and on the include path). Compile with sources [train_gpt2_fp32.cpp]. A hidden "
    "check runs your program on a short tiny-shakespeare run with fixed arguments and compares "
    "the losses it prints with llm.c's own trainer (atol 1e-2, llm.c's test tolerance). "
    "Success requires compile, verify and a zero-helper scan_helpers of the whole work "
    "directory (path '.') on the same sources; delete migrate's output (delete_file) once your "
    "port is done.")


def write_tokens(src: Path, dest: Path, n: int) -> None:
    """The first ``n`` tokens of an llm.c token file, with its 256-int header."""
    raw = src.read_bytes()
    header = struct.unpack("<256i", raw[:1024])
    if header[0] != HEADER_MAGIC or header[1] != HEADER_VERSION:
        raise ValueError(f"{src}: not an llm.c token file")
    if header[2] < n:
        raise ValueError(f"{src}: {header[2]} tokens, fewer than {n}")
    hdr = list(header)
    hdr[2] = n
    dest.write_bytes(struct.pack("<256i", *hdr) + raw[1024:1024 + 2 * n])


def make_trainer_task(llmc: Path, data: Path, stage_dir: Path, out: Path, *,
                      record: bool = True) -> Path:
    """Write the J3 trainer task. ``data`` holds gpt2_124M.bin, gpt2_tokenizer.bin and
    dev/data/tinyshakespeare/ (llm.c's starter pack). ``record``: run llm.c's CPU trainer once
    now (about 5 minutes) and serve its losses as the reference, instead of rerunning it at
    every verify."""
    if out.exists():
        shutil.rmtree(out)
    (out / "llmc").mkdir(parents=True)
    shutil.copyfile(llmc / "train_gpt2_fp32.cu", out / "train_gpt2_fp32.cu")
    shutil.copyfile(llmc / "train_gpt2.c", out / "train_gpt2.c")
    for h in LLMC_HEADERS:
        shutil.copyfile(llmc / "llmc" / h, out / "llmc" / h)
    for f in (WEIGHTS, TOKENIZER):
        os.symlink((data / f).resolve(), out / f)
    (out / "data").mkdir()
    shakespeare = data / "dev" / "data" / "tinyshakespeare"
    write_tokens(shakespeare / TRAIN, out / "data" / TRAIN, TRAIN_BATCHES * B * T + 1)
    write_tokens(shakespeare / VAL, out / "data" / VAL, VAL_BATCHES * B * T + 1)
    (out / "run_train.py").write_text(RUN_TRAIN)
    (out / "train_args.txt").write_text(TRAIN_ARGS + "\n")
    (out / "seed").mkdir()
    shutil.copyfile(stage_dir / "run1" / "port" / "gpt2_train.cpp",
                    out / "seed" / "gpt2_train_proven.cpp")
    task = {
        "name": "llmc_train_gpt2_fp32",
        "goal": GOAL,
        "cuda_path": "train_gpt2_fp32.cu",
        "include_dirs": ["."],
        # llm.c's CPU trainer, its 40-step loop cut to the 10 steps the check compares
        "reference_build": ["bash", "-c",
                            "mkdir -p build && sed 's/step <= 40/step <= 10/' "
                            "{task_dir}/train_gpt2.c > build/train_gpt2_ref.c && "
                            "gcc -O2 -fopenmp -I{task_dir} build/train_gpt2_ref.c -lm "
                            "-o build/reference"],
        "reference_cmd": "python3 {task_dir}/run_train.py --reference build/reference "
                         "{input} {output}",
        "candidate_cmd": "python3 {task_dir}/run_train.py build/candidate {input} {output}",
        "inputs": ["train_args.txt"],
        "mode": "array",
        "output_format": "f32",
        "base_flags": list(ONEMATH_FLAGS),
        "native_flags": ["-lcublas"],
        "tolerance": 1e-2,
        "rtol": 0.0,
        "museport": {"source": "karpathy/llm.c train_gpt2_fp32.cu", "stage": "train-main",
                     "seed": "seed",
                     "note": f"tiny-shakespeare cut to {TRAIN_BATCHES} train and {VAL_BATCHES} "
                             f"val batches of B={B}, T={T}; the val loss before and after "
                             "training and the 10 train losses checked against llm.c's "
                             "train_gpt2.c within 1e-2"},
    }
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    if record:
        record_reference(out)
    return out / "task.json"


def record_reference(task_dir: Path) -> list[float]:
    """Build and run the task's reference once; store its output as reference.out (with the
    log) and make the task serve it. The recipe stays in the task as ``museport.reference``."""
    import subprocess

    task_file = task_dir / "task.json"
    task = json.loads(task_file.read_text())
    work = task_dir / ".reference"
    if work.exists():
        shutil.rmtree(work)
    (work / "build").mkdir(parents=True)
    def sub(a: str) -> str:
        return a.replace("{task_dir}", str(task_dir.resolve()))

    build = subprocess.run([sub(a) for a in task["reference_build"]], cwd=work,
                           capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"reference build failed: {build.stderr[-2000:]}")
    cmd = (sub(task["reference_cmd"]).replace("{input}", str(task_dir / task["inputs"][0]))
           .replace("{output}", str(task_dir / "reference.out")))
    run = subprocess.run(cmd.split(), cwd=work, capture_output=True, text=True)
    (task_dir / "reference.log").write_text(run.stdout + run.stderr)
    if run.returncode != 0:
        raise RuntimeError(f"reference run failed: {(run.stderr or run.stdout)[-2000:]}")
    shutil.rmtree(work)
    raw = (task_dir / "reference.out").read_bytes()
    values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    task["museport"]["reference"] = {"build": task["reference_build"],
                                     "cmd": task["reference_cmd"], "values": values}
    task["reference_build"] = ["true"]
    task["reference_cmd"] = "cp {task_dir}/reference.out {output}"
    task_file.write_text(json.dumps(task, indent=2) + "\n")
    return values
