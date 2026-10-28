"""K1: train GPT-2 124M for one epoch of tiny-shakespeare and compare the loss curves of llm.c's
CUDA trainer and Muse's SYCL trainer, on NVIDIA and AMD.

Two kinds of GPU task, both run with ``oneapi-agent remote-test``:

- **record** (NVIDIA): llm.c's own ``train_gpt2_fp32.cu`` is built with ``nvcc`` and run twice
  (the toolkit's ``record_native``). The first run's losses become the reference curve; the
  difference between the two runs says how deterministic llm.c's CUDA trainer is.
- **sycl** (NVIDIA or AMD): Muse's J3 trainer (``evidence/j3-trainer/run1/port``), unchanged,
  checked against that recording. On NVIDIA it is also compared with llm.c's CUDA live.

The wrapper ``run_curve.py`` writes the losses as f32 (every val loss, then every train loss)
for the check, and prints them, with each step's time, as one JSON line. The toolkit keeps that
stdout in the results, so the curves come back with the evidence.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from .gpt2 import DATA_SHA256, ONEMATH_BACKENDS, STARTER_PACK, WEIGHTS, onemath_gpu_setup
from .llmc import ONEMATH_FLAGS
from .port import last_compiled_sources
from .trainer import LLMC_HEADERS, TOKENIZER, TOKENIZER_SHA256, TRAIN, VAL

# One epoch of tiny-shakespeare at B=4, T=256: llm.c's trainer runs train_num_batches steps.
# Val loss every 25 steps over 10 batches; sampling only at the last step (16 tokens).
B, T = 4, 256
CURVE_ARGS = f"-b {B} -t {T} -l 0.0001 -v 25 -m 10 -s 100000 -g 16"

RUN_CURVE = '''"""Loss-curve wrapper (museport K1): run an llm.c GPT-2 trainer for one epoch of
tiny-shakespeare, write every val loss then every train loss as f32, and print them, with each
step's time in ms, as one JSON line.

usage: run_curve.py <binary> <args file> <output>"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
binary, args_file, out = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
args = Path(args_file).read_text().split()
with tempfile.TemporaryDirectory() as run:
    run = Path(run)
    data = run / "dev" / "data" / "tinyshakespeare"
    data.mkdir(parents=True)
    for name in ("gpt2_124M.bin", "gpt2_tokenizer.bin"):
        os.symlink(here / name, run / name)
    for name in ("tiny_shakespeare_train.bin", "tiny_shakespeare_val.bin"):
        os.symlink(here / "data" / name, data / name)
    p = subprocess.run([str(binary), *args], cwd=run, capture_output=True, text=True,
                       timeout=5400)
text = p.stdout
if p.returncode != 0:
    sys.exit(f"exit {p.returncode}: " + (p.stderr or text)[-2000:])
num = r"([-+0-9.eE]+|nan|-?inf)"
val = [float(v) for v in re.findall(r"val loss " + num, text)]
steps = re.findall(r"step\\s+\\d+/\\d+: train loss " + num + r" \\(" + num + " ms", text)
train = [float(loss) for loss, _ in steps]
ms = [float(t) for _, t in steps]
if len(val) < 2 or len(train) < 2:
    sys.exit(f"expected val and train losses, got {len(val)} and {len(train)}; output tail: "
             + text[-1500:])
Path(out).write_bytes(struct.pack(f"<{len(val) + len(train)}f", *val, *train))
print(json.dumps({"val": val, "train": train,
                  "ms_per_step_median": sorted(ms)[len(ms) // 2]}, separators=(",", ":")))
'''


def _base(llmc: Path, dest: Path, vendor: str) -> dict[str, object]:
    """The files both tasks share: llm.c's CUDA trainer and headers, the full tiny-shakespeare
    data, the args and the wrapper. Weights and tokenizer are fetched on the machine."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "llmc").mkdir(parents=True)
    shutil.copyfile(llmc / "train_gpt2_fp32.cu", dest / "train_gpt2_fp32.cu")
    for h in LLMC_HEADERS:
        shutil.copyfile(llmc / "llmc" / h, dest / "llmc" / h)
    (dest / "data").mkdir()
    for name in (TRAIN, VAL):
        shutil.copyfile(llmc / "dev" / "data" / "tinyshakespeare" / name, dest / "data" / name)
    (dest / "curve_args.txt").write_text(CURVE_ARGS + "\n")
    (dest / "run_curve.py").write_text(RUN_CURVE)
    sha = {**DATA_SHA256, TOKENIZER: TOKENIZER_SHA256}
    return {
        "cuda_path": "train_gpt2_fp32.cu", "include_dirs": ["."],
        "reference_build": ["true"],
        "reference_cmd": "cp {task_dir}/reference.out {output}",
        "candidate_cmd": "python3 {task_dir}/run_curve.py build/candidate {input} {output}",
        "inputs": ["curve_args.txt"], "mode": "array", "output_format": "f32",
        "tolerance": 1e-2, "rtol": 0.0,
        "native_flags": ONEMATH_BACKENDS[vendor]["native_flags"],
        "fetch": [{"path": f, "url": f"{STARTER_PACK}{f}?download=true", "sha256": sha[f]}
                  for f in (WEIGHTS, TOKENIZER)],
        "benchmark_runs": 0,
        "museport": {"item": "K1", "args": CURVE_ARGS,
                     "note": "one epoch of tiny-shakespeare; f32 output = every val loss, "
                             "then every train loss"},
    }


def make_record_task(llmc: Path, dest: Path) -> Path:
    """llm.c's CUDA trainer, recorded twice on an NVIDIA GPU (no SYCL port is built)."""
    task = {"name": "llmc_gpt2_curve_record", **_base(llmc, dest, "nvidia"),
            "record_native": True}
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"


def make_sycl_task(llmc: Path, trainer_run: Path, recording: Path, dest: Path, model: str,
                   vendor: str) -> Path:
    """Muse's trainer (``trainer_run``: its port as last compiled) on ``vendor``'s GPU,
    checked against the recorded CUDA curve (``recording``, the record task's case0.out)."""
    task = {"name": f"llmc_gpt2_curve_sycl_{vendor}", **_base(llmc, dest, vendor)}
    srcs = last_compiled_sources(trainer_run / "trace.jsonl")
    if not srcs:
        raise ValueError(f"{trainer_run}: the model never compiled")
    shutil.copytree(trainer_run / "port", dest / "agent_port")
    shutil.copyfile(recording, dest / "reference.out")
    setup = f"onemath_{ONEMATH_BACKENDS[vendor]['backend']}_setup.sh"
    (dest / setup).write_text(onemath_gpu_setup(vendor))
    task.update({"sycl_sources": [f"agent_port/{s}" for s in srcs],
                 "base_flags": list(ONEMATH_FLAGS), "gpu_setup": setup,
                 "port_origin": f"agent-made: {model} via museport loop, "
                                f"{trainer_run.parent.name}/{trainer_run.name}; sources as "
                                "last compiled by the agent, unchanged"})
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"


def load_curve(results: Path) -> dict[str, object]:
    """The curve(s) a run printed: from a gpu-test results dir, every verify step's candidate
    and reference stdout that is a JSON curve line."""
    curves: dict[str, object] = {}
    for f in sorted(results.glob("*verify*.json")) + sorted(results.glob("native_repeat.json")):
        for case in json.loads(f.read_text()).get("evidence", {}).get("cases", []):
            for side in ("candidate", "reference"):
                out = (case.get(side) or {}).get("stdout", "").strip()
                if out.startswith("{"):
                    curves[f"{f.stem}:{side}"] = json.loads(out.splitlines()[-1])
    return curves


def read_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))
