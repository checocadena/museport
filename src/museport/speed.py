"""L1 speed: Muse's SYCL port of llm.c's bf16 trainer against llm.c's own CUDA trainer.

Both run llm.c's command line for SPEED_ARGS (30 training steps, B=4, T=1024, tiny-shakespeare,
validation only at the first and last step, no sampling) through ``run_speed.py``. It writes
the 30 losses as f32, so the two runs are also checked against each other, and prints
``kernel_ms=<median ms per step>``: llm.c's own step timer over steps 6-30, which gpu-test's
benchmark reads. Model loading and the first steps' warm-up are not counted.

1. ``make_record_task``: llm.c's CUDA trainer, built with nvcc and run twice on an NVIDIA GPU
   (``record_native``); its losses become the reference.
2. ``make_speed_task``: Muse's trainer (the port's ``train_gpt2.cpp``) against that recording;
   on NVIDIA also against llm.c's trainer live, and both are timed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .bf16 import NATIVE_FLAGS, ONEMATH_FLAGS, WEIGHTS_BF16, WEIGHTS_BF16_SHA256
from .gpt2 import STARTER_PACK
from .trainer import TOKENIZER, TOKENIZER_SHA256

STEPS = 30
# validation only at the first and last step (-v 0 would divide by zero in llm.c), 2 batches
SPEED_ARGS = f"-b 4 -t 1024 -x {STEPS} -v 1000 -m 2 -s 0 -n 0 -h 0"
# llm.c's trainer runs in bf16 with stochastic rounding: two different implementations drift
# apart over training steps. The tolerance is set from the recording's own spread and the CPU
# test (worst 0.036 over 10 steps of the debug batch); it is a cross-check, not the grade.
LOSS_TOL = 0.1

RUN_SPEED = '''"""Speed wrapper (museport L1): run an llm.c bf16 trainer for the args in
<args file>, write its step losses as f32, and print kernel_ms=<median ms per step, steps 6
on> for gpu-test.

usage: run_speed.py <binary> <args file> <output>"""
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
binary, args, out = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
with tempfile.TemporaryDirectory() as run:
    for name in ("gpt2_124M_bf16.bin", "gpt2_tokenizer.bin"):
        os.symlink(here / name, Path(run) / name)
    data = Path(run) / "dev" / "data" / "tinyshakespeare"
    data.mkdir(parents=True)
    for name in ("tiny_shakespeare_train.bin", "tiny_shakespeare_val.bin"):
        os.symlink(here / name, data / name)
    p = subprocess.run([str(binary), *Path(args).read_text().split()], cwd=run,
                       capture_output=True, text=True, timeout=3000)
text = p.stdout
steps = re.findall(r"step +(\\d+)/\\d+ \\| loss ([-+0-9.eEnaif]+).*?\\| ([0-9.]+) ms", text)
if p.returncode != 0 or len(steps) < 10:
    sys.exit(f"exit {p.returncode}; expected per-step lines 'step N/M | loss L ... | X ms', got "
             f"{len(steps)}; output tail: " + (text or p.stderr)[-2500:])
losses = [float(s[1]) for s in steps]
ms = [float(s[2]) for s in steps]
median = statistics.median(ms[5:]) if len(ms) > 5 else statistics.median(ms)
Path(out).write_bytes(struct.pack(f"<{len(losses)}f", *losses))
print(json.dumps({"steps": len(steps), "ms_per_step_median": median, "losses": losses}))
print(f"kernel_ms={median}")
'''


def _common(llmc: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "llmc").mkdir(parents=True)
    for f in sorted((llmc / "llmc").iterdir()):
        if f.is_file():
            shutil.copyfile(f, out / "llmc" / f.name)
    shutil.copyfile(llmc / "train_gpt2.cu", out / "train_gpt2.cu")
    for name in ("tiny_shakespeare_train.bin", "tiny_shakespeare_val.bin"):
        shutil.copyfile(llmc / "dev" / "data" / "tinyshakespeare" / name, out / name)
    (out / "run_speed.py").write_text(RUN_SPEED)
    (out / "args.txt").write_text(SPEED_ARGS + "\n")


def _task(name: str) -> dict:
    return {
        "name": name, "cuda_path": "train_gpt2.cu", "include_dirs": ["."],
        "reference_build": ["true"],
        "candidate_cmd": "python3 {task_dir}/run_speed.py build/candidate {input} {output}",
        "inputs": ["args.txt"], "mode": "array", "output_format": "f32",
        "base_flags": [*ONEMATH_FLAGS, "-DENABLE_BF16"], "native_flags": NATIVE_FLAGS,
        "tolerance": LOSS_TOL, "rtol": 0.0,
        "fetch": [{"path": WEIGHTS_BF16, "url": f"{STARTER_PACK}{WEIGHTS_BF16}?download=true",
                   "sha256": WEIGHTS_BF16_SHA256},
                  {"path": TOKENIZER, "url": f"{STARTER_PACK}{TOKENIZER}?download=true",
                   "sha256": TOKENIZER_SHA256}],
        "museport": {"item": "L1 speed", "args": SPEED_ARGS,
                     "note": "llm.c's bf16 trainer, 30 steps; kernel_ms is the median of "
                             "llm.c's own per-step timer over steps 6-30"}}


def make_record_task(llmc: Path, out: Path) -> Path:
    """llm.c's CUDA trainer, recorded twice on an NVIDIA GPU: the reference losses."""
    _common(llmc, out)
    task = {**_task("llmc_train_gpt2_bf16_speed_record"), "reference_cmd": "true",
            "record_native": True, "benchmark_runs": 0}
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return out / "task.json"


def make_speed_task(llmc: Path, port: Path, recorded: Path, out: Path, model: str, *,
                    vendor: str = "nvidia", onemath_patches: tuple[Path, ...] = (),
                    runs: int = 3) -> Path:
    """Muse's trainer (``port``: the run's final files) against the recording (``recorded``:
    the record task's results/recorded/ directory), timed ``runs`` times, with oneMath built on
    the machine for ``vendor``."""
    from .gpt2 import ONEMATH_BACKENDS, onemath_gpu_setup

    _common(llmc, out)
    shutil.copytree(port, out / "agent_port", ignore=shutil.ignore_patterns("build", "verify_out"))
    shutil.copyfile(recorded / "case0.out", out / "recorded_losses.out")
    for p in onemath_patches:
        shutil.copyfile(p, out / p.name)
    backend = ONEMATH_BACKENDS[vendor]
    setup = f"onemath_{backend['backend']}_setup.sh"
    (out / setup).write_text(onemath_gpu_setup(vendor, tuple(p.name for p in onemath_patches)))
    task = {**_task("llmc_train_gpt2_bf16_speed"),
            "reference_cmd": "cp {task_dir}/recorded_losses.out {output}",
            "sycl_sources": ["agent_port/train_gpt2.cpp"], "gpu_setup": setup,
            "benchmark_runs": runs,
            "port_origin": f"agent-made: {model} via museport loop; its final files, unchanged"}
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return out / "task.json"
