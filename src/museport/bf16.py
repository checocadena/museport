"""L1: llm.c's main trainer (bf16, `train_gpt2.cu` and `llmc/`), ported to SYCL in one Muse run.

The hidden check is llm.c's own test, `test_gpt2.cu`, as Muse ports it (`test_gpt2.cpp`), run
by a wrapper outside the work dir that writes, as f32:

1. 1.0 if the test exited 0 and printed "overall okay: 1" (logits, every gradient, 10 losses
   and a bit-exact save, reload and replay, all inside the test), else 0.0;
2. 1.0 if the test source keeps llm.c's tolerances (logits 25.0, loss 0.05 and the 16 gradient
   thresholds, in order), else 0.0, so a port cannot pass by loosening them;
3. the 10 step losses, compared by the check itself with PyTorch's at 0.05 (llm.c's bf16
   tolerance).

The reference is 1.0, 1.0 and PyTorch's 10 losses from llm.c's test. Before any model runs,
llm.c's own CUDA test goes through the same wrapper on an NVIDIA GPU (``make_proof_task``),
and so does a planted bug, which must fail.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

from .gpt2 import DATA_SHA256, STARTER_PACK
from .trainer import STATE, expected_losses

WEIGHTS_BF16 = "gpt2_124M_bf16.bin"
WEIGHTS_BF16_SHA256 = "6661f45628102b4c6e86835d9057b5ba2c024dbf9b81445175e258b7878a1a6f"
VAL = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
# llm.c's bf16 tolerances in test_gpt2.cu: logits, loss, then the 16 parameter gradients
LOGIT_TOL, LOSS_TOL = 25.0, 0.05
GRAD_TOLS = (5e-1, 4e-3, 1e-1, 4e-2, 5e-2, 3.5e-2, 2e-2, 3e-2, 5e-2, 3e-2, 3e-2, 3e-2, 2e-2,
             1e-2, 1e-1, 2e-2)
# the CUDA build of the proof: llm.c's Makefile flags for test_gpt2cu, single GPU, no cuDNN
NATIVE_FLAGS = ["--use_fast_math", "-DENABLE_BF16", "-lcublas", "-lcublasLt",
                "-L/usr/local/cuda/lib64/stubs", "-lnvidia-ml"]
ONEMATH_FLAGS = ["-O2", "-I${ONEMATH_PREFIX}/include", "-Wl,--no-as-needed",
                 "-L${ONEMATH_PREFIX}/lib", "-Wl,-rpath,${ONEMATH_PREFIX}/lib", "-lonemath"]

RUN_TEST = '''"""Hidden check wrapper (museport L1): run llm.c's bf16 GPT-2 test and write, as f32:
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
GRAD_TOLS = __GRAD_TOLS__


def tolerances_kept() -> bool:
    """The test's source keeps llm.c's thresholds: the port's test_gpt2.cpp in the work dir
    (or the task's agent_port/ on a GPU machine), else llm.c's own test_gpt2.cu."""
    for src in (Path("test_gpt2.cpp"), here / "agent_port" / "test_gpt2.cpp",
                here / "test_gpt2.cu"):
        if src.exists():
            break
    text = re.sub(r"/\\*.*?\\*/", "", re.sub(r"//[^\\n]*", "", src.read_text()), flags=re.S)
    val = r"(\\d+\\.?\\d*(?:[eE][-+]?\\d+)?)f?"
    nums = [float(v) for v in re.findall(r"(?<![\\w.])" + val + r"(?![\\w.])", text)]
    # every assignment (or #define) to a logit or loss threshold: llm.c's bf16 value must be
    # among them, and none may be looser
    sets = re.findall(r"(?:#define\\s+)?(\\w+)\\s*=?\\s*" + val + r"\\s*;?", text)
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
# C prints a NaN loss as "nan" or "-nan": match those before plain numbers
num = r"([-+]?(?:nan|inf)|[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)"
ok = re.findall(r"overall okay: (\\d+)", text)
losses = [float(v) for v in re.findall(r"step \\d+: loss " + num, text)]
if not ok or len(losses) < 10:
    sys.exit(f"exit {p.returncode}; expected 'overall okay' and 10 step losses, got {len(ok)} "
             f"and {len(losses)}; output tail: " + (text or p.stderr)[-2500:])
vals = [1.0 if ok[-1] == "1" and p.returncode == 0 else 0.0,
        1.0 if tolerances_kept() else 0.0, *losses[:10]]
Path(out).write_bytes(struct.pack(f"<{len(vals)}f", *vals))
print("overall okay:", ok[-1], "| tolerances kept:", bool(vals[1]), "| losses:", losses[:10])
if vals[0] != 1.0:  # the test's own printout (your program's output), to see which check failed
    failed = [ln for ln in text.splitlines()
              if re.search(r"NOT OK|MISMATCH|Nondeterminism", ln)]  # llm.c's failure lines
    print("failed checks:\\n" + "\\n".join(failed[:40]))
    print("test output tail:\\n" + text[-3000:])
'''.replace("__GRAD_TOLS__", "(" + ",\n             ".join(
    ", ".join(map(repr, GRAD_TOLS[i:i + 8])) for i in range(0, len(GRAD_TOLS), 8)) + ")")

GOAL = (
    "Port llm.c's main GPT-2 trainer from CUDA to SYCL 2020, in one go: train_gpt2.cu, "
    "test_gpt2.cu and the CUDA headers in llmc/ (cuda_common.h, cuda_utils.cuh, cublas_common.h, "
    "encoder.cuh, layernorm.cuh, matmul.cuh, attention.cuh, fused_classifier.cuh, gelu.cuh, "
    "adamw.cuh, global_norm.cuh, zero.cuh). Write train_gpt2.cpp and test_gpt2.cpp (which, like "
    "the CUDA test, defines TESTING and includes train_gpt2.cpp) and SYCL versions of those "
    "headers; keep llm.c's plain-C headers (utils.h, tokenizer.h, dataloader.h, rand.h, "
    "schedulers.h, sampler.h, logger.h, mfu.h, outlier_detector.h) as they are. Keep bf16 mixed "
    "precision as llm.c builds it by default (ENABLE_BF16: bf16 parameters and activations, fp32 "
    "master weights and accumulation; use sycl::ext::oneapi::bfloat16), its stochastic rounding, "
    "and its determinism. Single device: no NCCL, MPI or multi-GPU (keep the single-process path "
    "of zero.cuh), and no cuDNN (llm.c's default is its own attention kernels). cuBLASLt and "
    "cuBLAS matmuls become oneMath BLAS calls (oneapi::math::blas::column_major, as cuBLAS is "
    "column-major; its gemm takes bf16 A and B with an fp32 C, and has no bf16 C, so write to "
    "an fp32 buffer and convert) or your own SYCL kernels, with bias and GELU epilogues as "
    "kernels; "
    "CUB/warp reductions become SYCL group algorithms. The model must compute on the SYCL "
    "device, with zero dpct helpers. Keep llm.c's command line, printout and file formats. "
    "test_gpt2.cpp must keep llm.c's test exactly: the same checks (logits, every gradient, the "
    "10 losses against PyTorch, the save/reload/replay determinism check) and llm.c's own bf16 "
    "tolerances (logits 25.0, loss 0.05, and the 16 gradient thresholds), unchanged, and print "
    "'step %d: loss %f ...' and 'overall okay: %d'. A hidden check runs your test with the "
    "debug state and requires 'overall okay: 1', llm.c's tolerances in its source, and 10 "
    "step losses within 0.05 of PyTorch's. Compile with sources [test_gpt2.cpp] and the flag "
    "-DENABLE_BF16. Success requires compile, verify and a zero-helper scan_helpers of the whole "
    "work directory (path '.') on the same sources; delete migrate's output (delete_file) once "
    "your port is done. This is a large port: write each file early and in full, then fix "
    "compile errors with edit_file rather than rewriting whole files.")
# Without SYCLomatic (it cannot parse llm.c's trainer: nvtx3/nvToolsExt.h is not in its CUDA
# headers): no migrate tool, and the port is written from the CUDA directly.
NO_MIGRATE_GOAL = GOAL.replace(
    "; delete migrate's output (delete_file) once your port is done.",
    ". There is no migration tool: write the SYCL from the CUDA directly, and do not try to read "
    "all 4,900 lines before writing; port file by file, reading each one as you write its port.")

LLMC_CUDA = ("cuda_common.h", "cuda_utils.cuh", "cublas_common.h", "encoder.cuh", "layernorm.cuh",
             "matmul.cuh", "attention.cuh", "fused_classifier.cuh", "gelu.cuh", "adamw.cuh",
             "global_norm.cuh", "zero.cuh")


def _copy_llmc(llmc: Path, out: Path) -> None:
    (out / "llmc").mkdir(parents=True)
    for f in sorted((llmc / "llmc").iterdir()):
        if f.is_file():
            shutil.copyfile(f, out / "llmc" / f.name)
    for name in ("train_gpt2.cu", "test_gpt2.cu"):
        shutil.copyfile(llmc / name, out / name)
    shutil.copyfile(llmc / VAL, out / "tiny_shakespeare_val.bin")
    (out / "run_test.py").write_text(RUN_TEST)
    (out / "test_args.txt").write_text("(the test takes no arguments)\n")


def _reference(test_src: str, out: Path) -> list[float]:
    ref = [1.0, 1.0, *expected_losses(test_src)]
    (out / "reference.out").write_bytes(struct.pack(f"<{len(ref)}f", *ref))
    return ref


def _base(test_src: str, ref: list[float]) -> dict:
    return {
        "cuda_path": "test_gpt2.cu", "include_dirs": ["."],
        "reference_build": ["true"], "reference_cmd": "cp {task_dir}/reference.out {output}",
        "candidate_cmd": "python3 {task_dir}/run_test.py build/candidate {input} {output}",
        "inputs": ["test_args.txt"], "mode": "array", "output_format": "f32",
        "base_flags": [*ONEMATH_FLAGS, "-DENABLE_BF16"], "native_flags": NATIVE_FLAGS,
        "tolerance": LOSS_TOL, "rtol": 0.0,
        "museport": {"item": "L1", "source": "karpathy/llm.c f1e2ace test_gpt2.cu, train_gpt2.cu "
                     "and llmc/ (bf16)",
                     "reference": {"values": ref, "source": "1.0 for 'overall okay: 1', 1.0 for "
                                   "llm.c's tolerances kept, then expected_losses from llm.c's "
                                   "test_gpt2.cu (PyTorch)"}}}


SHELL_NOTE = (
    " You have a `run` tool (bash in the work dir): build and run your own test programs and "
    "your test_gpt2 itself with it. The files the test reads are in {task_dir}: "
    "gpt2_124M_bf16.bin, gpt2_124M_debug_state.bin and tiny_shakespeare_val.bin (the test opens "
    "dev/data/tinyshakespeare/tiny_shakespeare_val.bin); link them into your work dir. Do not "
    "change anything in {task_dir}. The hidden check still decides success.")


def make_task(llmc: Path, data: Path, out: Path, *, no_migrate: bool = False,
              shell: bool = False) -> Path:
    """The task Muse ports in one run, on the CPU device. ``data`` holds the bf16 weights and
    the debug state; they are linked, not copied. ``no_migrate``: the goal for a run without
    SYCLomatic."""
    if out.exists():
        shutil.rmtree(out)
    _copy_llmc(llmc, out)
    for f in (WEIGHTS_BF16, STATE):
        (out / f).symlink_to((data / f).resolve())
    test_src = (llmc / "test_gpt2.cu").read_text()
    ref = _reference(test_src, out)
    goal = NO_MIGRATE_GOAL if no_migrate else GOAL
    if shell:
        goal += SHELL_NOTE.replace("{task_dir}", str(out.resolve()))
    task = {"name": "llmc_test_gpt2_bf16", "goal": goal, **_base(test_src, ref)}
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return out / "task.json"


# Run 3 continues run 2's port: the work dir starts with run 2's final files (``seed``), and the
# goal says where run 2 stopped, in a note taken from run 2's own trace.
CONTINUE_NOTE = (
    " You are continuing your own earlier run, which ran out of turns: the work dir holds your "
    "port as you left it (train_gpt2.cpp with train_part1-3.cpp, test_gpt2.cpp, llmc/*.h, and "
    "your dbg_*.cpp debug programs). Where it stood: it compiles with zero helpers; the forward "
    "pass matches (your test's step-1 loss was 5.250530 against PyTorch's 5.270009); training "
    "goes to NaN from step 2 because something in the backward pass writes out of bounds into "
    "the gradient buffer (your last bisection, with a sentinel in grads_memory: 738 bad values "
    "after attention's backward, already present before its softmax backward step). Continue "
    "from these files; delete the dbg_*.cpp files once the port passes. When the test fails, "
    "the check now shows the tail of its own printout.")


def make_continue_task(llmc: Path, data: Path, seed: Path, out: Path) -> Path:
    """Run 3: the no-SYCLomatic task, starting from ``seed`` (run 2's final files)."""
    make_task(llmc, data, out, no_migrate=True)
    shutil.copytree(seed, out / "seed")
    task = json.loads((out / "task.json").read_text())
    task["goal"] += CONTINUE_NOTE
    task["museport"]["seed"] = "seed"
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return out / "task.json"


def make_proof_task(llmc: Path, out: Path, *, name: str = "llmc_test_gpt2_bf16_proof",
                    patch: tuple[str, str, str] | None = None) -> Path:
    """llm.c's own CUDA test, built with nvcc and run twice through the wrapper on an NVIDIA
    GPU (``record_native``; no SYCL port). ``patch`` = (file, old, new) plants a bug in llm.c's
    CUDA first. The weights and the debug state are fetched on the machine."""
    if out.exists():
        shutil.rmtree(out)
    _copy_llmc(llmc, out)
    if patch:
        rel, old, new = patch
        src = (out / rel).read_text()
        if src.count(old) != 1:
            raise ValueError(f"planted bug: {old!r} must occur once in {rel}")
        (out / rel).write_text(src.replace(old, new))
    test_src = (llmc / "test_gpt2.cu").read_text()
    task = {"name": name, **_base(test_src, _reference(test_src, out)), "record_native": True,
            "benchmark_runs": 0,
            "fetch": [{"path": WEIGHTS_BF16, "url": f"{STARTER_PACK}{WEIGHTS_BF16}?download=true",
                       "sha256": WEIGHTS_BF16_SHA256},
                      {"path": STATE, "url": f"{STARTER_PACK}{STATE}?download=true",
                       "sha256": DATA_SHA256[STATE]}]}
    task["museport"]["planted"] = list(patch) if patch else None
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return out / "task.json"


def make_gpu_task(evidence_dir: Path, dest: Path, model: str, vendor: str = "nvidia",
                  run: str = "run3", onemath_patches: tuple[Path, ...] = ()) -> Path:
    """A ``remote-test`` task for the L1 port: the task (wrapper, reference), Muse's final files
    from ``run`` (``test_gpt2.cpp`` and what it includes), unchanged, the weights and debug state
    fetched on the machine, and oneMath built there for ``vendor``. On NVIDIA the native build
    is llm.c's own CUDA test, run through the same wrapper."""
    from .gpt2 import ONEMATH_BACKENDS, onemath_gpu_setup

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(evidence_dir / "task", dest,
                    ignore=shutil.ignore_patterns("task-*.json", "run_test-before-*"))
    shutil.copytree(evidence_dir / run / "port", dest / "agent_port")
    backend = ONEMATH_BACKENDS[vendor]
    setup = f"onemath_{backend['backend']}_setup.sh"
    for p in onemath_patches:
        shutil.copyfile(p, dest / p.name)
    (dest / setup).write_text(onemath_gpu_setup(vendor, tuple(p.name for p in onemath_patches)))
    task = json.loads((dest / "task.json").read_text())
    task.pop("goal", None)
    task["sycl_sources"] = ["agent_port/test_gpt2.cpp"]
    task["gpu_setup"] = setup
    task["benchmark_runs"] = 0
    task["fetch"] = [{"path": WEIGHTS_BF16, "url": f"{STARTER_PACK}{WEIGHTS_BF16}?download=true",
                      "sha256": WEIGHTS_BF16_SHA256},
                     {"path": STATE, "url": f"{STARTER_PACK}{STATE}?download=true",
                      "sha256": DATA_SHA256[STATE]}]
    task["port_origin"] = (f"agent-made: {model} via museport loop, {evidence_dir.name}/{run} "
                           "(continuing run2); its final files, unchanged")
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"
