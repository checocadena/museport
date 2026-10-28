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


# ----------------------------------------------------------------------------- J3 step 3

STATE = "gpt2_124M_debug_state.bin"
RUN_TEST = '''"""Hidden check wrapper (museport J3): run llm.c's GPT-2 test and write, as f32,
whether it printed "overall okay: 1" (1.0 or 0.0), then the loss of each of its 10 steps.

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
with tempfile.TemporaryDirectory() as run:
    for name in ("gpt2_124M.bin", "gpt2_124M_debug_state.bin"):
        os.symlink(here / name, Path(run) / name)
    p = subprocess.run([str(binary)], cwd=run, capture_output=True, text=True, timeout=1500)
text = p.stdout
num = r"([-+0-9.eE]+|nan|-?inf)"
ok = re.findall(r"overall okay: (\\d+)", text)
losses = [float(v) for v in re.findall(r"step \\d+: loss " + num, text)]
if not ok or len(losses) < 10:
    sys.exit(f"exit {p.returncode}; expected 'overall okay' and 10 step losses, got {len(ok)} "
             f"and {len(losses)}; output tail: " + (text or p.stderr)[-1500:])
vals = [1.0 if ok[-1] == "1" and p.returncode == 0 else 0.0, *losses[:10]]
Path(out).write_bytes(struct.pack(f"<{len(vals)}f", *vals))
'''

TEST_GOAL = (
    "Port llm.c's GPT-2 test, test_gpt2_fp32.cu, to SYCL 2020 as test_gpt2_fp32.cpp, with zero "
    "dpct helpers. Like the CUDA test it starts with `#define TESTING` and "
    "`#include \"train_gpt2_fp32.cpp\"`: train_gpt2_fp32.cpp in the work dir is your own SYCL "
    "port of llm.c's trainer, already proven against llm.c (its training losses match llm.c's "
    "own trainer). Change train_gpt2_fp32.cpp only to wrap its main in `#ifndef TESTING` ... "
    "`#endif`, as train_gpt2_fp32.cu does; leave the rest of it as it is. Port the test's main "
    "(device setup, loading gpt2_124M_debug_state.bin, the logits, loss and gradient "
    "comparisons and the 10 training steps) using the trainer's functions, with the same "
    "printout ('step %d: loss %f ...', 'overall okay: %d'). Compile with sources "
    "[test_gpt2_fp32.cpp]. A hidden check runs your test and requires 'overall okay: 1' and "
    "10 step losses that match PyTorch's within 1e-2 (llm.c's test tolerance). Success requires "
    "compile, verify and a zero-helper scan_helpers of the whole work directory (path '.') on "
    "the same sources; delete migrate's output (delete_file) once your port is done.")

# Run 1 of the test port spent all 20 turns reading the 1,117-line trainer in slices and never
# wrote the test. Run 2 adds what the CUDA test uses from it (read off test_gpt2_fp32.cu and
# the trainer's own declarations), and asks for the file early.
TEST_API_HINT = (
    " What the test needs from train_gpt2_fp32.cpp: the device queue is the global "
    "`static sycl::queue *g_q` (the trainer's main sets it: `sycl::queue q{ "
    "sycl::property::queue::in_order{} }; g_q = &q;`; `qq()` returns it); the model's tensors "
    "are device USM (sycl::malloc_device), so copy with qq().memcpy(dst, src, bytes).wait(); "
    "the functions are gpt2_build_from_checkpoint(GPT2*, path), gpt2_forward(GPT2*, int* "
    "inputs, int* targets, B, T), gpt2_zero_grad(GPT2*), gpt2_backward(GPT2*), "
    "gpt2_update(GPT2*, lr, beta1, beta2, eps, weight_decay, t) and gpt2_free(GPT2*); the "
    "fields are model.acts.output, model.grads.<tensor>, model.grads_memory, model.mean_loss, "
    "model.config.*, model.param_sizes and model.num_parameters. main is the last function in "
    "train_gpt2_fp32.cpp: add the guard with two small edit_file calls (put `#ifndef TESTING` "
    "before the `int main(int argc, char *argv[]) {` line, and `#endif` after main's closing "
    "brace at the end of the file), not by rewriting the file. You do not need to read the "
    "whole trainer: write test_gpt2_fp32.cpp within your first few turns, then compile and "
    "verify, and fix what the compiler and verify report.")


def expected_losses(test_source: str) -> list[float]:
    """PyTorch's 10 losses, as written into llm.c's test."""
    import re

    m = re.search(r"expected_losses\[10\]\s*=\s*\{([^}]*)\}", test_source)
    if not m:
        raise ValueError("no expected_losses[10] in the test source")
    vals = [float(v.rstrip("f")) for v in re.findall(r"[-0-9.eE]+f?", m.group(1))]
    if len(vals) != 10:
        raise ValueError(f"expected 10 losses, found {len(vals)}")
    return vals


def make_test_task(llmc: Path, data: Path, trainer_cpp: Path, out: Path, *,
                   api_hint: bool = True) -> Path:
    """Write the J3 test task: llm.c's test_gpt2_fp32.cu, ported on top of the proven trainer
    ``trainer_cpp``; ``data`` holds gpt2_124M.bin and gpt2_124M_debug_state.bin."""
    if out.exists():
        shutil.rmtree(out)
    (out / "llmc").mkdir(parents=True)
    test_src = (llmc / "test_gpt2_fp32.cu").read_text()
    (out / "test_gpt2_fp32.cu").write_text(test_src)
    shutil.copyfile(llmc / "train_gpt2_fp32.cu", out / "train_gpt2_fp32.cu")
    for h in LLMC_HEADERS:
        shutil.copyfile(llmc / "llmc" / h, out / "llmc" / h)
    for f in (WEIGHTS, STATE):
        os.symlink((data / f).resolve(), out / f)
    (out / "run_test.py").write_text(RUN_TEST)
    (out / "test_args.txt").write_text("(the test takes no arguments)\n")
    (out / "seed").mkdir()
    shutil.copyfile(trainer_cpp, out / "seed" / "train_gpt2_fp32.cpp")
    ref = [1.0, *expected_losses(test_src)]
    (out / "reference.out").write_bytes(struct.pack(f"<{len(ref)}f", *ref))
    task = {
        "name": "llmc_test_gpt2_fp32",
        "goal": TEST_GOAL + (TEST_API_HINT if api_hint else ""),
        "cuda_path": "test_gpt2_fp32.cu",
        "include_dirs": ["."],
        "reference_build": ["true"],
        "reference_cmd": "cp {task_dir}/reference.out {output}",
        "candidate_cmd": "python3 {task_dir}/run_test.py build/candidate {input} {output}",
        "inputs": ["test_args.txt"],
        "mode": "array",
        "output_format": "f32",
        "base_flags": list(ONEMATH_FLAGS),
        "native_flags": ["-lcublas"],
        "tolerance": 1e-2,
        "rtol": 0.0,
        "museport": {"source": "karpathy/llm.c test_gpt2_fp32.cu", "stage": "test",
                     "seed": "seed",
                     "note": "llm.c's own test: 'overall okay: 1' (logits, loss and gradients "
                             "against PyTorch's debug state, inside the test) and its 10 "
                             "training losses against PyTorch's within 1e-2",
                     "reference": {"values": ref,
                                   "source": "1.0 for 'overall okay: 1', then expected_losses "
                                             "from llm.c's test_gpt2_fp32.cu (PyTorch)"}},
    }
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return out / "task.json"


# ----------------------------------------------------------------------------- J3 step 4

TOKENIZER_SHA256 = "6f3abc21e444e4e8300e225f4e03da48ea121cf17e30f67009b8dad7a66c2f13"


def make_j3_gpu_task(evidence_dir: Path, dest: Path, model: str,
                     vendor: str = "nvidia", run: str = "run1") -> Path:
    """A ``remote-test`` task for a J3 port (``evidence/j3-trainer`` or ``evidence/j3-test``):
    its task (wrapper, recorded reference), Muse's port as last compiled, the large files
    fetched on the machine with their sha256, and oneMath built there for the vendor. On
    NVIDIA the native build is llm.c's own CUDA program, run through the same wrapper. ``run``
    names the successful run's folder (``run3`` for the test)."""
    from .gpt2 import DATA_SHA256, ONEMATH_BACKENDS, STARTER_PACK, onemath_gpu_setup
    from .port import last_compiled_sources

    srcs = last_compiled_sources(evidence_dir / run / "trace.jsonl")
    if not srcs:
        raise ValueError(f"{evidence_dir}: the model never compiled")
    if dest.exists():
        shutil.rmtree(dest)
    big = (WEIGHTS, STATE, TOKENIZER)
    shutil.copytree(evidence_dir / "task", dest, ignore=shutil.ignore_patterns(*big, "seed"))
    shutil.copytree(evidence_dir / run / "port", dest / "agent_port")
    backend = ONEMATH_BACKENDS[vendor]
    setup = f"onemath_{backend['backend']}_setup.sh"
    (dest / setup).write_text(onemath_gpu_setup(vendor))
    task = json.loads((dest / "task.json").read_text())
    needs = (WEIGHTS, STATE) if task["museport"]["stage"] == "test" else (WEIGHTS, TOKENIZER)
    sha = {**DATA_SHA256, TOKENIZER: TOKENIZER_SHA256}
    task["sycl_sources"] = [f"agent_port/{s}" for s in srcs]
    task["native_flags"] = backend["native_flags"]
    task["gpu_setup"] = setup
    task["fetch"] = [{"path": f, "url": f"{STARTER_PACK}{f}?download=true", "sha256": sha[f]}
                     for f in needs]
    task["port_origin"] = (f"agent-made: {model} via museport loop, {evidence_dir.name}/{run}; "
                           "sources as last compiled by the agent, unchanged")
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"


# J3 step 5: llm.c-sycl's own Makefile on a GPU. The task's gpu_setup (sourced before gpu-test,
# its output archived as setup.log) runs `make` for the machine's GPU; the candidate wrapper then
# runs the test binary the Makefile built in place of gpu-test's own build. llm.c's CUDA (the
# native build) still runs as itself. Names in the setup are prefixed MP_ (the toolkit runner
# owns D, HERE, ...).
MAKEFILE_SETUP = r"""
# museport J3: build llm.c-sycl with its own Makefile, for this GPU
MP_TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP_CXX="${SYCL_CXX:-$(command -v icpx || command -v clang++)}"
if command -v nvidia-smi >/dev/null 2>&1; then
  MP_TARGET="TARGET=nvidia"
else
  MP_ARCH="$( (rocm_agent_enumerator 2>/dev/null; rocminfo 2>/dev/null \
    | awk '/Name: *gfx/{print $2}') | grep -v gfx000 | head -1)"
  MP_TARGET="TARGET=amd AMD_ARCH=$MP_ARCH"
fi
command -v make >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq make; }
echo "museport: make -C llmc_sycl SYCL_CXX=$MP_CXX $MP_TARGET"
make -C "$MP_TASK/llmc_sycl" SYCL_CXX="$MP_CXX" $MP_TARGET \
  && echo "museport: Makefile build ok" || echo "museport: Makefile build FAILED"
"""

RUN_MAKE_TEST = '''"""Hidden check wrapper (museport J3, Makefile on a GPU): gpu-test's own
SYCL build (build/sycl) is replaced by the test binary llm.c-sycl's Makefile built (gpu_setup);
llm.c's CUDA (build/native) runs as itself. Both go through run_test.py.

usage: run_make_test.py <binary> <input> <output>"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
binary = sys.argv[1]
if Path(binary).name != "native":
    binary = str(here / "llmc_sycl" / "test_gpt2fp32sycl")
    if not Path(binary).exists():
        sys.exit("the Makefile build is missing: see setup.log")
sys.exit(subprocess.run([sys.executable, str(here / "run_test.py"), binary, *sys.argv[2:]])
         .returncode)
'''


def make_j3_makefile_task(evidence_dir: Path, repo: Path, dest: Path, model: str,
                          vendor: str = "nvidia") -> Path:
    """The J3 test's GPU task (``make_j3_gpu_task``, run3), with llm.c-sycl (``repo``: its
    sources, headers and Makefile, not dev/sycl) built by its own Makefile in gpu_setup, and
    that build graded in place of gpu-test's."""
    path = make_j3_gpu_task(evidence_dir, dest, model, vendor, run="run3")
    task = json.loads(path.read_text())
    shutil.copytree(repo, dest / "llmc_sycl", ignore=shutil.ignore_patterns(
        "dev", "*.bin", "train_gpt2fp32sycl", "test_gpt2fp32sycl", "*.md"))
    setup = dest / task["gpu_setup"]
    setup.write_text(setup.read_text() + MAKEFILE_SETUP)
    (dest / "run_make_test.py").write_text(RUN_MAKE_TEST)
    task["name"] = "llmc_test_gpt2_fp32_makefile"
    task["candidate_cmd"] = "python3 {task_dir}/run_make_test.py build/candidate {input} {output}"
    task["port_origin"] = ("llm.c-sycl as committed, built by its own Makefile on the GPU "
                           f"(gpu_setup); test written by {model}, evidence/j3-test/run3")
    path.write_text(json.dumps(task, indent=2) + "\n")
    return path
