"""GPT-2 fp32 training port (plan G): the stage tasks and their hidden checks.

Muse ports llm.c's ``train_gpt2_fp32.cu`` in stages: 1, the forward pass; 2, the backward
pass (the gradients of all parameters). The CUDA
harness is the training file, unchanged, with a contract ``main`` in place of its own (llm.c's
``TESTING`` switch). The hidden reference is PyTorch's own result for llm.c's debug batch, read
from ``gpt2_124M_debug_state.bin``. The check uses llm.c's own tolerance, 1e-2 absolute, the
one ``test_gpt2_fp32.cu`` uses.

The weights (``gpt2_124M.bin``, about 500 MB) and the debug state (about 550 MB) are not
committed: the task links them from ``--data``. llm.c's ``dev/download_starter_pack.sh``
fetches them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .llmc import ONEMATH_FLAGS

WEIGHTS = "gpt2_124M.bin"
STATE = "gpt2_124M_debug_state.bin"
LLMC_HEADERS = ("utils.h", "tokenizer.h", "dataloader.h", "rand.h")
TOLERANCE = 1e-2  # test_gpt2_fp32.cu: fabsf(a - b) <= 1e-2 for logits, loss and gradients
# llm.c's CPU test (test_gpt2.c, check_tensor) allows 2e-2 for gradients. At 1e-2, llm.c's own
# CPU train_gpt2.c fails 1 of the 124,475,904 gradients (|0.9520 - 0.9635| = 0.0115, in fcw),
# so a 1e-2 check would reject llm.c itself; stage 2 uses llm.c's CPU gradient tolerance.
GRAD_TOLERANCE = 2e-2

REFERENCE = r"""// Hidden reference for GPT-2 stage 1 (forward): PyTorch's logits and loss
// for llm.c's debug batch, read from gpt2_124M_debug_state.bin, written as raw
// float32: B*T*V logits, then the loss.
// usage: reference <debug_state.bin> <output> <gpt2_124M.bin>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void must(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
int main(int argc, char** argv) {
    must(argc == 4, "usage: reference <debug_state.bin> <output> <gpt2_124M.bin>");
    int mh[256], sh[256];
    FILE* m = std::fopen(argv[3], "rb");
    must(m && std::fread(mh, 4, 256, m) == 256, "weights header");
    std::fclose(m);
    must(mh[0] == 20240326, "bad weights magic");
    FILE* s = std::fopen(argv[1], "rb");
    must(s && std::fread(sh, 4, 256, s) == 256, "state header");
    must(sh[0] == 20240327 && sh[1] == 2, "bad state magic or version");
    const long B = sh[2], T = sh[3], V = mh[3];
    std::vector<int> xy(2 * B * T);
    must(std::fread(xy.data(), 4, xy.size(), s) == xy.size(), "x, y");
    std::vector<float> out(B * T * V + 1);
    must(std::fread(out.data(), 4, out.size(), s) == out.size(), "logits, loss");
    std::fclose(s);
    FILE* o = std::fopen(argv[2], "wb");
    must(o && std::fwrite(out.data(), 4, out.size(), o) == out.size(), "write");
    std::fclose(o);
    return 0;
}
"""

CONTRACT_MAIN = r"""// museport stage 1 (forward): the contract main. Runs the model on llm.c's
// debug batch and writes raw float32: the B*T*V logits of a target-free forward
// pass, then the mean loss.
// usage: <prog> <debug_state.bin> <output> <gpt2_124M.bin>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    MUSEPORT_SETUP
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[3]);
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int sh[256];
    FILE* s = fopenCheck(argv[1], "rb");
    freadCheck(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B * T, s);
    freadCheck(y, sizeof(int), B * T, s);
    fcloseCheck(s);
    float* logits = (float*)mallocCheck((size_t)B * T * Vp * sizeof(float));
    gpt2_forward(&model, x, NULL, B, T);
    MUSEPORT_COPY_LOGITS(logits, (size_t)B * T * Vp);
    gpt2_forward(&model, x, y, B, T);
    FILE* o = fopenCheck(argv[2], "wb");
    for (int bt = 0; bt < B * T; bt++) fwrite(logits + (size_t)bt * Vp, sizeof(float), V, o);
    float loss = model.mean_loss;
    fwrite(&loss, sizeof(float), 1, o);
    fcloseCheck(o);
    free(x);
    free(y);
    free(logits);
    gpt2_free(&model);
    return 0;
}
"""

CUDA_HARNESS = r"""// museport GPT-2 stage 1 harness: llm.c's train_gpt2_fp32.cu (fp32 GPT-2
// training, unchanged) with a contract main in place of its own: a target-free
// forward pass on llm.c's debug batch.
#define TESTING
#include "train_gpt2_fp32.cu"
// set up the device and cuBLAS as llm.c's test_gpt2_fp32.cu does (TF32 off)
#define MUSEPORT_SETUP \
    cudaCheck(cudaSetDevice(0)); \
    cublasCheck(cublasCreate(&cublas_handle)); \
    cublas_compute_type = CUBLAS_COMPUTE_32F; \
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
#define MUSEPORT_COPY_LOGITS(dst, n) \
    cudaCheck(cudaMemcpy(dst, model.acts.output, (n) * sizeof(float), cudaMemcpyDeviceToHost))
""" + CONTRACT_MAIN

# Proof of the check before any model runs: llm.c's CPU implementation, same contract main.
CPU_PROOF = r"""// Proof of the GPT-2 stage-1 check: llm.c's CPU train_gpt2.c with the same
// contract main.
#define TESTING
#include "train_gpt2.c"
#define MUSEPORT_SETUP
#define MUSEPORT_COPY_LOGITS(dst, n) memcpy(dst, model.acts.logits, (n) * sizeof(float))
""" + CONTRACT_MAIN

HARNESS = "gpt2_forward_harness.cu"

REFERENCE_BACKWARD = r"""// Hidden reference for GPT-2 stage 2 (backward): PyTorch's gradients of
// every parameter and the loss for llm.c's debug batch, read from
// gpt2_124M_debug_state.bin, written as raw float32: the gradients, then the loss.
// usage: reference <debug_state.bin> <output> <gpt2_124M.bin>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void must(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
int main(int argc, char** argv) {
    must(argc == 4, "usage: reference <debug_state.bin> <output> <gpt2_124M.bin>");
    int mh[256], sh[256];
    FILE* m = std::fopen(argv[3], "rb");
    must(m && std::fread(mh, 4, 256, m) == 256, "weights header");
    std::fclose(m);
    must(mh[0] == 20240326, "bad weights magic");
    FILE* s = std::fopen(argv[1], "rb");
    must(s && std::fread(sh, 4, 256, s) == 256, "state header");
    must(sh[0] == 20240327 && sh[1] == 2, "bad state magic or version");
    const long B = sh[2], T = sh[3], V = mh[3];
    // header, x, y, logits, then the loss; the gradients fill the rest of the file
    const long loss_at = 4L * (256 + 2 * B * T + B * T * V);
    must(std::fseek(s, 0, SEEK_END) == 0, "seek");
    const long end = std::ftell(s);
    const long P = (end - loss_at - 4) / 4;
    must(P > 0 && loss_at + 4 + 4 * P == end, "state file size");
    float loss = 0.0f;
    must(std::fseek(s, loss_at, SEEK_SET) == 0 && std::fread(&loss, 4, 1, s) == 1, "loss");
    std::vector<float> grads(P);
    must(std::fread(grads.data(), 4, P, s) == (size_t)P, "gradients");
    std::fclose(s);
    FILE* o = std::fopen(argv[2], "wb");
    must(o && std::fwrite(grads.data(), 4, P, o) == (size_t)P, "write");
    must(std::fwrite(&loss, 4, 1, o) == 1, "write loss");
    std::fclose(o);
    return 0;
}
"""

CONTRACT_MAIN_BACKWARD = r"""// museport stage 2 (backward): the contract main. Runs a forward pass
// with targets on llm.c's debug batch, then the backward pass, and writes raw
// float32: the gradient of every parameter, then the mean loss.
// usage: <prog> <debug_state.bin> <output> <gpt2_124M.bin>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    MUSEPORT_SETUP
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[3]);
    int sh[256];
    FILE* s = fopenCheck(argv[1], "rb");
    freadCheck(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B * T, s);
    freadCheck(y, sizeof(int), B * T, s);
    fcloseCheck(s);
    gpt2_forward(&model, x, y, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    size_t P = model.num_parameters;
    float* grads = (float*)mallocCheck(P * sizeof(float));
    MUSEPORT_COPY_GRADS(grads, P);
    FILE* o = fopenCheck(argv[2], "wb");
    fwrite(grads, sizeof(float), P, o);
    float loss = model.mean_loss;
    fwrite(&loss, sizeof(float), 1, o);
    fcloseCheck(o);
    free(x);
    free(y);
    free(grads);
    gpt2_free(&model);
    return 0;
}
"""

CUDA_SETUP = r"""// set up the device and cuBLAS as llm.c's test_gpt2_fp32.cu does (TF32 off)
#define MUSEPORT_SETUP \
    cudaCheck(cudaSetDevice(0)); \
    cublasCheck(cublasCreate(&cublas_handle)); \
    cublas_compute_type = CUBLAS_COMPUTE_32F; \
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
"""

# Stage 3 (training): llm.c's test_gpt2_fp32.cu and test_gpt2.c run 10 AdamW steps on the
# debug batch and compare each step's loss with PyTorch's within 1e-2.
STEPS = 10
EXPECTED_LOSSES = (  # test_gpt2_fp32.cu, "expected losses are as follows, from Python"
    5.270007133483887, 4.059706687927246, 3.3751230239868164, 2.8007826805114746,
    2.315382242202759, 1.8490285873413086, 1.3946564197540283, 0.9991465210914612,
    0.6240804195404053, 0.37651097774505615,
)

REFERENCE_TRAIN = r"""// Hidden reference for GPT-2 stage 3 (training): PyTorch's loss at each of
// 10 AdamW steps on llm.c's debug batch, as llm.c's test_gpt2_fp32.cu lists
// them, written as raw float32.
// usage: reference <debug_state.bin> <output> <gpt2_124M.bin>
#include <cstdio>
#include <cstdlib>
static void must(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
int main(int argc, char** argv) {
    must(argc == 4, "usage: reference <debug_state.bin> <output> <gpt2_124M.bin>");
    int sh[256];
    FILE* s = std::fopen(argv[1], "rb");
    must(s && std::fread(sh, 4, 256, s) == 256, "state header");
    must(sh[0] == 20240327 && sh[1] == 2, "bad state magic or version");
    std::fclose(s);
    const float losses[] = {""" + ", ".join(f"{v!r}f" for v in EXPECTED_LOSSES) + r"""};
    FILE* o = std::fopen(argv[2], "wb");
    const size_t n = sizeof(losses) / sizeof(losses[0]);
    must(o && std::fwrite(losses, 4, n, o) == n, "write");
    std::fclose(o);
    return 0;
}
"""

CONTRACT_MAIN_TRAIN = r"""// museport stage 3 (training): the contract main. Runs 10 training
// steps on llm.c's debug batch as test_gpt2_fp32.cu does (forward with
// targets, zero the gradients, backward, AdamW update) and writes each step's
// mean loss as raw float32.
// usage: <prog> <debug_state.bin> <output> <gpt2_124M.bin>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
        return 1;
    }
    MUSEPORT_SETUP
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[3]);
    int sh[256];
    FILE* s = fopenCheck(argv[1], "rb");
    freadCheck(sh, sizeof(int), 256, s);
    int B = sh[2], T = sh[3];
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(x, sizeof(int), B * T, s);
    freadCheck(y, sizeof(int), B * T, s);
    fcloseCheck(s);
    float losses[10];
    for (int step = 0; step < 10; step++) {
        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);
        losses[step] = model.mean_loss;
    }
    FILE* o = fopenCheck(argv[2], "wb");
    fwrite(losses, sizeof(float), 10, o);
    fcloseCheck(o);
    free(x);
    free(y);
    gpt2_free(&model);
    return 0;
}
"""

STAGES: dict[str, dict[str, Any]] = {
    "forward": {
        "harness": HARNESS, "name": "llmc_gpt2_forward",
        "cuda": CUDA_HARNESS, "cpu": CPU_PROOF, "reference": REFERENCE,
        "does": "runs a target-free forward pass on the debug batch and then one with its "
                "targets, and writes raw float32: the B*T*V logits (V per row, not the padded "
                "Vp), then the mean loss",
        "tolerance": TOLERANCE,
        "note": "logits and loss checked against PyTorch within 1e-2"},
    "backward": {
        "harness": "gpt2_backward_harness.cu", "name": "llmc_gpt2_backward",
        "cuda": "// museport GPT-2 stage 2 harness: llm.c's train_gpt2_fp32.cu (fp32 GPT-2\n"
                "// training, unchanged) with a contract main in place of its own: a forward\n"
                "// and a backward pass on llm.c's debug batch.\n"
                "#define TESTING\n#include \"train_gpt2_fp32.cu\"\n" + CUDA_SETUP
                + "#define MUSEPORT_COPY_GRADS(dst, n) \\\n"
                  "    cudaCheck(cudaMemcpy(dst, model.grads_memory, (n) * sizeof(float), "
                  "cudaMemcpyDeviceToHost))\n" + CONTRACT_MAIN_BACKWARD,
        "cpu": "// Proof of the GPT-2 stage-2 check: llm.c's CPU train_gpt2.c with the same\n"
               "// contract main.\n#define TESTING\n#include \"train_gpt2.c\"\n"
               "#define MUSEPORT_SETUP\n#define MUSEPORT_COPY_GRADS(dst, n) "
               "memcpy(dst, model.grads_memory, (n) * sizeof(float))\n"
               + CONTRACT_MAIN_BACKWARD,
        "reference": REFERENCE_BACKWARD,
        "does": "runs a forward pass with targets on the debug batch, zeroes the gradients, "
                "runs the backward pass, and writes raw float32: the gradient of every "
                "parameter (model.num_parameters values, in llm.c's parameter order), then "
                "the mean loss",
        "tolerance": GRAD_TOLERANCE,
        "note": "gradients of all parameters and the loss checked against PyTorch within "
                "2e-2 (llm.c's CPU test tolerance for gradients)"},
    "train": {
        "harness": "gpt2_train_harness.cu", "name": "llmc_gpt2_train",
        "cuda": "// museport GPT-2 stage 3 harness: llm.c's train_gpt2_fp32.cu (fp32 GPT-2\n"
                "// training, unchanged) with a contract main in place of its own: 10 AdamW\n"
                "// training steps on llm.c's debug batch.\n"
                "#define TESTING\n#include \"train_gpt2_fp32.cu\"\n" + CUDA_SETUP
                + CONTRACT_MAIN_TRAIN,
        "cpu": "// Proof of the GPT-2 stage-3 check: llm.c's CPU train_gpt2.c with the same\n"
               "// contract main.\n#define TESTING\n#include \"train_gpt2.c\"\n"
               "#define MUSEPORT_SETUP\n" + CONTRACT_MAIN_TRAIN,
        "reference": REFERENCE_TRAIN,
        "does": "runs 10 training steps on the debug batch (each: a forward pass with "
                "targets, zero the gradients, the backward pass, then the AdamW update with "
                "lr 1e-4, beta1 0.9, beta2 0.999, eps 1e-8, weight decay 0.01, t = step + 1), "
                "and writes raw float32: the mean loss of each step's forward pass, 10 values",
        "tolerance": TOLERANCE,
        "note": "the loss of each of 10 AdamW steps checked against PyTorch's within 1e-2"},
}


def goal(stage: str = "forward") -> str:
    st = STAGES[stage]
    return (
        f"Port {st['harness']} (llm.c's train_gpt2_fp32.cu, the fp32 GPT-2 training program, "
        "which it includes) to idiomatic SYCL 2020 with zero dpct helpers, computed on the "
        "SYCL device. Keep the program's contract: `build/candidate <debug_state.bin> <output> "
        f"build/gpt2_124M.bin` loads the GPT-2 124M weights, {st['does']}, exactly as the "
        "CUDA main does. A hidden reference checks the output against PyTorch's within "
        f"{st['tolerance']:g} absolute (llm.c's own test tolerance), so use precise math "
        "(sycl::exp, not native::exp). "
        "cuBLAS calls go to oneMath (oneapi::math::blas::column_major gemm and gemm_batch, "
        "#include <oneapi/math.hpp>; the compile flags already link -lonemath). Success "
        "requires compile, verify and a zero-helper scan_helpers of the whole work directory "
        "(path '.') on the same sources: delete migrate's output under migrated/ "
        "(delete_file) once your own port is done. The files are long: read them in parts "
        "(read_file offset/limit)."
    )


def make_task(llmc_dir: Path, data_dir: Path, out: Path, stage: str = "forward") -> Path:
    """Write the task directory of a stage ("forward", "backward" or "train"); returns its
    task.json."""
    st = STAGES[stage]
    for name in (WEIGHTS, STATE):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"{data_dir / name} missing: run llm.c's "
                                    "dev/download_starter_pack.sh")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(llmc_dir / "train_gpt2_fp32.cu", out / "train_gpt2_fp32.cu")
    (out / "llmc").mkdir(exist_ok=True)
    for h in LLMC_HEADERS:
        shutil.copy(llmc_dir / "llmc" / h, out / "llmc" / h)
    (out / st["harness"]).write_text(st["cuda"])
    (out / "gpt2_ref.cpp").write_text(st["reference"])
    (out / "cpu_proof.c").write_text(st["cpu"])
    for name in (WEIGHTS, STATE):  # linked, never copied or committed
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((data_dir / name).resolve())
    task = {
        "name": st["name"],
        "goal": goal(stage),
        "cuda_path": st["harness"],
        "include_dirs": ["."],
        "reference_build": [
            "bash", "-c",
            f"mkdir -p build && ln -sf {{task_dir}}/{WEIGHTS} build/{WEIGHTS} && "
            "g++ -O2 -std=c++17 {task_dir}/gpt2_ref.cpp -o build/reference"],
        "reference_cmd": f"build/reference {{input}} {{output}} build/{WEIGHTS}",
        "candidate_cmd": f"build/candidate {{input}} {{output}} build/{WEIGHTS}",
        "inputs": [STATE],
        "mode": "array",
        "output_format": "f32",
        "base_flags": list(ONEMATH_FLAGS),
        "tolerance": st["tolerance"],
        "rtol": 0.0,
        "museport": {"source": "karpathy/llm.c train_gpt2_fp32.cu", "stage": stage,
                     "note": "B=4, T=64 debug batch from gpt2_124M_debug_state.bin; "
                             f"{st['note']}, as test_gpt2_fp32.cu does"},
    }
    path = out / "task.json"
    path.write_text(json.dumps(task, indent=2) + "\n")
    return path


def check_task(task_file: Path, llmc_dir: Path, work: Path) -> dict[str, object]:
    """Prove a stage's check before any model runs: llm.c's CPU implementation, built with the
    same contract main, must pass the hidden reference (CPU only, no model, no cost)."""
    from oneapi_agent.toolchain import run_command
    from oneapi_agent.tools.verify import verify

    task_dir = task_file.parent.resolve()
    if work.exists():
        shutil.rmtree(work)
    (work / "build").mkdir(parents=True)
    steps: dict[str, object] = {}
    task = json.loads(task_file.read_text())
    ref = run_command([a.replace("{task_dir}", str(task_dir)) for a in task["reference_build"]],
                      cwd=str(work), timeout=600)
    steps["reference_build"] = {"returncode": ref.returncode, "stderr": ref.stderr[-2000:]}
    cpu = run_command(["gcc", "-O3", "-fopenmp", f"-I{llmc_dir.resolve()}",
                       str(task_dir / "cpu_proof.c"), "-lm", "-o", "build/cpu_proof"],
                      cwd=str(work), timeout=600)
    steps["cpu_proof_build"] = {"returncode": cpu.returncode, "stderr": cpu.stderr[-2000:]}
    if ref.returncode or cpu.returncode:
        return {"ok": False, "steps": steps}
    res = verify(task["reference_cmd"], f"build/cpu_proof {{input}} {{output}} build/{WEIGHTS}",
                 [str(task_dir / i) for i in task["inputs"]], task["tolerance"],
                 rtol=task["rtol"], mode="array", cwd=str(work), work_dir=str(work / "verify"))
    steps["verify_cpu_proof"] = {"status": res.status, "summary": res.summary}
    return {"ok": res.status == "ok", "steps": steps}


# ----------------------------------------------------------------------------- GPU runs

# llm.c's dev/download_starter_pack.sh fetches the data from here; the GPU machine does the
# same (the files are 1 GB, far over a remote-test bundle's 50 MB), checked against these.
STARTER_PACK = "https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/"
DATA_SHA256 = {
    WEIGHTS: "3da8b207584030bcdcd207cf7a99952e3421dce92da218b351071857511bf162",
    STATE: "a80c4448eed47c561314a2189e62c49b2c7477c265fc2f15c8470d18401693d9",
}
# The same oneMath commit as the toolkit's CPU build (scripts/install-toolchain.sh).
ONEMATH_REF = "1308c5e979daf0c80e3d6ae61c1e2de5ad0f2e4a"

# Sourced on the GPU machine before gpu-test (the task's gpu_setup): builds oneMath's BLAS
# domain with the vendor's backend (cuBLAS on NVIDIA, rocBLAS on AMD), with the machine's SYCL
# compiler, and exports the ONEMATH_PREFIX that the task's base_flags use.
ONEMATH_GPU_SETUP_TEMPLATE = r"""# museport: oneMath (BLAS, __BACKEND__ backend) for __VENDOR__.
ONEMATH_PREFIX="$HOME/onemath-__BACKEND__"
if [ ! -f "$ONEMATH_PREFIX/lib/libonemath.so" ]; then
  CXX_BIN="${SYCL_CXX:-$(command -v icpx || command -v clang++)}"
  C_BIN="$(dirname "$CXX_BIN")/clang"
  [ -x "$C_BIN" ] || C_BIN="$(command -v icx || command -v clang || command -v gcc)"
  echo "oneMath __REF__ with $CXX_BIN"
  python3 -m pip install -q cmake ninja 2>/dev/null \
    || python3 -m pip install -q --break-system-packages cmake ninja
  export PATH="$HOME/.local/bin:$PATH"
__PRE__
  SRC="$(mktemp -d)/oneMath"
  git init -q "$SRC" \
    && git -C "$SRC" fetch -q --depth 1 https://github.com/uxlfoundation/oneMath.git __REF__ \
    && git -C "$SRC" checkout -q FETCH_HEAD
  cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" -DCMAKE_C_COMPILER="$C_BIN" \
    -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_MKLGPU_BACKEND=OFF __FLAGS__ \
    -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX="$ONEMATH_PREFIX" -DCMAKE_INSTALL_RPATH="$ONEMATH_PREFIX/lib" \
    && cmake --build "$SRC/build" -j"$(nproc)" && cmake --install "$SRC/build"
fi
export ONEMATH_PREFIX
export LD_LIBRARY_PATH="$ONEMATH_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ls -l "$ONEMATH_PREFIX/lib"
"""

# What differs per vendor: the backend, its CMake flags, a pre-step, and llm.c's native build.
ONEMATH_BACKENDS: dict[str, dict[str, Any]] = {
    "nvidia": {"backend": "cublas", "flags": "-DENABLE_CUBLAS_BACKEND=ON", "pre": "",
               "native_flags": ["-lcublas"]},
    # rocBLAS needs the GPU's gfx target; the machine reports it.
    "amd": {"backend": "rocblas",
            "flags": '-DENABLE_ROCBLAS_BACKEND=ON -DHIP_TARGETS="$HIP_ARCH" '
                     "-DCMAKE_PREFIX_PATH=/opt/rocm",
            "pre": '  HIP_ARCH="$( (rocm_agent_enumerator 2>/dev/null; rocminfo 2>/dev/null '
                   "| awk '/Name: *gfx/{print $2}') | grep -v gfx000 | head -1)\"\n"
                   '  echo "AMD target: $HIP_ARCH"',
            # llm.c's CUDA does not hipify into something that builds (evidence: Tier 1 AMD)
            "native_flags": []},
}


def onemath_gpu_setup(vendor: str) -> str:
    b = ONEMATH_BACKENDS[vendor]
    return (ONEMATH_GPU_SETUP_TEMPLATE.replace("__REF__", ONEMATH_REF)
            .replace("__BACKEND__", b["backend"]).replace("__VENDOR__", vendor.upper())
            .replace("__FLAGS__", b["flags"]).replace("__PRE__", b["pre"]))


ONEMATH_GPU_SETUP = onemath_gpu_setup("nvidia")


def make_gpu_task(stage_dir: Path, dest: Path, model: str, vendor: str = "nvidia") -> Path:
    """A ``remote-test`` task for a passing GPT-2 stage: its task (harness, hidden reference),
    Muse's port as last compiled, the data fetched on the machine, and oneMath built there with
    the vendor's backend (cuBLAS or rocBLAS). On NVIDIA the native build is llm.c's CUDA with
    cuBLAS."""
    from .port import last_compiled_sources

    srcs = last_compiled_sources(stage_dir / "run1" / "trace.jsonl")
    if not srcs:
        raise ValueError(f"{stage_dir}: the model never compiled")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(stage_dir / "task", dest, ignore=shutil.ignore_patterns(WEIGHTS, STATE))
    shutil.copytree(stage_dir / "run1" / "port", dest / "agent_port")
    backend = ONEMATH_BACKENDS[vendor]
    setup = f"onemath_{backend['backend']}_setup.sh"
    (dest / setup).write_text(onemath_gpu_setup(vendor))
    task = json.loads((dest / "task.json").read_text())
    task["sycl_sources"] = [f"agent_port/{s}" for s in srcs]
    task["native_flags"] = backend["native_flags"]
    task["gpu_setup"] = setup
    task["fetch"] = [{"path": f, "url": f"{STARTER_PACK}{f}?download=true",
                      "sha256": DATA_SHA256[f]} for f in (WEIGHTS, STATE)]
    task["port_origin"] = (f"agent-made: {model} via museport loop, {stage_dir.name}; "
                           "sources as last compiled by the agent, unchanged")
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"
