# GPT-2 stage 3 (training) on an NVIDIA GPU

Muse's SYCL port of llm.c's GPT-2 training ([`../run1/port/gpt2_train.cpp`](../run1/port/gpt2_train.cpp),
unchanged, as Muse last compiled it) ran on a real NVIDIA GPU. It was checked against PyTorch's
losses and against llm.c's own CUDA training program, built with `nvcc` on the same machine.

- **Date:** 2026-10-27, about 16:32-16:35 UTC.
- **Machine:** Runpod, NVIDIA RTX 4000 Ada Generation ($0.28/h), created and deleted by
  `oneapi-agent remote-test --provider runpod --vendor nvidia --max-usd 1 --max-minutes 60`. The
  log ends with `deleted runpod machine <id>: confirmed`.
- **Task:** written by `museport gpt2-gpu-task evidence/gpt2-stage3`. It uses two task fields
  added to the toolkit for this run (see [`docs/CHANGES-TO-TOOLKIT.md`](../../../docs/CHANGES-TO-TOOLKIT.md)):
  - `fetch`: the machine downloads `gpt2_124M.bin` and `gpt2_124M_debug_state.bin` from llm.c's
    Hugging Face starter pack and checks their sha256. The files are 1 GB, too big for the
    upload bundle.
  - `gpu_setup`: the machine builds oneMath (the same commit as the CPU build) with its
    **cuBLAS backend**, using the SYCL compiler it runs. [`setup.log`](nvidia/nvidia-rtx-4000-ada-generation/llmc_gpt2_train/setup.log)
    shows `libonemath_blas_cublas.so` installed.

## Result: PASS against PyTorch and against llm.c's CUDA

| Check | Result (10 losses, atol 1e-2) |
|---|---|
| Muse's SYCL port vs PyTorch (hidden reference) | **PASS**, worst max abs err 0.000135 |
| Muse's SYCL port vs llm.c's CUDA (`train_gpt2_fp32.cu`, cuBLAS) | **PASS**, worst max abs err 0.0000059 |
| llm.c's CUDA vs PyTorch | PASS, worst max abs err 0.000128 |

The SYCL program ran on the CUDA backend (`cuda:0`, "NVIDIA CUDA BACKEND, NVIDIA RTX 4000 Ada
Generation 8.9"). Its matmuls go through oneMath to cuBLAS.

## Time

Whole-program wall time for the 10 training steps, 10 runs each: SYCL median 1,310 ms, llm.c's
CUDA median 1,300 ms. Both include loading the 498 MB weights from disk, so this is not a kernel
speed comparison. Muse's port uses simple kernels, not llm.c's tuned ones.

## Cost

$0.011 (the tool's estimate: 2.3 minutes at $0.28/h). Four cheaper GPU types were out of stock
first; nothing was created for them. Log: [`logs/`](logs/).
