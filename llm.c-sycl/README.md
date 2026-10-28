# llm.c-sycl

[karpathy/llm.c](https://github.com/karpathy/llm.c), ported from CUDA to SYCL 2020 by Meta's
`muse-spark-1.3` with [Muse Port](../README.md). One source runs on a CPU, an NVIDIA GPU and an
AMD GPU, and `make` builds it with `icpx -fsycl`.

Every piece here passed a hidden check the model never saw before it was kept. The checks are
built from llm.c's own CPU code, PyTorch's numbers and llm.c's tolerances.

| Program | CPU | NVIDIA RTX 2000 Ada | AMD Instinct MI350X |
|---|---|---|---|
| `test_gpt2fp32sycl`: llm.c's own test, `overall okay: 1`; worst loss error vs PyTorch | 0.000154 | 0.000127 | 0.0000236 |
| `train_gpt2fp32sycl`: worst loss error vs llm.c's CPU trainer (10 steps) | 0.00227 | 0.0023 | 0.00242 |
| `dev/sycl`: 19 kernel files, each against its hidden check | 19 of 19 | | |

On NVIDIA, both programs also match llm.c's own CUDA programs, built with `nvcc` on the same
GPU: 7.03e-06 for the test and 0.00147 for the trainer. On AMD, llm.c's CUDA does not convert
to HIP automatically, so there is no such comparison. Evidence:
[test](../evidence/j3-test/README.md), [trainer](../evidence/j3-trainer/README.md),
[kernels](../evidence/j3-assemble/README.md), [fresh clone](../evidence/j3-fresh-clone/README.md).

## Quick start

You need a SYCL compiler (`icpx` from Intel oneAPI DPC++ for the CPU, or intel/llvm built with
the CUDA or HIP backend for a GPU). You also need [oneMath](https://github.com/uxlfoundation/oneMath)
built with its BLAS domain for your device: MKL-CPU, cuBLAS or rocBLAS. Then set
`ONEMATH_PREFIX` to the oneMath install. The
[oneapi-agent-toolkit](https://github.com/checocadena/oneapi-agent-toolkit) script
`scripts/install-toolchain.sh` (with `WITH_ONEMATH=1`) installs both for the CPU.

```bash
cd llm.c-sycl
./dev/download_starter_pack.sh      # llm.c's weights, debug state, tokenizer, tiny-shakespeare
make                                 # the CPU device; or: make TARGET=nvidia / TARGET=amd AMD_ARCH=gfx950
./test_gpt2fp32sycl                  # llm.c's test: ends with "overall okay: 1"
./train_gpt2fp32sycl                 # llm.c's trainer, with llm.c's options (-b -t -l -v -m -s -g ...)
```

From a fresh clone on the CPU device, `make` took 26 s and the test 34 s. The trainer with
llm.c's defaults (B=4, T=1024, one epoch of 74 steps) took about 38 s per step on that CPU.
On a GPU it is meant to run as llm.c's trainer does.

## What is here

- **`train_gpt2_fp32.cpp`:** llm.c's fp32 GPT-2 trainer: forward, backward, AdamW, with the
  matmuls through oneMath, and llm.c's real `main` (options, dataloaders, validation, sampling).
  - Muse ported the model in three stages, each checked against PyTorch
    ([GPT-2 stages](../README.md#gpt-2-llmcs-real-training-program)).
  - Muse then wrapped the model in llm.c's `main`, keeping 94% of the proven lines verbatim, in
    19 turns for $0.71.
- **`test_gpt2_fp32.cpp`:** llm.c's own test, `test_gpt2_fp32.cu`, ported.
  - **What it checks:** the logits, the loss and every gradient against PyTorch's debug state,
    then the losses of 10 training steps, with llm.c's tolerances (1e-2).
  - **How it was made:** Muse wrote it in 24 turns for $0.43, on its third run. It also guarded
    the trainer's `main` with `#ifndef TESTING`, as llm.c does.
- **`dev/sycl/`:** the 19 llm.c `dev/cuda` kernel files Muse ported, unchanged from the graded
  runs, with their provenance ([PROVENANCE.json](dev/sycl/PROVENANCE.json)).
  - `make -C dev/sycl` builds them.
  - `museport llmc-sycl check` grades each against its hidden check.
- **`llmc/`:** llm.c's plain-C host headers, unchanged.
- **`dev/download_starter_pack.sh`:** llm.c's data script, unchanged.
- **`LICENSE`:** llm.c's (MIT).

## How this folder is put together

`museport llmc-sycl assemble --llmc <an llm.c checkout>` writes every file here except this
README. A test checks that its output matches the committed folder, byte for byte.

- **Kernels:** it copies each kernel port from its graded run.
- **Programs:** it copies the trainer and the test from the run that wrote the test. It first
  checks that this trainer is the graded J3 trainer plus only the two `#ifndef TESTING` guard
  lines.
- **The rest:** it copies llm.c's own files, and writes both Makefiles and the `.gitignore`
  files.

The Makefiles are templates written by hand for llm.c, not converted by Muse. The command works
for llm.c only: the list of kernels and runs is part of the code.

## Notes and limits

- **fp32 only.** llm.c's bf16 production trainer (`train_gpt2.cu`: cuBLASLt, cuDNN,
  multi-GPU) is not ported.
- **Lean rewrites:** the kernels are simple SYCL rewrites, not llm.c's tuned kernels.
  Performance was not compared.
- **Short runs:** the trainer was checked on 10 training steps, not a full run.
- **Sources on the GPUs:** the GPU runs built the sources the graded runs compiled. The trainer
  here differs from them only by the `#ifndef TESTING` guard.
- **GPU builds:** the table's GPU results came from the toolkit's `gpu-test` build.
  - **This Makefile on GPUs:** `make TARGET=nvidia` and `make TARGET=amd AMD_ARCH=gfx950` built
    both programs on an RTX 2000 Ada and an MI350X.
  - **Its test there:** the Makefile-built test passed vs PyTorch, 0.000126 on NVIDIA and
    0.0000236 on AMD. On NVIDIA it also matched llm.c's CUDA (5.01e-06)
    ([evidence](../evidence/j3-makefile-gpu/README.md)).
- **Cost:** making the two programs cost $2.33 of Muse (four loop runs), and the GPU runs $0.63
  (`docs/J3-PLAN.md`).
