<!-- Draft PR for uxlfoundation/oneMath, base: develop.
     Title: [BLAS][cuBLAS] Implement gemm for bfloat16 inputs with float output -->

# Description

**The bug:** the cuBLAS backend throws `oneapi::math::unimplemented` for column-major `gemm` with
`bfloat16` A and B and `float` C. It does so in both the buffer and USM APIs. The interface is
declared in oneMath's BLAS API and is implemented by the rocBLAS backend
(`rocblas_level3.cpp`, `GEMM_EX_LAUNCHER(bfloat16, bfloat16, float, float, ...)`). The fp16
version of the same call works on cuBLAS: it goes through `cublasGemmEx` with `CUDA_R_16F`.

**The fix:** this PR adds the bf16 case beside the fp16 ones, through the existing launcher
macros, and removes the two functions that threw:

```cpp
GEMM_EX_LAUNCHER(bfloat16, bfloat16, float, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F)
GEMM_EX_LAUNCHER_USM(bfloat16, bfloat16, float, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F)
```

`cublasGemmEx` supports `CUDA_R_16BF` inputs with `CUDA_R_32F` output and fp32 compute. The
existing `gemm_ex` helper passes the output type, `CUDA_R_32F`, as the compute type, as it does
for fp16 inputs with fp32 output. Row-major is unchanged: it still throws, like every row-major
level-3 call in this backend.

**How we found it:** a SYCL port of [llm.c](https://github.com/karpathy/llm.c)'s bf16 GPT-2
trainer, which does its matmuls through oneMath, aborted on NVIDIA GPUs with this exception. On
the same GPU, llm.c's CUDA build runs. With this change the port passes llm.c's own test on
NVIDIA: logits, all 16 gradient tensors, and 10 training losses against PyTorch. It already
passed on an AMD MI350X through the rocBLAS backend. The port and its evidence are at
https://github.com/checocadena/museport/tree/main/evidence/l1-llmc-bf16.

**How to reproduce, without the change:**

1. Build oneMath with `-DENABLE_CUBLAS_BACKEND=ON -DTARGET_DOMAINS=blas` and a reference BLAS.
2. Run `test_main_blas_ct --gtest_filter='*Gemm*Bfloat16*'` on an NVIDIA GPU.
3. Every column-major `Bfloat16Bfloat16FloatPrecision` case is reported as SKIPPED, because the
   test harness maps `unimplemented` to a skip.

**AI assistance:** this change was written by an AI assistant (Claude) working with me. It was
tested as described below.

Fixes # (no issue filed; the reproduction is above)

# Checklist

## All Submissions

- [ ] Do all unit tests pass locally? Attach a log.
  - Not all pass, but the failures are the same with and without this change.
  - **Setup:** built at `develop` `585e610` with only the cuBLAS backend and the BLAS domain,
    and the whole BLAS suite run before and after this change. The run used an NVIDIA L4
    (driver 595.91.07, CUDA 13.2), the intel/llvm nightly of 2026-10-30, and
    `ONEAPI_DEVICE_SELECTOR=cuda:*`.
  - **Compile-time and run-time APIs alike:** 323 passed, 88 failed and 575 skipped before;
    **325 passed**, 88 failed and 573 skipped after.
  - **The change:** in both APIs, `GemmTests.Bfloat16Bfloat16FloatPrecision/Column_Major` and
    `GemmUsmTests.Bfloat16Bfloat16FloatPrecision/Column_Major` go from SKIPPED to OK. Nothing
    else changes.
  - **The 88 failures** are the same set before and after this change: level-1 tests (scal,
    copy, swap, nrm2, ...), each failing within about 1 ms. Whatever runs first in a test
    process fails this way on this setup; a run filtered to `*Gemm*` showed the same for gemm.
    This looks like a start-up issue of this environment, unrelated to the change. I can rerun
    on another setup if you want a clean baseline.
  - Log attached: `onemath-cublas-blas-tests.log`.

## Bug fixes

- [x] Have you added relevant regression tests?
  - The existing `GemmTests.Bfloat16Bfloat16FloatPrecision` and
    `GemmUsmTests.Bfloat16Bfloat16FloatPrecision` already cover this; on cuBLAS they now run
    instead of being skipped.
- [x] Have you included information on how to reproduce the issue (either in a GitHub issue or
  in this PR)? Yes, above.
