<!-- Draft for https://github.com/uxlfoundation/oneMath/issues/new (template: "Request a feature").
     Title: BLAS: bf16 gemm_batch and bf16-output gemm (parity with fp16) -->

# Summary

oneMath's BLAS API offers two fp16 matrix-multiply forms that have no bf16 equivalent:

- `gemm_batch` takes `sycl::half` but not `bfloat16`;
- `gemm` with fp16 output (`half, half -> half`) exists, but bf16 has only `bfloat16, bfloat16 ->
  float`.

bf16 training code needs both. We hit the gap while porting a bf16 GPT-2 trainer (llm.c) to SYCL
and oneMath, and measured what it costs on NVIDIA GPUs. We ask whether these forms can be added,
and whether that needs an RFC first.

*Disclosure: this issue was drafted by an AI assistant (Claude) working with the author, from a
port written by another AI model (Meta's Muse). The numbers below come from that project's
committed evidence.*

# Problem statement

[llm.c](https://github.com/karpathy/llm.c)'s main trainer runs GPT-2 in bf16 mixed precision.
Its attention computes many small per-head matmuls with one strided-batched cuBLAS call, and its
dense layers write bf16 outputs directly.

A SYCL port of it that keeps to oneMath cannot do either:

- **Attention:** with no bf16 `gemm_batch`, the port loops over oneMath `gemm` calls, one per
  (batch, head). It also needs fp32 scratch, because the only bf16 `gemm` writes fp32.
- **Dense layers:** with no bf16-output `gemm`, each matmul writes fp32, and a separate kernel
  converts it to bf16. That is an extra pass over the output.

**Measured:** llm.c's trainer, B=4, T=1024, the median of llm.c's own step timer over steps 6-30.
On NVIDIA it runs beside llm.c's CUDA build on the same GPU:

| GPU | oneMath only: port ms / CUDA ms | Port / CUDA | Same code, plus `cublasGemmStridedBatchedEx` and `cublasGemmEx` (bf16) through SYCL backend interop | Port / CUDA |
|---|---|---|---|---|
| RTX 2000 Ada | 250.6 / 184.1 | **1.36x** | 205.5 / 181.9 | **1.13x** |
| RTX A4500 | 139.8 / 91.8 | **1.52x** | 113.3 / 93.6 | **1.21x** |

Each cell is a median of 2 or 3 runs. Other runs of the interop variant agree: 1.14x on an
RTX 4090 and on an L4.

Everything else in the two variants is identical. The port passes llm.c's own test (logits, all
gradients, losses against PyTorch) on NVIDIA and on an AMD MI350X. On AMD the interop path
compiles out, and the port runs through oneMath's rocBLAS backend.

Code and evidence:

- the port's matmul code: [`batched_gemm.hpp`](https://github.com/checocadena/museport/blob/main/evidence/l1-llmc-bf16/gtune/port/llmc/batched_gemm.hpp)
  and [`matmul.hpp`](https://github.com/checocadena/museport/blob/main/evidence/l1-llmc-bf16/gtune/port/llmc/matmul.hpp);
- the measurements: [`evidence/l1-llmc-bf16/gtune`](https://github.com/checocadena/museport/tree/main/evidence/l1-llmc-bf16/gtune).

# Preferred solution

Add the bf16 forms beside the existing fp16 ones, in both the buffer and USM APIs:

1. `gemm` with `Ta = Tb = bfloat16`, `Tc = bfloat16` and `Ts = float`.
2. `gemm_batch`, strided and group forms, with `Ta = Tb = bfloat16` and `Tc = float` and/or
   `Tc = bfloat16`, matching whatever the fp16 batch forms cover.

**Backends:**

- **cuBLAS:** `cublasGemmEx` and `cublasGemmStridedBatchedEx` with `CUDA_R_16BF` and
  `CUBLAS_COMPUTE_32F`.
- **rocBLAS:** `rocblas_gemm_ex` and `rocblas_gemm_strided_batched_ex` with
  `rocblas_datatype_bf16_r`. Note that `src/blas/backends/rocblas/rocblas_level3.cpp` already has
  `GEMM_EX_LAUNCHER(bfloat16, bfloat16, bfloat16, float, ...)`, which the public API does not
  declare.
- **oneMKL and the others:** the maintainers will know whether the oneAPI specification and
  oneMKL already define these forms. If they do, this is mostly API plumbing plus the vendor
  backends.

A separate pull request makes the existing `bfloat16, bfloat16 -> float` `gemm` work on the
cuBLAS backend, where it currently throws `unimplemented`: LINK_TO_PR.
