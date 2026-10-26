# Tier 3 hidden checks: proven before Muse ran

Tier 3 files call cuBLAS/cuBLASLt. Ports call oneMath (UXL), which here runs on its
MKL-CPU backend. oneMKL and oneMath come from the toolkit's
`install-toolchain.sh` with `WITH_ONEMATH=1`; a oneMath SGEMM sanity check passes,
max abs err 5.72e-06.

| Item | Variants | Proof | Result |
|---|---|---|---|
| `t3-matmul-backward` | 1 | SYCLomatic's raw migration, linked with oneMKL, on the CPU device | PASS, max abs err 2.1e-04 (atol 1e-3) |
| `t3-matmul-backward-bias` | 1-5, 7-9 | llm.c's CUDA (`nvcc`), NVIDIA RTX 4000 Ada | PASS, 1.91e-05 (atol 5e-3) |
| `t3-matmul-forward` | 1-4 | llm.c's CUDA with cuBLAS + cuBLASLt (`nvcc -lcublas -lcublasLt`), NVIDIA | PASS, 0.0121 (atol 1e-1, TF32 on as in llm.c) |
| `t3-trimat-forward` | 0-4 | llm.c's CUDA with cuBLAS (`nvcc -lcublas`), fp32 math, NVIDIA RTX 4000 Ada | PASS, 5.96e-07 (atol 1e-4) |

GPU runs: [`gpu-native/`](gpu-native/); logs in [`gpu-native/logs/`](gpu-native/logs/).
Every pod's deletion was confirmed; about $0.07 for the five runs.

## What it took

- **c2s on the CPU device was not enough for three items.**
  - `matmul_backward_bias`: variants 4 and 7 return zeros.
  - `trimat_forward`: the output does not compile (`no matching function` at line 614).
  - `matmul_forward`: cuBLASLt maps to a oneDNN-based dpct helper, and oneDNN is not
    installed.

  llm.c's own CUDA on a GPU proves all three.
- **Toolkit change: `native_flags`** ([CHANGES-TO-TOOLKIT](../../../docs/CHANGES-TO-TOOLKIT.md)).
  The toolkit's native `nvcc` build linked only the CUDA runtime, so CUDA that calls cuBLAS
  failed to link (`undefined reference to cublasCreate_v2`; `harnesscheck3` logs). A task
  can now list link flags for that build.
- **TF32.** llm.c's `setup_main` turns TF32 on from Ampere. With it, llm.c's own 1e-4 check
  of `trimat_forward`'s cuBLAS variant 0 fails on an RTX 2000 Ada: 49,484 elements off by up
  to 2e-4 (`harnesscheck3b`). That harness now runs cuBLAS in fp32 (`CUBLAS_DEFAULT_MATH`),
  as `matmul_backward`'s own main does. `matmul_forward` keeps TF32 with llm.c's 1e-1.

## Changes to llm.c's files (mechanical, in `src/museport/llmc.py`)

As in Tiers 1-2: fp32 build, CPU reference hidden, contract `main`. Tier 3 adds:
- **common.h:** keeps cuBLAS for files that call it, and cuBLASLt only for
  `matmul_forward`.
- **`#include <omp.h>`** is removed from the harness; only the hidden CPU reference used
  it.
- **`trimat_forward`:** the causal-masked upper triangle is zeroed in both mains; llm.c
  writes NaN there and skips it.
- **Scratch buffers:** `matmul_backward_bias`'s `dbias_buffer` and `trimat_forward`'s
  `d_qkvr` are allocated as in llm.c's `main`.
- **Out of scope:** `nccl_all_reduce` needs MPI and NCCL on several GPUs. The two attention
  files (1150-1390 lines) are not specified yet.
