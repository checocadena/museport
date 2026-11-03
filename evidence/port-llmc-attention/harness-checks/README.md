# Attention hidden checks: proven before Muse ran

llm.c's `dev/cuda/attention_forward.cu` and `attention_backward.cu` call cuBLAS
(`cublasSgemmStridedBatched`, `cublasGemmStridedBatchedEx`). Their specs are in
[`src/museport/llmc.py`](../../../src/museport/llmc.py) (`TIER3`). Each check was proven
with no model involved, as for the other Tier 3 items.

| Item | Variants (fp32) | Proof | Result |
|---|---|---|---|
| `t3-attention-forward` | 1-6 | llm.c's CUDA built from the harness with `nvcc -lcublas`, NVIDIA L4 | PASS, max abs err 2.38e-07 (atol 1e-3) |
| `t3-attention-backward` | 1-8 | llm.c's CUDA built from the harness with `nvcc -lcublas`, NVIDIA RTX A4000 | PASS, 7.15e-07 (atol 1e-3) |

The GPU runs ([`gpu-native/`](gpu-native/)) used `remote-test` with the hidden reference
compiled as a stand-in SYCL source; the step that counts is `native_verify_vs_reference`.
Three pods, each deletion confirmed; about $0.02 in total (estimates $0.01, $0.004, $0.008).

## What it took

- **SYCLomatic's raw output is not enough for either file** ([`cpu-c2s/`](cpu-c2s/)).
  - Forward: it does not compile (`use of undeclared identifier
    'cudaDevAttrMaxSharedMemoryPerBlock'`).
  - Backward: it compiles, and on the CPU device variants 1-7 pass and variant 8 fails
    (193,717 of 1,179,648 elements; max error 0.03).
- **Variant 8 of the backward looks racy.** Its loop over 4 rows reuses the shared
  `block_acc` array with no barrier between rows, so one warp can overwrite a sum another
  warp is still reading. Variant 7, the same reduction for one row, passes on the CPU device,
  which points to that loop. llm.c's CUDA passes on the GPU, where the timing hides it. A
  port that passes the hidden check (which runs on the CPU device) has to fix it.
- **The forward only builds in bf16 in llm.c.** In the fp32 build, `CUBLAS_LOWP` is
  undefined. The harness defines it as `CUDA_R_32F`, an option `common.h`'s own comment
  names, and `CUBLAS_LOWP_COMPUTE` as `CUBLAS_COMPUTE_32F` (new `extra_defines` spec field).
- **Backward sizes: T=256, not 128.** The first proof, at T=128, failed with llm.c's own
  CUDA (`backward-T128-nvidia-l4-fail/`). Variant 6 writes each row up to the next
  multiple of the block size (256). With T=128 that runs into the next row; llm.c's T=1024
  never does. Variant 6 also reads the upper triangle of `att`, which the forward pass never
  writes (in llm.c it is fresh `cudaMalloc` memory), so the harness zeroes `att` first.
- **cuBLAS in fp32** (`CUBLAS_DEFAULT_MATH`) in both, as for `trimat_forward`.
- **Variant 10 of the forward (cuDNN) is not built.** It needs `ENABLE_CUDNN` and cuDNN.

## Changes to llm.c's files (mechanical, in `src/museport/llmc.py`)

As in Tiers 1-3: fp32 build, CPU reference hidden, contract `main`. Each spec's `note`
lists its size changes; only the forward's `out` and the backward's `dinp` are compared,
the outputs llm.c checks for every variant.
