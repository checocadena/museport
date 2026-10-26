# Target survey: karpathy/llm.c `dev/cuda` (prep, 2026-10-26)

Source: https://github.com/karpathy/llm.c at `f1e2ace` (2025-05-10), MIT license.
`dev/cuda/` holds 22 standalone kernel files (about 11,100 lines). Each one (except NCCL)
contains several GPU versions of one kernel, a **CPU reference** (`*_cpu`) and
`validate_result`, which compares GPU output with the CPU reference. That gives every kernel
a ready-made correctness check, which is what Muse Port's hidden grader needs.

Shared assumptions in `common.h`, included by every file: `cublas_v2.h` and `cublasLt.h`,
`#define WARP_SIZE 32U`, and warp shuffles (`__shfl_xor_sync`) in the reductions. SYCL does
not guarantee a sub-group size of 32, so these must be ported deliberately, not mechanically.

## Kernels by difficulty

Columns: lines; GPU versions (`case N:`); uses cuBLAS/cuBLASLt beyond `common.h`; cooperative
groups; bf16.

**Tier 1: elementwise or simple, no library calls** (good first targets)

| File | Lines | Versions | Notes |
|---|---|---|---|
| gelu_forward.cu | 173 | 2 | bf16 in one version |
| gelu_backward.cu | 195 | 2 | |
| residual_forward.cu | 167 | 2 | |
| encoder_forward.cu | 259 | 3 | |
| encoder_backward.cu | 202 | 2 | atomics |
| crossentropy_forward.cu | 147 | 1 | |
| crossentropy_softmax_backward.cu | 164 | 1 | |
| adamw.cu | 226 | 2 | optimizer step |
| permute.cu | 181 | 0 (`permute_kernel`) | index math |

**Tier 2: reductions, warp shuffles, cooperative groups**

| File | Lines | Versions | Notes |
|---|---|---|---|
| softmax_forward.cu | 732 | 8 | warp reductions, online softmax |
| layernorm_forward.cu | 628 | 6 | cooperative groups |
| global_norm.cu | 269 | 4 | cooperative groups |
| fused_residual_forward.cu | 695 | 6 | |
| matmul_backward_bias.cu | 669 | 8 | cooperative groups, bf16 |
| classifier_fused.cu | 771 | 5 | cooperative groups |
| layernorm_backward.cu | 1,550 | 10 | heavy bf16 and cooperative groups |

**Tier 3: NVIDIA libraries (the library-migration story)**

| File | Lines | Library calls | oneAPI equivalent |
|---|---|---|---|
| matmul_forward.cu | 439 | cuBLAS, cuBLASLt (67 mentions) | oneMath BLAS (`gemm`); bias + GELU epilogue by hand |
| matmul_backward.cu | 281 | cuBLAS | oneMath BLAS |
| trimat_forward.cu | 650 | cuBLAS | oneMath BLAS + custom causal mask |
| attention_forward.cu | 1,390 | cuBLAS and **cuDNN** | oneMath BLAS; cuDNN version out of scope (oneDNN, not attempted) |
| attention_backward.cu | 1,154 | cuBLAS, cooperative groups | oneMath BLAS |

**Out of scope:** `nccl_all_reduce.cu` (multi-GPU NCCL + MPI).

## Plan for Muse Port

- Start with Tier 1 to prove the pipeline end to end, then Tier 2 (the warp-size problem),
  then one Tier 3 kernel (matmul_forward: cuBLAS → oneMath) for the library story.
- Each `dev/cuda` file becomes a task: its CPU reference is the reference program, the
  ported kernel is the candidate, both write the same output arrays, and the toolkit's
  `verify` compares them. Muse never sees the grader.
- Risks: bf16 paths (port the fp32 versions first); `WARP_SIZE 32` reductions; several
  versions per file (port one chosen version per kernel, not all of them).
