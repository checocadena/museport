# Tier 2 hidden checks: how each was proven before Muse ran

A hidden check is only worth something if it accepts a correct port and rejects a wrong
one. Each Tier 2 task was proven one of two ways, with no model involved:

| Item | Variants (fp32) | Proof | Result |
|---|---|---|---|
| `t2-global-norm` | 1-4 | SYCLomatic's raw migration on the CPU device | PASS, max abs err 1.19e-07 (atol 1e-2) |
| `t2-layernorm-forward` | 1-6 | SYCLomatic's raw migration on the CPU device | PASS, 1.19e-06 (atol 1e-5) |
| `t2-layernorm-backward` | 1-3, 5-10 | SYCLomatic's raw migration on the CPU device | PASS, 1.72e-05 (atol 1e-3) |
| `t2-softmax-forward` | 1-8 | llm.c's CUDA built from the harness with `nvcc`, NVIDIA RTX A4500 | PASS, 2.86e-05 (atol 1e-4) |
| `t2-classifier-fused` | 1-5 | llm.c's CUDA built from the harness with `nvcc`, NVIDIA RTX A4500 | PASS, 1.34e-05 (atol 1e-4) |

The GPU runs ([`gpu-native/`](gpu-native/)) used `remote-test` with the hidden reference
compiled as a stand-in SYCL source. The step that counts is `native_verify_vs_reference`.
Both pods were deleted; about $0.006 in total.

## Why two items needed the GPU proof

- **softmax_forward:** SYCLomatic's output does not compile (`no matching function for
  call to 'reduce_over_group'`, from the file's `cg::reduce`). That is a c2s limit.
- **classifier_fused:** SYCLomatic's output compiles but fails. Two separate causes, both in
  c2s's translation, not in the harness:
  - **Variants 1-4:** c2s turned every `expf` into `sycl::native::exp` (28 calls), whose
    precision is implementation-defined. On the CPU device the losses then miss llm.c's
    tolerance by about 2e-4. Replacing them with `sycl::exp` makes variants 1-4 pass
    (max abs err 1.34e-05, the same as CUDA on the GPU). `-fp-model=precise` alone did not
    help.
  - **Variant 5:** wrong after translation (losses off by units). CUDA's own variant 5
    passes the same reference on the GPU.

So a correct port must keep `expf`'s precision. c2s's default output does not.

## Changes to llm.c's files (mechanical, in `src/museport/llmc.py`)

As in Tier 1: fp32 build, CPU reference hidden, a contract `main`, and `common.h` without
cuBLAS. Tier 2 adds:
- **x20 outliers:** `softmax_forward` and `classifier_fused` apply llm.c's x20 outliers in
  both mains.
- **Padded vocabulary:** `classifier_fused` keeps llm.c's padded vocabulary, with the
  padding filled with `0xff`.
- **layernorm_backward:**
  - `mean` and `rstd` come from `layernorm_forward_cpu` on the host, as in llm.c, so that
    one CPU function stays visible.
  - `#include <cuda_bf16.h>` is added because llm.c only builds the file in bf16: its
    bf16-only kernel 4 uses `__nv_bfloat162` unguarded. Variant 4 is not run.
  - `C=768`, because `C=1600` trips variant 9's `C % (32 * x128::size)` assert in fp32.
- **global_norm:** 1024 device slots for kernel 4's block sums; only `out[0]` is compared.
