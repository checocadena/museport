# Muse ports llm.c's Tier 3 kernels: cuBLAS and cuBLASLt calls to oneMath

`museport port --tier 3` on 4 of the 7 Tier 3 items in Muse's
[plan](../plan-llmc-devcuda-stream/README.md), with `muse-spark-1.3`. The ports call
**oneMath** (UXL Foundation, `oneapi::math::blas`) where llm.c calls cuBLAS. Each hidden
check was [proven first](harness-checks/README.md).

## Results

| Item | Variants | Loop result (turns) | Muse cost | Hidden grade (final sources, CPU) | Kernels + oneMath calls / variants | NVIDIA GPU: vs reference / vs original CUDA |
|---|---|---|---|---|---|---|
| `t3-matmul-backward` | 1 | max_iterations (16) | $0.2795 | PASS, 0.00021, 0 helpers | 2 + 2 / 1 | not run: calls oneMath, which the GPU hosts do not have |
| `t3-trimat-forward` | 5 | success (9) | $0.2112 | PASS, 7.15e-07, 0 helpers | 5 + 1 / 5 | not run: calls oneMath, which the GPU hosts do not have |
| `t3-matmul-backward-bias` | 8 | budget (14) | $0.5007 | PASS, 1.53e-05, 0 helpers | 12 / 8 | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_matmul_backward_bias/SUMMARY.md) on RTX 2000 Ada Generation: 7.75e-06 / 1.34e-05 |
| `t3-matmul-forward` | 4 | success (15) | $0.4046 | PASS, 5.72e-05, 0 helpers | 3 + 2 / 4 | not run: calls oneMath, which the GPU hosts do not have |

- **Muse spend:** $1.3960 (16 turns max each, $1 cap per item). NVIDIA: 1 run, about
  $0.004, pod deletion confirmed.
- **Hidden grade: 4/4 correct with 0 helpers.** 2/4 also met the loop's strict rule. Of
  the other two, `matmul_backward` ran out of turns and `matmul_backward_bias` hit the token
  budget.
- **cuBLAS to oneMath.** Every port of a file that calls cuBLAS calls
  `oneapi::math::blas::column_major::gemm`, and none uses a oneMKL product API.
  `matmul_forward` keeps llm.c's four variants apart:
  - variant 1: a naive kernel;
  - variant 2: `cublasSgemm` + bias becomes oneMath `gemm` + a bias kernel;
  - variant 3: cuBLASLt's bias epilogue becomes oneMath `gemm` + a bias kernel (oneMath has
    no epilogue);
  - variant 4: the hand-written tiled kernel.
- **Variant check** now counts oneMath calls as well as distinct kernels, because a cuBLAS
  variant becomes a library call, not a kernel. No Tier 3 port copies variants.
- **GPU:** only `matmul_backward_bias` (no library) ran on NVIDIA; it passes against the
  reference and against llm.c's CUDA. The other three were verified on the CPU device
  through oneMath's MKL-CPU backend. On a GPU they need oneMath built with its cuBLAS (or
  HIP) backend on that machine, which the toolkit's GPU bundle does not do yet.

## Not attempted

- `t3-attention-forward` and `t3-attention-backward` (1150-1390 lines; batched gemms,
  one cuDNN variant): no spec written yet.
- `t3-nccl-all-reduce`: needs MPI and NCCL on several GPUs (oneCCL in SYCL).

## Limits

- fp32 only; reduced sizes (per task in `task.json`, `museport.note`).
- One run per item, one model; AMD not run.
- `trimat_forward`'s cuBLAS variant runs in fp32 in the harness. With TF32, as llm.c's
  `setup_main` sets from Ampere, llm.c's own 1e-4 check of that variant fails.

## Files

Same layout as Tiers 1-2; `harness-checks/` holds the proofs and the `native_flags`
toolkit change.
