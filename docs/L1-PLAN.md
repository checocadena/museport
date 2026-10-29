# L1: llm.c's main bf16 trainer, in one Muse run (plan, Oct 29)

**Goal:** Muse ports llm.c's production trainer from CUDA to SYCL in one unattended run, graded
by llm.c's own test. Then the port trains on NVIDIA, AMD and, with access, an Intel GPU.

**Caps (the author's, Oct 29):** Muse $80 and GPU $40 for L1 and what follows it (speed tuning,
the benchmark, the oneMath fix). They replace the earlier comparison cap for this work.

## What is ported

`train_gpt2.cu`, `test_gpt2.cu` and llm.c's `llmc/` headers, at llm.c `f1e2ace`: 12 CUDA files,
4,884 lines, 25 kernels (`museport scan`).

| Kept | Left out (as llm.c's defaults or a single GPU allow) |
|---|---|
| bf16 mixed precision (`PRECISION=BF16`, llm.c's default) | cuDNN flash attention (`USE_CUDNN=0`, llm.c's default) |
| cuBLASLt and cuBLAS matmuls, to oneMath or SYCL kernels | multi-GPU: NCCL, MPI and ZeRO sharding (`NO_MULTI_GPU=1`) |
| llm.c's options, dataloaders, validation, sampling, checkpoints | |

The work dir is llm.c's tree: Muse writes `train_gpt2.cpp`, `test_gpt2.cpp` and SYCL versions of
the `llmc/` CUDA headers. The plain-C headers (`utils.h`, `tokenizer.h`, `dataloader.h`, ...) are
used as they are.

## The hidden check

llm.c's own test, `test_gpt2.cu`, ported by Muse as `test_gpt2.cpp`, run by a wrapper outside
the work dir:

1. **The test passes:** exit 0 and `overall okay: 1`. That covers the logits, every gradient,
   10 losses against PyTorch and a save, reload and replay that must be bit-identical.
2. **The losses, checked by the wrapper itself:** the 10 step losses against PyTorch's, within
   0.05, llm.c's bf16 tolerance.
3. **The test's tolerances are llm.c's:** the wrapper reads `test_gpt2.cpp` and requires llm.c's
   thresholds: logits 25.0, loss 0.05 and the 16 gradient thresholds. A port that loosens
   them fails.

**Proven before any model runs, on an NVIDIA GPU:** llm.c's own CUDA test, built with `nvcc`,
passes through the same wrapper, and a planted bug in llm.c's CUDA fails it. There is no
known-good bf16 SYCL program to prove it with on the CPU.

## Steps and cost

| Step | Muse | GPU |
|---|---|---|
| 1. `museport plan` on the 12 files | ~$0.3 | |
| 2. Prove the check: llm.c's CUDA test passes, a planted bug fails (NVIDIA) | | ~$0.1 |
| 3. **One Muse run** ports all 12 files; the check runs on the CPU device | cap $15 | |
| 4. If step 3 fails: report why, then a staged port (forward, backward, update, then `main`) | up to ~$20 | |
| 5. The port on NVIDIA (vs PyTorch and vs llm.c's CUDA) and AMD | | ~$2 |
| 6. Training curves on NVIDIA, AMD (and Intel with access) vs llm.c's CUDA | | ~$5-10 |

## Risks

- **Size:** about 4,900 lines; Muse's largest single port so far is 907 lines (GPT-2 stage 3).
- **bf16:** `sycl::ext::oneapi::bfloat16`, and bf16 matmuls in oneMath's backends (MKL on the
  CPU, cuBLAS, rocBLAS) are not yet tested here.
- **cuBLASLt** has no direct oneAPI equivalent: its fused bias and GELU epilogues become a gemm
  plus kernels.
- **Stochastic rounding** and the test's bit-exact replay need a deterministic port.
