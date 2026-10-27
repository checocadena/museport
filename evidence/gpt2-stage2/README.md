# GPT-2 training port, stage 2 (backward): task, check, and Muse's run (Oct 27)

This is plan G. Stage 2 asks Muse to port llm.c's `train_gpt2_fp32.cu` far enough to run
the forward pass and then the **backward pass**: the gradient of every one of GPT-2 124M's
124,475,904 parameters on llm.c's debug batch (B=4, T=64).

## The task ([`task/`](task/))

Written by `museport gpt2-task --stage backward`.

- **Harness:** `gpt2_backward_harness.cu` is llm.c's training file, unchanged behind
  `TESTING`. Its contract `main` runs the forward pass with targets, zeroes the gradients, runs
  the backward pass, and writes every gradient, then the loss.
- **Hidden reference:** PyTorch's gradients and loss from `gpt2_124M_debug_state.bin`.
- **Tolerance: 2e-2 absolute.** That is llm.c's CPU test's tolerance for gradients
  (`test_gpt2.c`, `check_tensor`). The CUDA test (`test_gpt2_fp32.cu`) uses 1e-2. At 1e-2,
  **llm.c's own CPU `train_gpt2.c` fails** on 1 of the 124,475,904 gradients: an `fcw` element
  where PyTorch has 0.96353 and llm.c's CPU code computes 0.95199, a difference of 0.0115. A
  check that rejects llm.c itself is stricter than llm.c, so this stage uses llm.c's CPU
  gradient tolerance.

## Proof that the check works ($0)

| Candidate | Result |
|---|---|
| llm.c's CPU `train_gpt2.c` with the same contract `main` ([`cpu-proof.json`](cpu-proof.json)) | **PASS**, worst max abs err 0.0115 (atol 0.02) |
| PyTorch's own gradients with one gradient moved by 0.05 | **FAIL**: 1 of 124,475,905 elements outside tolerance |

## Run 1: Muse passes stage 2 ([`run1/`](run1/))

`museport loop task/task.json --max-iterations 40 --max-cost 3` with `muse-spark-1.3`, within
the author's $7 GPT-2 cap.

| | |
|---|---|
| Loop result | **success** at turn 27 of 40 |
| Hidden grade of the final sources | **PASS**, worst max abs err 0.0124 (atol 0.02); 0 helpers |
| Cost | **$1.2134**: reported usage, plus turn 9 charged at its worst case (below) |
| Device | OpenCL CPU device (Intel Xeon), oneMath with its MKL-CPU backend |
| Benchmark | median 4,397 ms for the forward and backward passes (5 runs, CPU) |

- **What Muse wrote:** [`run1/port/gpt2_sycl.cpp`](run1/port/gpt2_sycl.cpp), 764 lines: 16
  SYCL kernels and 17 oneMath `gemm` or `gemm_batch` calls for the forward and backward passes.
  As in stage 1, it is a rewrite with simple kernels, not llm.c's tuned kernels.
- **The port does not read the answers.** The input file also holds PyTorch's gradients. The
  port reads only its header and the input and target tokens (`main`, lines 740-747). It
  loads the weights from `gpt2_124M.bin` and computes every gradient on the device.
- **Its worst error is above llm.c's CUDA tolerance.** At 0.0124, it would fail the CUDA test's
  1e-2, as llm.c's own CPU code does (0.0115).

### How the run went

1. **Turns 1-12: reading.** Muse read llm.c's **original** `train_gpt2_fp32.cu` in parts and
   `llmc/utils.h`. Toolkit #23 made that possible; in stage 1 Muse could read only
   SYCLomatic's translation.
2. **Turn 9: lost connection.** The connection dropped while the answer streamed. The loop
   marked the turn and carried on (museport #5). The API reported no usage for it, so it is
   charged as its full output budget.
3. **Turn 13:** Muse wrote the whole port (32,858 characters).
4. **Turn 14:** 20 compile errors.
5. **Turns 15-21:** Muse read the failing parts, rewrote the file, and it compiled.
6. **Turn 22:** verify passed (0.0124).
7. **Turns 23-27:** the helper scan found SYCLomatic's leftovers. Muse deleted `migrated/`,
   rescanned (0 helpers), recompiled, verified again and benchmarked.

## Spend on GPT-2

Stage 1 $0.7441 plus stage 2 $1.2134 comes to **$1.9575 of the $7 cap**.

## Next

**Stage 3:** 10 AdamW training steps, checked against llm.c's expected losses.
