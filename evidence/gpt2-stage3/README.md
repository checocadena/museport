# GPT-2 training port, stage 3 (training): task, check, and Muse's run (Oct 27)

This is plan G. Stage 3 asks Muse to port llm.c's `train_gpt2_fp32.cu` far enough to
**train**: 10 steps of forward pass, backward pass and AdamW update. It starts from the real
GPT-2 124M weights and trains on llm.c's debug batch (B=4, T=64). It is the test that llm.c's
own `test_gpt2_fp32.cu` runs.

## The task ([`task/`](task/))

Written by `museport gpt2-task --stage train`.

- **Harness:** `gpt2_train_harness.cu` is llm.c's training file, unchanged behind `TESTING`.
  Its contract `main` runs 10 steps as `test_gpt2_fp32.cu` does: `gpt2_forward` with
  targets, `gpt2_zero_grad`, `gpt2_backward`, then
  `gpt2_update(lr 1e-4, beta1 0.9, beta2 0.999, eps 1e-8, weight decay 0.01, t = step + 1)`.
  It writes each step's mean loss: 10 float32 values.
- **Hidden reference:** PyTorch's 10 losses, as `test_gpt2_fp32.cu` and `test_gpt2.c` list
  them ("expected losses are as follows, from Python").
- **Tolerance:** 1e-2 absolute per step. Both llm.c tests use it.

## Proof that the check works ($0)

| Candidate | Result |
|---|---|
| llm.c's CPU `train_gpt2.c` with the same contract `main` ([`cpu-proof.json`](cpu-proof.json)) | **PASS**, worst error 0.0015 (atol 0.01) |
| The same, with the learning rate set to 1.1e-4 in place of 1e-4 ([`planted-bug-proof.json`](planted-bug-proof.json)) | **FAIL** on 9 of 10 steps, by up to 0.185 |

A 10% error in one optimizer setting fails the check. Step 0 still passes because it runs
before any update.

## Run 1: Muse passes stage 3 ([`run1/`](run1/))

`museport loop task/task.json --max-iterations 40 --max-cost 3` with `muse-spark-1.3`,
within the author's $7 GPT-2 cap.

| | |
|---|---|
| Loop result | **success** at turn 29 of 40 |
| Hidden grade of the final sources | **PASS**, worst error 0.000154 (atol 0.01); 0 helpers |
| Cost | **$1.1489** (from the API's reported usage) |
| Device | OpenCL CPU device (Intel Xeon), oneMath with its MKL-CPU backend |
| Benchmark | median 22,836 ms for the 10 training steps (5 runs, CPU) |

**The losses** (from the final verify's output files):

| Step | PyTorch | Muse's SYCL port | Error |
|---|---|---|---|
| 0 | 5.270007 | 5.270010 | 0.000003 |
| 1 | 4.059707 | 4.059662 | 0.000044 |
| 2 | 3.375123 | 3.375039 | 0.000084 |
| 3 | 2.800783 | 2.800668 | 0.000115 |
| 4 | 2.315382 | 2.315286 | 0.000096 |
| 5 | 1.849029 | 1.848921 | 0.000108 |
| 6 | 1.394656 | 1.394537 | 0.000119 |
| 7 | 0.999147 | 0.998992 | 0.000154 |
| 8 | 0.624080 | 0.624035 | 0.000045 |
| 9 | 0.376511 | 0.376496 | 0.000015 |

On this batch the port is closer to PyTorch than llm.c's CPU code is (0.0015). That was
measured, not explained.

- **What Muse wrote:** [`run1/port/gpt2_train.cpp`](run1/port/gpt2_train.cpp), 907 lines. It
  has 17 SYCL kernel launches, including the AdamW update on the device, and 9 oneMath `gemm`
  or `gemm_batch` calls. As in stages 1 and 2, it is a rewrite with simple kernels, not
  llm.c's tuned kernels.
- **The port does not read the answers.** The debug-state file also holds PyTorch's logits
  and gradients. The port reads only its header and the input and target tokens (`main`,
  lines 877-885). It loads the weights from `gpt2_124M.bin` and trains on the device. The 10
  expected losses are in neither file it reads.

### How the run went

1. **Turn 1:** `migrate` (SYCLomatic) wrote 6 files with 53 warnings.
2. **Turns 2-13: reading.** Muse read parts of the migrated code, then all of llm.c's
   original `train_gpt2_fp32.cu` (1,755 lines, in five ranges) and `llmc/utils.h`.
3. **Turn 14:** Muse wrote the whole port in one `write_file` (39,331 characters). It
   compiled on the first try.
4. **Turn 16:** verify failed on all 10 steps. Muse wrote: "Losses are far too high — my
   non-batched oneMath transpose mapping is wrong. Fixing the GEMM layouts."
5. **Turns 17-22:** Muse reread its matmul code and rewrote the file. It fixed the
   row-major to column-major transposes and leading dimensions of the three non-batched
   `gemm` calls, and replaced a barrier with `sycl::group_barrier`.
6. **Turn 24:** verify passed (0.000154).
7. **Turns 25-29:** Muse deleted `migrated/`, the scan found 0 helpers, it compiled without
   the dpct include path, verified again and ran the benchmark.

## Spend on GPT-2

Stage 1 cost $0.7441, stage 2 $1.2134 and stage 3 $1.1489: **$3.1064 of the $7 cap**.

## Limits

- **GPU:** the port, unchanged, also passes on an NVIDIA RTX 4000 Ada against PyTorch and
  llm.c's own CUDA, with oneMath built on the machine with its cuBLAS backend
  ([`gpu/NVIDIA.md`](gpu/NVIDIA.md)). AMD was not run.
- **One run, 10 steps, one batch.** It is llm.c's own test, not a longer training run.
- **Performance was not compared** with llm.c's CUDA.
