# L1: llm.c's main bf16 trainer, ported by Muse

llm.c's production trainer at `f1e2ace`: `train_gpt2.cu`, `test_gpt2.cu` and the CUDA headers
in `llmc/`. That is 12 CUDA files, 4,884 lines and 25 kernels, in bf16 mixed precision, with
cuBLASLt matmuls, warp reductions and stochastic rounding. The scope is a single device without
cuDNN, which are llm.c's defaults. Plan: [`docs/L1-PLAN.md`](../../docs/L1-PLAN.md).

**Result:** Muse's SYCL port passes llm.c's own test on the CPU device: logits, all 16 gradient
tensors, 10 training losses against PyTorch, and a bit-exact save, reload and replay, with
llm.c's own bf16 tolerances. It took **three runs, 299 turns and $14.34**, plus a $0.18 plan.
The third run continued from the second run's files. A clean rebuild outside the loop passes
too.

| Step | Result | Cost |
|---|---|---|
| `museport plan` (Muse) | 12 items; the grounding check caught one kernel name Muse made up ([plan](plan/plan.md)) | $0.18 |
| Prove the check (NVIDIA RTX 2000 Ada) | llm.c's CUDA test passes; a planted GELU bug fails ([check-proof.json](check-proof.json)) | $0.008 GPU |
| [Run 1](run1) | stopped by the toolkit's 2M-token limit after 34 turns of reading; SYCLomatic could not parse the trainer | $0.70 |
| [Run 2](run2) | 200 turns: the whole port written and compiling; forward pass right; training NaN | $11.79 |
| [Run 3](run3), continuing run 2's files | **success** in 65 turns: `overall okay: 1`, worst loss 0.0326 from PyTorch's | $1.86 |
| [Independent grade](independent-grade) | rebuilt outside the loop from Muse's files alone: PASS | $0 |

## The hidden check

llm.c's own test, as Muse ports it (`test_gpt2.cpp`), run by a wrapper outside the work dir
([`task/run_test.py`](task/run_test.py)). It writes three things, compared with a reference at
0.05, llm.c's bf16 loss tolerance:

1. 1.0 if the test exits 0 and prints `overall okay: 1`. That covers llm.c's logit check, its
   16 gradient checks, its 10 losses against PyTorch, and its save, reload and replay.
2. 1.0 if the test's source keeps llm.c's tolerances: logits 25, loss 0.05 and the 16 gradient
   thresholds. Every assignment to them is read, so a loosened test fails (tests in
   `tests/test_bf16.py`).
3. The 10 step losses, checked by the wrapper itself against PyTorch's.

**Proven before any model ran, on an NVIDIA RTX 2000 Ada:**

| Candidate | okay | Worst loss vs PyTorch | Result |
|---|---|---|---|
| llm.c's own CUDA test, bf16, `nvcc` | 1 | 0.0254 | PASS |
| planted bug: GELU constant 0.04715 ([diff](planted/gelu.diff)) | 0 | within 0.05 | FAIL |

The planted bug's losses stay within tolerance; only llm.c's own logit and gradient checks catch
it, which is why the check needs `overall okay`.

## How the three runs went

- **Run 1** (34 turns, $0.70) ran with SYCLomatic, which failed to parse the trainer:
  `nvtx3/nvToolsExt.h`, NVIDIA's profiling header, is not in its CUDA headers. Muse read the
  sources for 30 turns, and the toolkit's per-run limit of 2M tokens, which counts cached
  re-reads, stopped the run before it wrote anything. `museport loop` gained `--max-total-tokens`
  and `--no-migrate` for the next run.
- **Run 2** (200 turns, $11.79) had no SYCLomatic, and the USD cap set the limit.
  - **Writing:** Muse read everything and checked bf16 and oneMath's matmul layout in small
    test programs. It then wrote the port in turns 48-65: 12 SYCL kernel headers, a trainer in
    three parts, and the test.
  - **Debugging:** with no tool that runs a program, Muse compiled small debug programs as the
    candidate and read their printout in the check's failure message. That way it found and
    fixed an invalid oneMath argument and an attention bug, and a layernorm bug: a reduction
    over a 32-wide sub-group, CUDA's warp, which is narrower on the CPU.
  - **Where it stopped:** the step-1 loss reached 5.2505 against PyTorch's 5.2700, so the
    forward pass was right. It then traced the training NaNs to an out-of-bounds write into the
    gradient buffer, located in attention's backward pass, when the run reached its 200 turns.
- **Run 3** (65 turns, $1.86) started from run 2's files.
  - **What it was given:** the goal added a note, taken from run 2's own trace, on where run 2
    stopped. When the test failed, the check now also showed the test's own printout.
  - **Fixes:** Muse fixed the out-of-bounds write (turn 27: finite losses, but too slow), then
    the gradient scale (turn 47: all 10 losses right; one gradient, `lnfw`, at 103% of its
    threshold), then that gradient. From turn 52 the test passed.
  - **Finish:** Muse deleted its debug programs, and the run stopped on success at turn 65:
    compile, verify, and 0 helpers in the work dir, on the same sources.

## The independent grade

The check trusts what the compiled program prints, and the loop does not force that program to
be built from `test_gpt2.cpp`. J3's check has the same gap. So the result above is not taken
from the loop alone ([`independent-grade/`](independent-grade)):

- **Rebuild:** Muse's final files were copied to a clean folder and `test_gpt2.cpp` was built
  with `icpx -fsycl -O2 -DENABLE_BF16` and oneMath, with no task or SYCLomatic include path:
  14 s, no warnings.
- **The check on that build:** PASS. `overall okay: 1`, tolerances kept, and the worst loss
  0.0326 from PyTorch's ([grade.json](independent-grade/grade.json)).
- **llm.c's test output** ([test-output.txt](independent-grade/test-output.txt)):
  - logits OK, with the largest difference 9.46 against a tolerance of 25;
  - 16 of 16 gradient tensors OK;
  - 10 losses OK against PyTorch, and the 10 replayed losses bit-identical;
  - `overall okay: 1`, exit 0. The test took about 108 s on the CPU.
- **The code, read:**
  - `test_gpt2.cpp` is llm.c's test, with the same checks, thresholds and expected losses. It
    differs from `test_gpt2.cu` only in the trainer include, comments and small C++ changes
    (`std::max`, `fabsf`).
  - The model runs on the SYCL device: 32 SYCL kernels in `llmc/*.h`, and 18 oneMath BLAS calls
    for the matmuls.

## What else to know

- **Not one pass:** the port took three runs. Run 1 was stopped by our setup before Muse wrote
  any code; run 3 continued run 2's files.
- **The plain-C headers:** Muse also wrote its own copies of llm.c's plain-C headers
  (`utils.h`, `tokenizer.h`, `dataloader.h`, ...) instead of using them unchanged. Some are
  trimmed: `mfu.h`, for example, drops NVIDIA's monitoring library.
- **Two fixes to our check, recorded in [run2/NOTES.md](run2/NOTES.md):**
  - The wrapper crashed on a `-nan` loss. It was fixed mid-run, and since a NaN fails either way,
    no verdict changed.
  - From run 3, a failing test also shows its own printout.
- **Scope:** CPU only so far. One machine, one run of the test. bf16 stochastic rounding and
  llm.c's tolerances are as in llm.c. Speed was not compared.

**Muse cost:** $14.52 in all (plan $0.18 and runs $0.70, $11.79 and $1.86), of the $80 cap.
**GPU:** $0.008 of the $40 cap so far.
