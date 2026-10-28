# J3 step 3: llm.c's own test, `test_gpt2_fp32.cu`, in SYCL

llm.c checks its CUDA trainer with `test_gpt2_fp32.cu`:

- It loads GPT-2 124M and a debug state that PyTorch wrote: inputs, logits, loss and every gradient.
- It runs the model and compares those, then trains 10 steps and compares each loss with PyTorch's.
- It prints `overall okay: 1` when everything matches.

This run asked Muse to port that test on top of its own trainer
([step 2](../j3-trainer/README.md)), which started in its work dir (`seed`).

## The hidden check

- **The run:** the wrapper `run_test.py` sits in the task dir, outside the work dir, so the
  model can't edit it. It runs the program next to the weights and the debug state.
- **What is compared:** 11 values. The first is whether the program printed
  `overall okay: 1` and exited 0, written as 1.0 or 0.0. The other 10 are the losses of its 10
  training steps.
- **The reference:** 1.0, then the 10 expected losses that llm.c's test hard-codes (PyTorch's).
- **Tolerance:** 1e-2, llm.c's own.

Both parts are needed. A bug can keep every loss within 1e-2 yet fail llm.c's logit or loss
checks. The planted bug below does exactly that.

**The check was proven at $0, on the CPU** ([`check-proof.json`](check-proof.json)):

| Candidate | Result |
|---|---|
| llm.c's own CPU test, `test_gpt2.c`, on `train_gpt2.c` | PASS: okay 1, worst loss error 0.0015 |
| planted bug: GELU coefficient 0.04715 instead of 0.044715 ([`planted/gelu_bug.diff`](planted/gelu_bug.diff)) | FAIL: okay 0, though its losses stay within 1e-2 |

## Result: success on run 3, $0.43

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Outcome | no test written in 20 turns | test written; blocked on the trainer's `main` | **success** in 24 turns |
| Cost | $0.49 | $0.70 | **$0.43** |
| Evidence | [`run1/`](run1/) | [`run2/`](run2/) | [`run3/`](run3/) |

Muse needed three runs. Here is what changed between them:

- **Run 1** spent all its turns reading the 1,117-line trainer.
- **Run 2's goal** named the trainer API the test uses. Muse wrote the test, then hit
  "redefinition of `main`": the test includes the trainer, which has its own `main`. The toolkit
  then had no way to change part of a file, so Muse re-read the trainer to rewrite it and ran out
  of turns.
- **For run 3** we added an `edit_file` tool to the toolkit
  ([#28](https://github.com/checocadena/oneapi-agent-toolkit/pull/28)). The goal now says to guard
  `main` with `#ifndef TESTING`, as llm.c does, using two small edits.

**Run 3:**

- **The guard:** Muse made the two edits. Those 2 lines are the only change to the trainer.
- **The test:** Muse wrote `test_gpt2_fp32.cpp`, 170 lines. It keeps llm.c's checks and
  tolerances: logits, loss and all gradients within 1e-2, and the 10 step losses within 1e-2.
- **Grading:** the first verify passed, with a worst error of 0.000154 and 0 helpers. The loop
  stopped on its success rule.
- **The trainer still passes its own check** after the guard: worst error 0.00227 (step 2's
  check, rebuilt and rerun at $0).

## Built from the repo

`make test_gpt2fp32sycl` in [`llm.c-sycl`](../../llm.c-sycl/README.md) builds the same source in
11 s. The hidden check passes on the CPU device (worst error 0.000154, 27 s). Its printout is in
[`repo-build-cpu.log`](repo-build-cpu.log): logits OK, loss OK, gradients `TENSOR OK`, 10 losses
OK, `overall okay: 1`.

## On NVIDIA and AMD GPUs (step 4)

These runs use the same source, unchanged: `run3/port`, as Muse last compiled it. Each machine
fetched llm.c's weights and debug state, checked their sha256, built oneMath for its vendor, and
ran the hidden check above.

| GPU | vs PyTorch (hidden reference) | vs llm.c's own CUDA test | Cost |
|---|---|---|---|
| NVIDIA RTX 2000 Ada (Runpod; oneMath on cuBLAS) | **PASS**, okay 1, worst 0.000127 | **PASS**, worst 7.03e-06 | $0.01 |
| AMD Instinct MI350X (DigitalOcean; oneMath on rocBLAS) | **PASS**, okay 1, worst 0.0000236 | not run: llm.c's CUDA does not hipify into something that builds, as for every llm.c file on AMD | $0.14 |

Results: [NVIDIA](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_test_gpt2_fp32/SUMMARY.md),
[AMD](gpu/amd/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/llmc_test_gpt2_fp32/SUMMARY.md). Logs are in [`gpu/logs`](gpu/logs).

The first AMD run failed before any test ran ($0.13). oneMath's rocBLAS backend did not
configure because `hip-config` could not find `AMDDeviceLibs`. GPT-2 stage 3 had fixed this on
its fourth AMD run, but that fix never reached `museport/gpt2.py`. It is there now, with a test,
and the second run passed.

So llm.c's own test, ported by Muse, prints `overall okay: 1` on three targets: the CPU, an
NVIDIA GPU and an AMD GPU.
