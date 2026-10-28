# J3 step 2: llm.c's real fp32 trainer, in SYCL

GPT-2 stage 3 ([`../gpt2-stage3`](../gpt2-stage3/README.md)) proved Muse's port of llm.c's GPT-2
model (forward, backward, AdamW, with oneMath) behind a test `main`. That port passed on a CPU,
an NVIDIA GPU and an AMD GPU.

This run asked Muse to turn it into llm.c's real trainer, `train_gpt2_fp32.cu`. That means its
command line, the train and val DataLoaders, the validation loss, sampling with the tokenizer, and
its printout. Muse's work dir started with its own proven port (`task/seed/`).

## The hidden check

- **The run:** the program trains on llm.c's tiny-shakespeare data, cut to 10 training and 5
  validation batches (B=4, T=64) in llm.c's own token format (`task/data/`), with fixed arguments
  (`task/train_args.txt`).
- **What is compared:** the validation loss before training, the 10 training losses, and the
  validation loss after.
- **The reference:** llm.c's own CPU trainer, `train_gpt2.c`. It uses the same dataloader,
  shuffle and AdamW, and its fixed 40-step loop is cut to 10. It ran once and its losses were
  recorded (`task/reference.out`); `task.json` keeps the recipe under `museport.reference`.
- **Tolerance:** 1e-2, llm.c's own test tolerance.
- **The wrapper:** `run_train.py` runs a program in a scratch dir laid out as llm.c expects. It
  sits in the task dir, outside the work dir, so the model cannot edit it. This needs toolkit
  [#27](https://github.com/checocadena/oneapi-agent-toolkit/pull/27).

**The check was proven alongside the run, at $0, on the CPU** ([`check-proof.json`](check-proof.json), planted sources in [`planted/`](planted/)):

| Candidate | Result |
|---|---|
| llm.c's own `train_gpt2.c`, the same 10 steps | PASS, worst error 0 |
| planted bug: learning rate 3e-4 instead of 1e-4 (what ignoring `-l` would do) | FAIL, 10 of 12 values out of tolerance |
| planted bug: training data not shuffled | FAIL, 11 of 12 values out of tolerance |

## Result: success, first verify, $0.71

| | |
|---|---|
| Loop | **success** in 19 turns: verify passed, with 0 helpers on the same compiled sources |
| Hidden grade after the run | PASS, worst error 0.00227 (atol 1e-2), 0 helpers |
| Cost | **$0.7065** (338,370 input and 38,462 output tokens) |
| Port | [`run1/port/train_gpt2_fp32.cpp`](run1/port/train_gpt2_fp32.cpp), 1,117 lines |

- **Model code:** Muse kept **851 of its 907 proven lines verbatim (94%)** and added llm.c's
  `main`: the same `-i -j -o -b -t -l -v -m -s -g` options, `gpt2_build_from_checkpoint`, both
  DataLoaders with llm.c's shuffle settings, `tokenizer_init`, validation every `-v` steps and at
  the last step, and `sample_softmax` sampling.
- **llm.c's host headers** are included unchanged.
- **Its first verify passed;** the run ended on the loop's success rule.
- **The same source, built by `llm.c-sycl`'s own Makefile** (`make` at the top level), passes the
  same check: worst error 0.00227 in 47 s on the CPU device.

## On NVIDIA and AMD GPUs (step 4)

These runs use the same source, unchanged: `run1/port`. Each machine fetched llm.c's weights and
tokenizer, checked their sha256, built oneMath for its vendor, and ran the hidden check above.

| GPU | vs llm.c's CPU trainer (hidden reference) | vs llm.c's own CUDA trainer | Cost |
|---|---|---|---|
| NVIDIA RTX 2000 Ada (Runpod; oneMath on cuBLAS) | **PASS**, worst 0.0023 | **PASS**, worst 0.00147 | $0.009 |
| AMD Instinct MI350X (DigitalOcean; oneMath on rocBLAS) | **PASS**, worst 0.00242 | not run: llm.c's CUDA does not hipify into something that builds | $0.143 |

On NVIDIA, llm.c's own CUDA trainer is itself 0.00386 from the CPU reference.

Results: [NVIDIA](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_train_gpt2_fp32/SUMMARY.md),
[AMD](gpu/amd/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/llmc_train_gpt2_fp32/SUMMARY.md). Logs are in [`gpu/logs`](gpu/logs).

llm.c's own test, ported in step 3, is in [`../j3-test`](../j3-test/README.md).

## Not yet done

- A longer training run (K1).
