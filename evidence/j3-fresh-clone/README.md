# J3 step 5: `llm.c-sycl` from a fresh clone

This run followed the Quick start in [`llm.c-sycl/README.md`](../../llm.c-sycl/README.md)
exactly, on Oct 28. It started from a fresh `git clone` of this repository, on the CPU device
in the cloud session, at $0.

| Step | Result | Log |
|---|---|---|
| `./dev/download_starter_pack.sh` | llm.c's weights, debug state, tokenizer and tiny-shakespeare, in 10 s | [`download.log`](download.log) |
| `make` | both programs built, 26 s | [`make.log`](make.log) |
| `./test_gpt2fp32sycl` | logits OK, loss OK, gradients `TENSOR OK`, 10 losses OK, **`overall okay: 1`**, 34 s | [`test.log`](test.log) |
| `./train_gpt2fp32sycl` (llm.c's defaults: B=4, T=1024) | val loss 4.51, then about 38 s per training step out of 74; stopped after 4 min | [`train-first-4min.log`](train-first-4min.log) |

**The first attempt found a bug:** `dev/download_starter_pack.sh` was committed without its
executable bit, so `./dev/download_starter_pack.sh` failed with "Permission denied". The
fix is in this branch, and the table above is the rerun from a fresh clone of it.

**Edits to the logs:** in `download.log`, the session's scratch path is replaced by `<clone>`.

The trainer's losses were checked against llm.c's own CPU trainer in
[step 2](../j3-trainer/README.md), on a short run. These 4 minutes only show that the program
built from a fresh clone starts, loads, validates and trains with llm.c's default options.
