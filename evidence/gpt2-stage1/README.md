# GPT-2 training port, stage 1 (forward): the task and its hidden check (Oct 27)

This is plan G (see [`docs/GPT2-PLAN.md`](../../docs/GPT2-PLAN.md)). Stage 1 asks Muse to port
llm.c's `train_gpt2_fp32.cu` far enough to run the forward pass. The pass runs on llm.c's
debug batch (B=4, T=64) with the real GPT-2 124M weights. **No model has run yet:** this
directory holds the task and the proof that its check works.

## The task ([`task/`](task/))

Written by `museport gpt2-task --llmc <llm.c> --out task`.

- **Harness:** `gpt2_forward_harness.cu` is llm.c's training file, unchanged, included with
  llm.c's own `TESTING` switch. It adds a contract `main` that sets up cuBLAS as llm.c's
  `test_gpt2_fp32.cu` does (TF32 off). It writes raw float32: the B×T×V logits of a target-free
  forward pass, then the mean loss of a forward pass with targets.
- **Hidden reference:** `gpt2_ref.cpp` reads PyTorch's own logits and loss for the same batch
  from `gpt2_124M_debug_state.bin`. The check is llm.c's own tolerance, 1e-2 absolute, the one
  `test_gpt2_fp32.cu` uses.
- **Data:** `gpt2_124M.bin` (498 MB) and `gpt2_124M_debug_state.bin` (549 MB) are linked from
  the llm.c checkout, not committed. llm.c's `dev/download_starter_pack.sh` fetches them from
  Hugging Face.
- **Library calls:** cuBLAS calls are to go to oneMath, so the compile flags link `-lonemath`.

## Proof that the check works, before any model runs ($0)

| Candidate | Result |
|---|---|
| llm.c's CPU `train_gpt2.c` with the same contract `main` (`museport gpt2-task --check`; output in [`cpu-proof.json`](cpu-proof.json)) | **PASS**, worst max abs err 0.000839 (atol 0.01). Loss 5.269889 against PyTorch's 5.270009 |
| SYCLomatic's raw migration of the CUDA harness, with 11 `sycl::native::` math calls replaced by precise ones (oneMKL, OpenCL CPU device) | **PASS**, worst max abs err 0.0013 |
| The same raw migration as SYCLomatic wrote it (`sycl::native::exp`) | **FAIL**: 14 of the 256 rows are off by up to 0.076. The loss still matches |
| A copy of the CPU output with one logit moved by 0.02 | **FAIL**: 1 of 12,865,793 elements outside tolerance |

The second and third rows are a manual check made in this session, not stored as files.
SYCLomatic's raw output fails only because of its fast-math mapping. That shows the check is
strict enough to catch imprecise math, and why the goal asks for precise math.

## Next

Stage 1 runs with Muse once the author approves the cap. The proposed cap is $7 for the three
stages, at most $3 per stage. The run is on the CPU device. NVIDIA needs plan item D2 (oneMath
built with its cuBLAS backend).
