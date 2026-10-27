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

## Run 1: Muse passes stage 1 ([`run1/`](run1/))

`museport loop task/task.json --max-iterations 30 --max-cost 3` with `muse-spark-1.3`, on
Oct 27. The author approved $7 for the three stages, at most $3 each.

| | |
|---|---|
| Loop result | **success** at turn 28 of 30: verify passed and zero dpct helpers on the same compiled sources |
| Hidden grade of the final sources | **PASS**, worst max abs err 0.00121 (atol 0.01, llm.c's own test tolerance); 0 helpers |
| Cost | **$0.7441** (from the API's reported usage) |
| Device | OpenCL CPU device (Intel Xeon), oneMath with its MKL-CPU backend |
| Benchmark | median 2,351 ms for the two forward passes (5 runs, CPU) |

**What Muse wrote:** [`run1/port/gpt2_forward.cpp`](run1/port/gpt2_forward.cpp), 463 lines.
It is a forward-only rewrite of llm.c's GPT-2, not a translation of the whole training file:

- 9 SYCL kernels: encoder, layernorm, residual, GELU, attention softmax, the classifier and
  the helpers they need;
- oneMath `gemm` for the matmuls and `gemm_batch` for attention, in place of cuBLAS and of
  llm.c's own `matmul_forward_kernel4`.

The kernels are the simple forms, with one work-item per element or per row. llm.c's training
file uses tuned warp-level kernels (for example `layernorm_forward_kernel3`). Muse kept the
numbers, not those algorithms. Performance was not compared with llm.c.

**The port does not read the answers.** The input file also holds PyTorch's expected logits.
The port reads only its header and the input and target tokens (`main`, lines 432-439), and
computes the logits on the device.

**How the run went** (replay in [`run1/trace.jsonl`](run1/trace.jsonl)):

1. **Turns 1-15: reading.** Muse ran `migrate`, then read SYCLomatic's 2,404-line output in
   parts with `read_file` line ranges. It could not read llm.c's original
   `train_gpt2_fp32.cu`: the toolkit allows only the task's main CUDA file, not the files it
   includes. It worked from the migrated copy instead.
2. **Turn 16:** it wrote the port.
3. **Turn 17:** one compile error, the oneMath `transpose` enum's namespace.
4. **Turns 18-23:** the fix took six turns.
5. **Turn 24:** the port compiled.
6. **Turn 25:** Muse deleted `migrated/` with `delete_file`.
7. **Turns 26-28:** it recompiled, verify passed, the helper scan found 0 helpers, and it ran
   the benchmark.

## Next

- **Stage 2 (backward):** step-0 gradients for all 124M parameters, against PyTorch's.
- **Stage 3 (training):** 10 AdamW steps, against llm.c's expected losses.

Both stages stay on the CPU device. The NVIDIA GPU needs plan item D2 (oneMath built with its
cuBLAS backend). Spent so far on GPT-2: $0.7441 of the $7 cap.
