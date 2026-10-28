# J4: a second, unseen CUDA repo (one 772-line file), end to end: llama2.cu

[ankan-ban/llama2.cu](https://github.com/ankan-ban/llama2.cu) (MIT) at `cb0c6fa`: a CUDA port
of Karpathy's llama2.c, Llama 2 inference in one file, `llama2.cu` (772 lines, 8 kernels, fp16,
CUB block and warp reductions). Muse Port had never run on it. The plan is
[`docs/J4-PLAN.md`](../../docs/J4-PLAN.md).

| Step | Result | Cost |
|---|---|---|
| `museport scan` | 1 CUDA file, 8 kernels, CUB (5 uses), shared memory (6), fp16 | $0 |
| `museport plan` (Muse) | 1 item, the whole file; grounded, 0 issues ([plan](plan/plan.md)) | $0.029 |
| Record llama2.cu on NVIDIA | 2 stories, two runs bit-identical ([record](record)) | $0.003 |
| Prove the check on the CPU | llama2.c's `run.c` passes; 2 planted bugs fail ([check-proof.json](check-proof.json)) | $0 |
| Muse ports `llama2.cu` | **success** in 20 turns: 2 of 2 stories identical, 0 helpers ([run1](run1)) | $0.45 |
| The port on NVIDIA RTX 2000 Ada | identical to the recording and to llama2.cu live ([gpu/nvidia](gpu/nvidia)) | $0.004 |
| The port on AMD MI350X | identical to the recording ([gpu/amd](gpu/amd)) | $0.127 |
| `llama2-sycl/` from a fresh clone | `make llama2`, both stories identical ([j4-fresh-clone](../j4-fresh-clone/README.md)) | $0 |

The 128-steps recording ([record-128steps](record-128steps)) cost another $0.003.

## The hidden check

- **The run:** greedy decoding (temperature 0) with Karpathy's `stories15M` model, 64 steps,
  two prompts: "Once upon a time" and "Tom and his dog went to the park".
- **What is compared:** the generated story, byte for byte. The wrapper,
  [`task/run_story.py`](task/run_story.py), cuts it from the program's printout, between the
  `<s>` line and the final `achieved tok/s` line. It sits in the task dir, outside the model's
  work dir.
- **The reference:** llama2.cu's own CUDA program, built with `nvcc` and recorded on an NVIDIA
  RTX 2000 Ada (J1's `record_native`). It was run twice and the runs were identical.
- **Why 64 steps:** at 128 steps, llama2.c's fp32 `run.c` (correct, but rounding differently
  from llama2.cu's fp16) flips a near-tie token at byte 322 of the first story. The step count
  was changed to 64 before any model ran.

**The check was proven at $0, on the CPU** ([`check-proof.json`](check-proof.json)):

| Candidate | Result |
|---|---|
| llama2.c's fp32 `run.c` (known good) | PASS: both stories identical |
| planted bug: RMSNorm epsilon 1e-2 instead of 1e-5 ([diff](planted/eps.diff)) | FAIL: differs at bytes 16 and 33 |
| planted bug: attention scores divided by sqrt(2·head_size) ([diff](planted/scale.diff)) | FAIL: differs at bytes 87 and 53 |

## Muse's port

- **Run:** `museport loop` with a $3 cap and 30 turns ([`run1/run.log`](run1/run.log),
  [trace](run1/trace.jsonl), [goal](run1/goal.txt)).
- **How it went:** Muse ran SYCLomatic, read its output, then wrote a 737-line `llama2.cpp`
  in one turn (turn 6). That port compiled at once. Muse deleted SYCLomatic's output, rewrote its
  barriers in SYCL 2020 style, and then ran verify once, on turn 20: 2 of 2 stories identical.
- **The port:** [`run1/port/llama2.cpp`](run1/port/llama2.cpp). It keeps fp16 storage
  (`sycl::half`) and fp32 accumulation, turns CUB reductions into SYCL group algorithms, and uses
  0 dpct helpers.
- **Cost:** $0.45 (205k input tokens, 557k cached, 26k output).

## GPUs

The GPU runs use the port as last compiled by Muse, unchanged
(`museport.llama2.make_gpu_task`). gpu-test pins the SYCL device to the GPU
(`ONEAPI_DEVICE_SELECTOR=cuda:0` on NVIDIA).

- **NVIDIA RTX 2000 Ada:** both stories identical to the recording, and to llama2.cu built
  with `nvcc` and run on the same GPU. Whole-program time for 64 steps, including loading the
  model: median 430 ms for the port and 378 ms for llama2.cu, 3 runs each. This is a rough
  number, not a tuned comparison.
- **AMD Instinct MI350X** (DigitalOcean): both stories identical to the recording. Median 412 ms
  per 64-step run, 3 runs. llama2.cu itself was not run on AMD: its automatic HIP conversion
  did not build, because of a byte-order mark at the top of `llama2.cu` and no `hipcub`
  on the machine ([native_build.json](gpu/amd/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/llama2_story/results/native_build.json)).

## `llama2-sycl`

[`llama2-sycl/`](../../llama2-sycl/README.md) is llama2.cu's repo with `llama2.cu` replaced by
Muse's `llama2.cpp`, unchanged, and a `llama2` Makefile target for `icpx -fsycl`. That target
was written by hand, not by Muse. `museport llama2-sycl assemble` regenerates it, and a test
checks that its `llama2.cpp` is the graded port. From a fresh clone, its Quick start works on
the CPU ([evidence](../j4-fresh-clone/README.md)). Its Makefile's GPU targets have not been run.

**Total J4 cost:** Muse $0.48 (plan and port) of the $8 cap; GPU $0.137 (recordings $0.006, NVIDIA $0.004, AMD $0.127).
