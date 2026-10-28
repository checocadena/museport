# J4: a second, unseen CUDA repo, end to end (plan, Oct 28)

**Done Oct 28:** Muse's port succeeded on its first run, and the check moved from 128 to 64
steps before any model ran (see the evidence). Muse cost $0.48 and GPU $0.137;
results in [`evidence/j4-llama2`](../evidence/j4-llama2/README.md).

**Goal:** the J1-J3 pipeline on a CUDA repo it has never seen, with no hand-written hidden check
and a reassembled SYCL repo at the end. The steps are scan, Muse's plan, a recorded check, Muse's
port, GPU runs, and reassembly.

**Repo (the author's choice):** [ankan-ban/llama2.cu](https://github.com/ankan-ban/llama2.cu)
(MIT), at `cb0c6fa`.
- **What it is:** a CUDA port of Karpathy's llama2.c, Llama 2 inference.
- **CUDA code:** one file, `llama2.cu`, 772 lines.
- **Kernels:** 8, using fp16 (`half`) and CUB block and warp reductions. No cuBLAS.
- **Reference in the same repo:** llama2.c's fp32 CPU program, `run.c`, with the same command
  line and the same printout.

`museport scan` (no model): 1 CUDA file, 8 kernels, CUB (5 uses), shared memory (6), fp16 (1).

## The hidden check

- **The run:** greedy decoding (temperature 0) with Karpathy's `stories15M` model, a fixed
  number of steps and fixed prompts.
- **What is compared:** the generated story, byte for byte. The wrapper cuts it from the
  program's printout, between `<s>` and the final `achieved tok/s` timing line.
- **The reference:** llama2.cu's own CUDA program, recorded on an NVIDIA GPU (J1's
  `record_native`, run twice to show it repeats).
  - llama2.c's fp32 `run.c` is compared with that recording.
  - If they agree, the check is not sensitive to fp16 versus fp32.
  - If they disagree, the recording stays the reference, and the difference is reported.
- **Proof before any model runs:** `run.c` runs as a candidate, and a planted bug in it must
  fail.

## Steps and cost

| Step | Muse | GPU |
|---|---|---|
| 1. `museport plan` on the repo | ~$0.3 | |
| 2. Record llama2.cu on NVIDIA; prove the check on the CPU | | ~$0.02 |
| 3. Muse ports `llama2.cu` to one SYCL program, `llama2.cpp`: zero dpct helpers, the check passed on the CPU device | $1-3 per run, up to 2 runs | |
| 4. The port on NVIDIA (vs the recording and vs llama2.cu live) and on AMD (vs the recording) | | ~$0.2 |
| 5. Reassemble `llama2-sycl/`: the repo with `llama2.cu` replaced by the port, a Makefile for `icpx -fsycl`, a fresh-clone check | | |
| **Total** | **about $1.5-6.5, cap $8** (the author's) | **about $0.25** |

## Risks

- **fp16:** the port must use `sycl::half` where the CUDA uses `half`, or match the CUDA output
  some other way. Greedy decoding can flip a token where two logits nearly tie. The recorded
  repeat run shows whether the CUDA program itself is stable.
- **CUB:** `cub::BlockReduce` and `cub::WarpReduce` map to SYCL group algorithms
  (`reduce_over_group`).
- **One file is the whole program:** there is no per-kernel test, so the port is graded only
  end to end, as in GPT-2 stage 3.
