# Muse ports llm.c's Tier 1 CUDA kernels to SYCL; the ports run on NVIDIA GPUs

`museport port` ran the toolkit's agent loop, driven by **`muse-spark-1.3`**, on each of the
10 Tier 1 items in Muse's own [plan](../plan-llmc-devcuda-stream/README.md) of
[karpathy/llm.c](https://github.com/karpathy/llm.c) `dev/cuda` (`f1e2ace`). Every port was
checked against a reference the model never saw, then the same files, unchanged, were built
and checked on rented NVIDIA GPUs against the original llm.c CUDA kernels.

## Results

| Item | Variants | Loop result (turns) | Muse cost | Hidden grade (final sources, CPU) | NVIDIA GPU: vs reference / vs original CUDA |
|---|---|---|---|---|---|
| `t1-crossentropy-forward` | 1 | max_iterations (12) | $0.1315 | PASS, 5.96e-08, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_crossentropy_forward/SUMMARY.md) on RTX 2000 Ada Generation: 1.19e-07 / 0 |
| `t1-crossentropy-softmax-backward` | 1 | success (12) | $0.1916 | PASS, 0, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_crossentropy_softmax_backward/SUMMARY.md) on RTX 2000 Ada Generation: 0 / 0 |
| `t1-permute` | 1 | success (10) | $0.1188 | PASS, 0, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_permute/SUMMARY.md) on RTX 2000 Ada Generation: 0 / 0 |
| `t1-residual-forward` | 2 | max_iterations (12) | $0.1747 | PASS, 0, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-a4500/llmc_residual_forward/SUMMARY.md) on RTX A4500: 0 / 0 |
| `t1-gelu-forward` | 2 | success (9) | $0.1559 | PASS, 1.19e-07, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-a4500/llmc_gelu_forward/SUMMARY.md) on RTX A4500: 1.19e-07 / 1.19e-07 |
| `t1-gelu-backward` | 2 | max_iterations (12) | $0.1293 | PASS, 2.38e-07, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_gelu_backward/SUMMARY.md) on RTX 2000 Ada Generation: 2.38e-07 / 1.79e-07 |
| `t1-encoder-forward` | 3 | max_iterations (12) | $0.1710 | PASS, 0, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_encoder_forward/SUMMARY.md) on RTX 2000 Ada Generation: 0 / 0 |
| `t1-encoder-backward` | 2 | success (12) | $0.1210 | PASS, 4.77e-07, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_encoder_backward/SUMMARY.md) on RTX 2000 Ada Generation: 0 / 0 |
| `t1-adamw` | 2 | max_iterations (12) | $0.1266 | PASS, 1.79e-07, 0 helpers | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_adamw/SUMMARY.md) on RTX 2000 Ada Generation: 5.96e-08 / 5.96e-08 |
| `t1-fused-residual-forward` | 6 | max_iterations (12) | $0.3494 | verify PASS, **53 helpers left** | not run (helpers left) |

- **Muse spend:** $1.6698 for the 10 items (12 turns max each, $1 cap per item).
  NVIDIA GPU time: 9 runs on Runpod, about $0.034 in total, every pod's deletion confirmed.
- **Loop result** is the toolkit's strict rule: compile, verify and a zero-helper scan of the
  whole work tree, all on the same final sources, before the turns run out. 4/10 met it.
  The other 6 used all 12 turns. Most had a correct port and spent their last turns emptying
  SYCLomatic's leftover files, or never called verify on their final version.
- **Hidden grade** is the toolkit's `evaluate`, run after the loop on the final sources:
  compile, verify against the hidden reference, whole-tree helper scan. The model never sees
  it. **9/10 final ports are correct with 0 helpers.** `fused_residual_forward` (6 variants) is
  correct, but 53 helpers remain in leftover c2s files, so it does not count and was not
  sent to a GPU.
- **GPU:** all 9 pass on NVIDIA, against the reference and against llm.c's original CUDA
  built with `nvcc` from the same harness on the same GPU.
- Every final port submits its kernels to the SYCL device (`parallel_for`); none calls a CPU
  reference or keeps a dpct helper.
- No port copied one implementation into several variants: every port has at least as
  many distinct kernel bodies as variants (`variant_check`, added with Tier 2; see SUMMARY.md).

## How a task is built (museport, `src/museport/port.py` and `llmc.py`)

- **What Muse ports:** `<file>_harness.cu`, the original llm.c file with three mechanical
  changes. It is an fp32 build (`#define ENABLE_BF16` removed). The CPU reference functions
  are removed. llm.c's `main` is replaced by a contract `main` that reads an input file, runs
  **every kernel variant** through the file's own dispatcher, and writes each variant's
  output. `common.h` loses only its cuBLAS setup, which no Tier 1 kernel uses.
- **Hidden reference:** the file's own CPU functions, extracted verbatim, plus a generated
  `main`. Built with `g++`.
- **Tolerance:** llm.c's own fp32 rule, `|gpu - cpu| <= tol + |cpu| * FLT_EPSILON`
  (`atol` = the file's tol, `rtol` = FLT_EPSILON). Not widened.
- **Each check was proved before any model ran:** SYCLomatic's raw migration of every
  harness passes its reference (`museport port --tasks-only --check`, 10/10). A bug planted
  in one variant of gelu_forward fails 785,685 of 1,572,864 elements.

## Limits

- **fp32 only.** llm.c compiles six of these files in bf16; the bf16 path was not ported.
- **Reduced sizes** for a CPU device, given per task in `task.json` (`museport.note`).
  For example, B=4, T=256 instead of B=8, T=1024; V=1024 instead of 50257 for the encoder.
- **One run per item, one model.** Tier 2 and 3 items (sub-groups, cuBLAS) are not done.
- **AMD: not run.** DigitalOcean's AMD machines are reached at a bare IP, which this cloud
  session's network proxy did not get through. The droplets never answered and the toolkit
  deleted both (confirmed; no SSH key left). That cost about $1.0. See
  [`gpu/logs/t1-gelu-forward-amd.log`](gpu/logs/t1-gelu-forward-amd.log).
- Benchmarks in the GPU summaries are rough single-process wall times, not a speed claim.

## Files

- [`SUMMARY.md`](SUMMARY.md) and [`summary.json`](summary.json): loop results and hidden grades.
- `<item>/task/`: harness, hidden reference, `task.json`. `input.bin` is not committed;
  `make_task` regenerates it deterministically.
- `<item>/report.md`, `<item>/trace.jsonl`: every turn and tool result.
- `<item>/port/`: Muse's final sources. `<item>/best/`: the loop's best verified snapshot.
- `gpu/nvidia/<gpu>/<task>/`: the toolkit's GPU results; `gpu/logs/`: remote-test logs.
- `run-batch1.log`, `run-batch2.log`: the two batches (2 items, then 8).
