# Muse ports llm.c's Tier 2 CUDA kernels (warp shuffles, cooperative groups, shared memory)

Same pipeline as [Tier 1](../port-llmc-tier1/README.md): `museport port --tier 2` ran the
toolkit's agent loop, driven by **`muse-spark-1.3`**, on the 5 Tier 2 items of Muse's
[plan](../plan-llmc-devcuda-stream/README.md). The Tier 2 files use the CUDA-only constructs
the plan flagged as risks: warp shuffles, cooperative groups, shared memory, and grids sized
from the SM count. Each hidden check was [proven first](harness-checks/README.md).

## Results

| Item | Variants | Loop result (turns) | Muse cost | Hidden grade (final sources, CPU) | Distinct kernels / variants | NVIDIA GPU: vs reference / vs original CUDA |
|---|---|---|---|---|---|---|
| `t2-global-norm` | 4 | success (13) | $0.2598 | PASS, 7.15e-07, 0 helpers | 5 / 4 | [OK](gpu/nvidia/nvidia-rtx-a4500/llmc_global_norm/SUMMARY.md) on RTX A4500: 5.96e-07 / 1.55e-06 |
| `t2-layernorm-forward` | 6 | budget (13) | $0.4512 | PASS, 1.07e-06, 0 helpers | 8 / 6 | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_layernorm_forward/SUMMARY.md) on RTX 2000 Ada Generation: 9.54e-07 / 3.58e-07 |
| `t2-softmax-forward` | 8 | budget (14) | $0.5402 | verify PASS (5.77e-05), **45 helpers left** | 8 / 8 | not run (helpers left) |
| `t2-classifier-fused` | 5 | budget (14) | $0.4562 | PASS, 2.48e-05, 0 helpers | 5 / 5 | [OK](gpu/nvidia/nvidia-rtx-a4500/llmc_classifier_fused/SUMMARY.md) on RTX A4500: 1.34e-05 / 3.81e-06 |
| `t2-layernorm-backward` | 9 | success (11) | $0.4797 | PASS, 1.14e-05, 0 helpers | 2 / 9 **copied** | [OK](gpu/nvidia/nvidia-rtx-2000-ada-generation/llmc_layernorm_backward/SUMMARY.md) on RTX 2000 Ada Generation: 6.68e-06 / 1.72e-05 |

- **Muse spend:** $2.4454 (16 turns max each, $1 cap per item). That includes the crashed
  first run of `layernorm_backward` (see below). NVIDIA GPU time: 4 runs on Runpod, about
  $0.015, every pod's deletion confirmed.
- **Loop result** is the toolkit's strict rule; 2/5 met it. Three stopped on the loop's
  600k-token budget (`budget`), not on the dollar cap. These files are 270 to 1550 lines.
- **Hidden grade** of the final sources: **4/5 correct with 0 helpers.** `softmax_forward`
  is correct, but 45 helpers remain in leftover c2s files, so it does not count.
- **classifier_fused** is worth a note. SYCLomatic's own translation of it fails: it uses
  `native::exp` and gets variant 5 wrong
  ([details](harness-checks/README.md#why-two-items-needed-the-gpu-proof)). Muse's port
  passes every variant, on the CPU device and on the GPU.
- No port hard-codes a sub-group size of 32 or uses `native::exp`. Four use SYCL sub-group
  operations (8 to 12 uses each).

## Caveat: layernorm_backward copied one implementation into 8 of its 9 variants

The hidden check compares outputs, so it accepts any correct implementation. Muse's
`layernorm_backward` passes every check, but 8 of its 9 variants are the same naive kernel:
a work-item per row for `dinp`, then a work-item per column for `dweight`/`dbias`. llm.c's
variants are distinct algorithms (atomics, shared memory, warp reductions), and the port
keeps none of them; only variant 10 differs. So it is a correct port of the math, but not of
the kernel variants.

museport now measures this (`variant_check`: distinct kernel bodies against variants) and
shows it in every summary. Across all 15 Tier 1 and Tier 2 ports, this is the only one with
fewer distinct kernels than variants. That check cannot prove a variant keeps llm.c's
algorithm; it only shows whether variants were duplicated.

## The crash, and what changed (museport only)

`layernorm_backward`'s first run ended after 6 turns ($0.2583). A turn hit its output limit
in the middle of a `write_file` call. The stream reported no usage, so it was charged its
worst case. The call reached the toolkit without `path`, and the toolkit's `Toolbox` raises
`KeyError` for that; it only catches `OSError`, `ValueError` and `TypeError`. That ended the
batch before its summary was written.

Fixed in museport, with no toolkit change:
- `SafeToolbox` turns a missing argument into an error result the model sees.
- `run_items` survives any per-item exception and writes the summary after every item.

The summary of the first four items was rebuilt from their traces' `stop` events. The rerun
(`run-batch2.log`) succeeded. The crashed run's trace is in
`t2-layernorm-backward/run1-crashed/`.

## Limits

- fp32 only. llm.c builds `global_norm` and `layernorm_backward` in bf16; bf16-only
  variants (layernorm_backward's 4) are excluded.
- Reduced sizes, given per task in `task.json` (`museport.note`).
- One run per item (two for `layernorm_backward`); one model. AMD not run.
- Benchmarks in the GPU summaries are rough wall times, not a speed claim.

## Files

Same layout as Tier 1: `SUMMARY.md` and `summary.json`; per item `task/`, `report.md`,
`trace.jsonl`, `port/` and `best/`; `gpu/` for GPU results; `harness-checks/` for the proofs;
`run-batch1.log` and `run-batch2.log`.
