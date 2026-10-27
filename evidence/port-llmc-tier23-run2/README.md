# Tiers 2 and 3 again, with the loop fixes (Oct 27)

The same 9 Tier 2 and Tier 3 kernel files as the first runs ([Tier 2](../port-llmc-tier2/README.md),
[Tier 3](../port-llmc-tier3/README.md)). The tasks, hidden references, model
(`muse-spark-1.3`) and 16-turn limit are the same. What changed is the loop around Muse, as in
[Tier 1 run 2](../port-llmc-tier1-run2/README.md):

- `delete_file`, and `read_file` with a line range;
- 32,000 output tokens per turn, and no token limit below the USD cap ($0.70 per item here,
  against $1.00 in the first runs);
- a connection lost mid-answer no longer ends the run.

The goal is the default one, not the stricter `--strict-goal` of the earlier reruns. The command
and its output are in [`run.log`](run.log).

## Result: 6 of 9 met the loop's success rule, up from 4; 8 of 9 correct, as before; $2.7433

| Item | First run | This run |
|---|---|---|
| `t2-global-norm` | success, 13 turns | max_iterations, 16 turns; **correct, 0 helpers** |
| `t2-layernorm-forward` | token budget, 13 turns; correct | **success**, 7 turns |
| `t2-softmax-forward` | token budget, 14 turns; correct, **45 helpers** | max_iterations, 16 turns; **fails verify**, copies variants |
| `t2-classifier-fused` | token budget, 14 turns; correct | **success**, 7 turns |
| `t2-layernorm-backward` | success, 11 turns; **copies variants** | success, 14 turns; **still copies variants** |
| `t3-matmul-backward` | max_iterations, 16 turns; correct | **success**, 8 turns |
| `t3-trimat-forward` | success, 9 turns | max_iterations, 16 turns; **correct, 0 helpers** |
| `t3-matmul-backward-bias` | token budget, 14 turns; correct | **success**, 13 turns |
| `t3-matmul-forward` | success, 15 turns | success, 12 turns |

"Correct" is the hidden grade of the final sources: the reference matches and no dpct helper is
left anywhere in the tree. Per-item costs and the variant check are in [`SUMMARY.md`](SUMMARY.md).

**Totals:**

| | First runs | This run |
|---|---|---|
| Loop success | 4 / 9 | 6 / 9 |
| Correct with 0 helpers (hidden grade) | 8 / 9 | 8 / 9 |
| Turns per item, mean | 13.2 | 12.1 |
| Muse cost | $3.85 (incl. one crashed run) | $2.7433 |
| `write_file` calls that overwrote SYCLomatic's output | 18 | 0 (`delete_file` 10 times) |

## What went wrong, item by item

- **`global_norm` and `trimat_forward`:** both final ports are correct and helper-free. Each
  run ended one step short: at turn 16, Muse had recompiled after deleting `migrated/` and
  had not yet verified again on those sources. The loop's rule needs verify and the helper
  scan on the same compiled sources, so neither counts as a loop success.
- **`softmax_forward` (worse than the first run):** Muse read only the first 120 of
  SYCLomatic's 1,043 lines, then wrote its port in one turn. It compiled, but verify failed:
  36,435 of 25,731,584 values outside tolerance. Muse wrote "Early variants look good but a
  later one diverges", then spent turns 9-16 reading its own file and never wrote a fix. The
  variant check also finds one implementation copied across variants (6 distinct kernels for
  8 variants). The first run's port was correct but kept 45 helpers.
- **`layernorm_backward`:** correct and helper-free, a loop success, but it again copies one
  implementation across variants (8 distinct kernels for 9 variants). The stricter goal of
  the earlier rerun fixed this ([round 2](../port-llmc-reruns-2/README.md)); this run used the
  default goal.

## Limits

- **One sample per item.** The differences from the first runs are not measured success
  rates.
- **Several changes at once**, as for Tier 1 run 2.
- **GPUs:** the 5 correct ports that make no oneMath call pass on NVIDIA GPUs against both
  the reference and llm.c's own CUDA, for about $0.04 ([NVIDIA](gpu/NVIDIA.md)). Ports that
  call oneMath (`matmul_forward`, `matmul_backward`, `trimat_forward`) are verified on the CPU
  device only, because oneMath is not built for the GPU hosts yet (plan item D2). AMD was not
  run for these.
