# Tier 1 again, with the loop fixes (Oct 27)

The same 10 Tier 1 kernel files as the [first runs](../port-llmc-tier1/README.md), with the
same tasks, hidden references, model (`muse-spark-1.3`) and 12-turn limit. What changed is the
loop around Muse (museport [#5](https://github.com/checocadena/museport/pull/5), toolkit
[#21](https://github.com/checocadena/oneapi-agent-toolkit/pull/21)):

- **New tools:**
  - `delete_file`. Before, Muse cleared SYCLomatic's output by overwriting each file with a stub,
    one turn per file.
  - `read_file` with a line range, so long SYCLomatic output can be read in parts.
- **Limits:**
  - 32,000 output tokens per turn, up from 16,000.
  - The toolkit's per-run token limit is raised, so only the USD cap ends a run. The cap was
    $0.70 per item, against $1.00 in the first runs.
  - A connection lost mid-answer no longer ends the run.
- **Goal:** it names the two new tools.

The command and its output are in [`run.log`](run.log).

## Result: 9 of 10 met the loop's success rule, up from 4; $1.0064, down from $1.6698

| Item | First runs | This run |
|---|---|---|
| `t1-crossentropy-forward` | max_iterations, 12 turns, $0.1315 | **success**, 9 turns, $0.0904 |
| `t1-crossentropy-softmax-backward` | success, 12 turns, $0.1916 | success, 6 turns, $0.0654 |
| `t1-permute` | success, 10 turns, $0.1188 | success, 7 turns, $0.0526 |
| `t1-residual-forward` | max_iterations, 12 turns, $0.1747 | **success**, 9 turns, $0.0887 |
| `t1-gelu-forward` | success, 9 turns, $0.1559 | success, 7 turns, $0.0899 |
| `t1-gelu-backward` | max_iterations, 12 turns, $0.1293 | **success**, 11 turns, $0.1502 |
| `t1-encoder-forward` | max_iterations, 12 turns, $0.1710 | **success**, 7 turns, $0.0842 |
| `t1-encoder-backward` | success, 12 turns, $0.1210 | success, 7 turns, $0.0782 |
| `t1-adamw` | max_iterations, 12 turns, $0.1266 | **success**, 7 turns, $0.0832 |
| `t1-fused-residual-forward` | max_iterations, 12 turns, $0.3494, **53 helpers left** | max_iterations, 12 turns, $0.2235, **0 helpers** |

**Totals:**

| | First runs | This run |
|---|---|---|
| Loop success | 4 / 10 | 9 / 10 |
| Correct with 0 helpers (hidden grade) | 9 / 10 | 10 / 10 |
| Turns per item, mean | 11.5 | 8.2 |
| Muse cost | $1.6698 | $1.0064 |

Costs are rounded to four decimals from the two `summary.json` files.

- **Correctness:** every final port passes the hidden reference with no dpct helper left in the
  tree. No variant is duplicated: the variant check passes on all 10.
- **The new tools:**
  - Muse called `delete_file` once in every run, on `migrated/`.
  - 9 of the 10 runs used `read_file` with a line range.
  - No turn came back empty.
- **`t1-fused-residual-forward`:** its final port is now correct and helper-free by the hidden
  grade. The run did not meet the loop's success rule within 12 turns.

## Limits

- **One sample per item.** These are single runs, so the differences are not a measured
  success rate.
- **Several changes at once.** The run changed several things together, as listed above, so
  it does not measure each fix on its own.
- **First runs unchanged.** The first runs stay on record. The README's tier table still
  counts them.
