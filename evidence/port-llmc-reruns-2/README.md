# Reruns, second round (plan C2-C3), Oct 27

The [first round of reruns](../port-llmc-reruns/README.md) used `--strict-goal`, and two of its
three runs ended on an empty answer. This round reruns those two items. It uses the same goal,
with three changes:

- the loop marks an empty answer instead of failing on it (`stream_transport(mark_empty=True)`);
- the output budget is 32,000 tokens per turn instead of 16,000 (`--max-output 32000`);
- for the last run only, the toolkit's per-run token limit is 2,000,000 instead of 600,000
  (`--max-total-tokens 2000000`), so that only the USD cap ends the run.

**Setup:**
- Model `muse-spark-1.3`, up to 14 turns.
- Caps: $0.70 per item and $1.40 for the round. The layernorm retry was capped at $0.66, the rest
  of the round's cap.
- Commands and output are in [`run.log`](run.log).

## Result ($1.4502)

| Item | First run (tier dirs) | This round |
|---|---|---|
| `t2-layernorm-backward` | correct, 0 helpers, but 2 distinct kernel bodies for 9 variants | **correct, 0 helpers, 10 distinct kernel bodies for 9 variants**, $0.7144 (plus a $0.1967 attempt that lost its connection, below) |
| `t1-fused-residual-forward` | correct, 53 helpers left | correct, 45 helpers left, $0.5391 |

### `t2-layernorm-backward`

- **Final port:** passes the hidden grade, with worst max abs error 1.53e-05 at atol 1e-3, and no
  dpct helper is left anywhere in the tree.
- **Variant check:** it no longer copies one implementation into several variants.
- **How the run went:**
  - Muse wrote the whole port in one `write_file` (50,505 characters, 24,278 output tokens). That
    one write would not have fit in the old 16,000-token budget.
  - It compiled and passed verify on the first try.
  - Muse then cleared `migrated/` and verified again.
- **Why it is not a loop success:** the run hit its $0.66 cost cap after turn 14, before it ran
  `scan_helpers` on the final tree. The cap is checked between turns, so the run ended $0.0544
  above it. The hidden grade ran on the final sources afterwards and found 0 helpers.
- **Earlier attempt:** the first try ended at turn 7 with `ConnectionResetError` while the answer
  was streaming. It was not retried automatically, because the request may have been billed.
  Its trace is in `t2-layernorm-backward/run1-crashed/`, and its $0.1967 is counted in the spend.

### `t1-fused-residual-forward`

- **How the run went:** Muse wrote a working port at turn 8 and passed verify at turn 10.
- **Why it ended early:** the toolkit's 600,000-token per-run limit stopped the run at turn 12,
  while Muse was clearing `migrated/`. One of the two SYCLomatic files was still there, so
  45 helpers remain in the final tree.

## Spend

The round spent $1.4502 against the $1.40 approved. The cap is checked between turns, so the
last run's final turn went over it by $0.0544.
