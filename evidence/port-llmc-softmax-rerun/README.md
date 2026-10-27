# `softmax_forward` again, with the stricter goal (Oct 27)

In [Tiers 2-3 run 2](../port-llmc-tier23-run2/README.md), Muse's `softmax_forward` port failed
the hidden check and copied one implementation across its 8 kernel variants. This run retries
that one item with the fixed loop and `--strict-goal`. The stricter goal tells Muse to delete
`migrated/` early and to keep each CUDA variant's own algorithm. It also gets 20 turns in place
of 16, with the same $0.70 cap. It is reported beside the headline table, not in it, because the
other items ran with the default goal and 16 turns.

The command and its output are in [`run.log`](run.log).

## Result: still fails the hidden check; the variants are no longer copied

| | Tiers 2-3 run 2 | This run |
|---|---|---|
| Loop result | max_iterations, 16 turns | stopped by the USD cap after 19 turns |
| Hidden grade | **FAIL**, 36,435 of 25,731,584 values outside tolerance | **FAIL**, 41,906 of 25,731,584 values outside tolerance |
| Helpers left | 0 | 0 |
| Distinct kernels / variants | 6 / 8 (copied) | **8 / 8** |
| Cost | $0.5857 | $0.7470 |

The cap is checked between turns, so the last turn took the run $0.047 past its $0.70 cap.

## How the run went

1. **Turns 1-4:** `migrate`, then Muse wrote the whole port in one `write_file`. It worked
   from the CUDA harness in its first message; it did not read SYCLomatic's output.
2. **Turns 5-8:** it compiled, deleted `migrated/`, recompiled and ran verify. Verify failed;
   the helper scan found 0 helpers.
3. **Turns 9-15:** Muse wrote: "First two variants pass but the third is off by 16× — fixing
   the double-counted reduction." It reread its file in parts and rewrote it.
4. **Turns 16-17:** the rewrite compiled, but verify still failed, on 41,906 values.
5. **Turns 18-19:** Muse was rereading its code when the USD cap stopped the run.

The variant check confirms that the stricter goal worked on what it targets: 8 distinct
kernels for 8 variants. The numerical bug was not found in time.

## Spend

$0.7470 (reported usage). This brings the comparison runs to $16.5436 of the $30 cap.
