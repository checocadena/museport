# `softmax_forward` with more room: passes (Oct 27)

The third post-fix attempt at `softmax_forward`:

- In [Tiers 2-3 run 2](../port-llmc-tier23-run2/README.md) (default goal, 16 turns), the port
  failed the hidden check and copied variants.
- In [the first strict rerun](../port-llmc-softmax-rerun/README.md) (strict goal, 20 turns,
  $0.70), the variants were fixed but the check still failed when the USD cap stopped the run.

The author raised the per-run cap to $3.00. This run has the same strict goal, with 30 turns
and a $3.00 cap. It is reported beside the headline table, not in it, because the other items
ran with the default goal, 16 turns and $0.70.

The command and its output are in [`run.log`](run.log).

## Result: loop success in 23 turns, $0.9063; passes on NVIDIA

| | |
|---|---|
| Loop result | **success** at turn 23 of 30 |
| Hidden grade of the final sources | **PASS**, worst max abs err 2.86e-05 (atol 1e-4); 0 helpers |
| Variant check | 8 distinct kernels for 8 variants: none copied |
| Cost | **$0.9063** (reported usage), well under the $3.00 cap |
| NVIDIA | **PASS** on an L4 against the reference and llm.c's own CUDA (worst 5.96e-08 vs CUDA), about $0.01 ([`gpu/`](gpu/)) |

## How the run went

1. **Turns 1-6:** `migrate`, then Muse wrote the whole port in one `write_file`. The first
   compile failed with one error.
2. **Turns 7-10:** Muse switched variant 6 to the portable group reduction, rewrote the file,
   and it compiled.
3. **Turns 11-14:** Muse deleted `migrated/` and recompiled. The scan found 0 helpers, but
   verify failed on 41,906 values, the same failure as the earlier strict rerun.
4. **Turns 15-19:** Muse read its port in parts and wrote: "Early variants look clean but
   variant 4 diverges — hardening its reduction to match the CUDA algorithm."
5. **Turns 20-23:** it rewrote the reductions. The port compiled, the scan found 0 helpers,
   and verify passed.

In the earlier strict rerun, the first verify failed at turn 8 and the one fix attempt failed
at turn 17; the $0.70 cap stopped it at turn 19. Here Muse had the turns and budget to find
the bug in variant 4.

## Spend

$0.9063 for Muse and about $0.01 of GPU time. Comparison runs: $17.4499 of the $30 cap.
