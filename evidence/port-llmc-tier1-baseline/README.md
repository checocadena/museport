# Tier 1 without feedback (plan item E1, Oct 28)

How much do the loop's checks help Muse? This run takes away its feedback. The setup is the
same as [Tier 1 run 2](../port-llmc-tier1-run2/README.md): the same 10 llm.c kernel files,
tasks, hidden references, model (`muse-spark-1.3`), SYCLomatic, 12-turn limit and $0.70 cap per
item.

The one change is `museport port --baseline`, the toolkit's ablation arm:

- Muse has no `verify`, `scan_helpers` or `benchmark` tool, so it can't run or test its port.
- The loop does not check success; the run ends when Muse calls `finish`.
- A note in the goal says so, since the goal still names `verify`.
- After the run, the same hidden grade checks the final sources, unchanged.

The command and its output are in [`run.log`](run.log).

## Result: 10 of 10 correct with 0 helpers; $1.1803

| Item | With feedback (run 2) | Without feedback |
|---|---|---|
| `t1-crossentropy-forward` | success, 9 turns, $0.0904 | finished, 6 turns, $0.0673 |
| `t1-crossentropy-softmax-backward` | success, 6 turns, $0.0654 | finished, 9 turns, $0.0968 |
| `t1-permute` | success, 7 turns, $0.0526 | finished, 8 turns, $0.0774 |
| `t1-residual-forward` | success, 9 turns, $0.0887 | finished, 7 turns, $0.0801 |
| `t1-gelu-forward` | success, 7 turns, $0.0899 | finished, 8 turns, $0.0859 |
| `t1-gelu-backward` | success, 11 turns, $0.1502 | max_iterations, 12 turns, $0.1488 |
| `t1-encoder-forward` | success, 7 turns, $0.0842 | max_iterations, 12 turns, $0.1667 |
| `t1-encoder-backward` | success, 7 turns, $0.0782 | finished, 9 turns, $0.0890 |
| `t1-adamw` | success, 7 turns, $0.0832 | finished, 10 turns, $0.1205 |
| `t1-fused-residual-forward` | max_iterations, 12 turns, $0.2235 | finished, 10 turns, $0.2478 |

"Finished": Muse called `finish`. Every final port, in both runs, passes the hidden grade with
no dpct helper in the tree.

**Totals:**

| | With feedback | Without |
|---|---|---|
| Correct with 0 helpers (hidden grade) | 10 / 10 | **10 / 10** |
| Copied variants (variant check) | 0 | 1 (`fused_residual_forward`: 5 distinct kernels for 6 variants) |
| Turns per item, mean | 8.2 | 9.1 |
| Muse cost | $1.0064 | $1.1803 |

## What this shows

- **On these 10 simple kernels, feedback did not change correctness.** Muse wrote correct ports
  without ever running them.
- **The checks still caught one thing.** Without feedback, one port copies an implementation
  across two variants. With feedback, none did.
- **One run per item.** This is not a success rate.

## A correction to the variant check

The run's [`summary.json`](summary.json) also flags `t1-crossentropy-forward` as copied, with 0
kernels found. That flag is wrong:

- The port launches its kernel through a functor (`q.parallel_for(launch,
  CrossEntropyForwardKernel1{...})`), and the variant check read only lambda bodies.
- The check now reads functor kernels too, with a test.
- Rechecked with the fix, it has 1 kernel for 1 variant, and the result above uses that count.
- Rechecking every earlier batch with the fix changed nothing else.
