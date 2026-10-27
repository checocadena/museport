# Tier 1 without SYCLomatic (Oct 27)

How much does SYCLomatic help Muse? This run takes it away. The setup is the same as
[Tier 1 run 2](../port-llmc-tier1-run2/README.md): the same 10 llm.c kernel files, tasks,
hidden references, model (`muse-spark-1.3`), 12-turn limit and $0.70 cap per item.

The one change is `museport port --no-migrate`:

- the `migrate` tool (SYCLomatic, `c2s`) is not offered, and the toolbox refuses it;
- the system prompt's procedure line starts with "read the CUDA source -> write the SYCL port
  yourself" in place of "migrate -> compile";
- the first message no longer says "Start with migrate";
- the goal says that migrate is not available.

The rest is unchanged: the other tools, `verify`, `scan_helpers`, the success rule and the
hidden grade. The toolkit's first message already holds the whole CUDA harness, as in every
run.

The command and its output are in [`run.log`](run.log).

## Result: 10 of 10 met the loop's success rule; $0.4759

| Item | Tier 1 run 2 (with SYCLomatic) | This run (without) |
|---|---|---|
| `t1-crossentropy-forward` | success, 9 turns, $0.0904 | success, 4 turns, $0.0191 |
| `t1-crossentropy-softmax-backward` | success, 6 turns, $0.0654 | success, 4 turns, $0.0234 |
| `t1-permute` | success, 7 turns, $0.0526 | success, 5 turns, $0.0207 |
| `t1-residual-forward` | success, 9 turns, $0.0887 | success, 5 turns, $0.0284 |
| `t1-gelu-forward` | success, 7 turns, $0.0899 | success, 6 turns, $0.0498 |
| `t1-gelu-backward` | success, 11 turns, $0.1502 | success, 4 turns, $0.0382 |
| `t1-encoder-forward` | success, 7 turns, $0.0842 | success, 6 turns, $0.0654 |
| `t1-encoder-backward` | success, 7 turns, $0.0782 | success, 8 turns, $0.0689 |
| `t1-adamw` | success, 7 turns, $0.0832 | success, 7 turns, $0.0732 |
| `t1-fused-residual-forward` | max_iterations, 12 turns, $0.2235 | **success**, 4 turns, $0.0888 |

**Totals:**

| | With SYCLomatic (run 2) | Without |
|---|---|---|
| Loop success | 9 / 10 | 10 / 10 |
| Correct with 0 helpers (hidden grade) | 10 / 10 | 10 / 10 |
| Turns per item, mean | 8.2 | 5.3 |
| Muse cost | $1.0064 | $0.4759 |
| Tool calls: `migrate` / `read_file` / `delete_file` | 10 / 22 / 10 | 0 / 9 / 0 |

The tool counts are from the `tool_call` events in each item's `trace.jsonl`.

- **Correctness:** every final port passes the hidden reference with no dpct helper anywhere
  in the tree. The variant check passes on all 10: no variant is a copy of another.
- **How the runs went:** Muse listed the files, often read part of the harness, wrote the whole
  port in one `write_file`, compiled, and then ran `verify`, `scan_helpers` and `benchmark`.
  8 of the 10 ports compiled on their first compile; `t1-adamw` and `t1-encoder-backward`
  needed a second write and compile.

## What this does and does not show

- **On these 10 files, SYCLomatic did not help.** Muse ported them directly from the CUDA
  source in fewer turns and for half the cost. With SYCLomatic, Muse spent turns running it,
  reading its long output (22 `read_file` calls) and deleting it.
- **Tier 1 is the simple tier.** Nine of the ten CUDA files are one-thread-per-element kernels
  with no shared memory and no warp operations.
- **The one file with warp-level code got simpler kernels.** `fused_residual_forward.cu` uses
  shared memory, warp shuffles and `__syncthreads` in its variants. Without SYCLomatic, Muse
  wrote all 6 variants as plain `parallel_for` over rows: six distinct, correct kernels, but
  with no work-group, sub-group or local memory structure. With SYCLomatic (run 2), its port
  kept local memory and barriers. The default goal does not demand each variant's algorithm
  (`--strict-goal` does), and performance was not compared.
- **One sample per item.** These are single runs, not a measured success rate.

## Spend, including a broken first attempt

The first attempt ran in a shell where I had not loaded the oneAPI environment, so `icpx` was
missing. The first four items wrote a port, found no compiler and stopped as `blocked`. I
stopped the batch during the fifth item. Its log is [`aborted-no-icpx.log`](aborted-no-icpx.log).

| | Cost |
|---|---|
| Aborted attempt, reported usage of 4 items | $0.1061 |
| Aborted attempt, the request in flight when stopped | not reported; at most $0.15 |
| This run | $0.4759 |
| **Total** | **$0.5820**, plus at most $0.15 |
