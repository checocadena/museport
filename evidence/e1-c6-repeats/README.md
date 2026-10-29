# E1 and C6: three runs per item, with and without the loop's checks (Oct 29)

llm.c's 10 Tier 1 kernel files, ported three times in each arm. Each run had the same tasks,
hidden checks, model (`muse-spark-1.3`), SYCLomatic, 12-turn limit and $0.70 cap per item.

- **Feedback** (C6, and E1's feedback arm): `museport port`, the agent loop with `verify`,
  `scan_helpers` and `benchmark`. Run 2 is from Oct 27; runs 3 and 4 are new.
- **Baseline** (E1): `museport port --baseline`. Muse cannot run or test its port, and the run
  ends when it calls `finish`. The first run is from Oct 28; runs 2 and 3 are new.

After every run, the same hidden grade checks the final sources: **correct** means the port
passes the hidden check with 0 dpct helpers in the tree. The variant check flags a port with
fewer distinct kernel bodies than the file has variants (a **copied variant**). The hidden
check compares outputs only, so it cannot see a copied variant.

## Result

**Feedback: 29 of 30 correct** (per run:
10, 10, 9). **Baseline: 28 of
30** (per run: 10, 9, 9).

| Arm | Run | Correct, 0 helpers | Loop success | Copied variant | Mean turns | Cost |
|---|---|---|---|---|---|---|
| feedback | [port-llmc-tier1-run2](../port-llmc-tier1-run2) | 10 / 10 | 9 / 10 | none | 8.2 | $1.01 |
| feedback | [port-llmc-tier1-run3](../port-llmc-tier1-run3) | 10 / 10 | 10 / 10 | `fused-residual-forward` | 7.1 | $0.91 |
| feedback | [port-llmc-tier1-run4](../port-llmc-tier1-run4) | 9 / 10 | 9 / 10 | none | 8 | $1.06 |
| baseline | [port-llmc-tier1-baseline](../port-llmc-tier1-baseline) | 10 / 10 | n/a | `fused-residual-forward` | 9.1 | $1.18 |
| baseline | [port-llmc-tier1-baseline-2](../port-llmc-tier1-baseline-2) | 9 / 10 | n/a | none | 8.7 | $1.26 |
| baseline | [port-llmc-tier1-baseline-3](../port-llmc-tier1-baseline-3) | 9 / 10 | n/a | none | 9.9 | $1.13 |

| Item | Feedback: correct runs | Baseline: correct runs |
|---|---|---|
| `t1-crossentropy-forward` | 3 / 3 | 3 / 3 |
| `t1-crossentropy-softmax-backward` | 3 / 3 | 3 / 3 |
| `t1-permute` | 3 / 3 | 3 / 3 |
| `t1-residual-forward` | 3 / 3 | 3 / 3 |
| `t1-gelu-forward` | 3 / 3 | 3 / 3 |
| `t1-gelu-backward` | 3 / 3 | 3 / 3 |
| `t1-encoder-forward` | 3 / 3 | 3 / 3 |
| `t1-encoder-backward` | 3 / 3 | 3 / 3 |
| `t1-adamw` | 3 / 3 | 3 / 3 |
| `t1-fused-residual-forward` | 2 / 3 | 1 / 3 |

- **The one hard item is `fused_residual_forward`,** the only Tier 1 file with warp-level CUDA.
  Each miss is the same: the port computes the right values, but Muse used all 12 turns with 98
  dpct helpers still in the tree. That happened in 1 of 3 runs with feedback and 2 of 3
  without. Copied variants turned up once in each arm, both in this file.
- **The other 9 files are correct in every run of both arms.** On these simple kernels, the
  checks make no measurable difference to correctness.
- **Without feedback, runs are longer and cost more:** 9.2 turns per item on average against
  7.8, and $3.57 against $2.98 for the
  three runs.
- **Loop success** (the loop's stricter rule, met inside the run) is 28 of 30 with feedback.
  The baseline arm has no such rule.

**Limits:** three runs per item is a small sample, and these are the simplest kernels. The
GPT-2 training baseline (`evidence/gpt2-stage3-baseline`) is still one run per arm.

## Notes

- **An API error in baseline run 2:** `t1-gelu-backward`'s first attempt ended after 6 turns on
  an HTTP 502 from Meta's API ("upstream request failed"). It was rerun alone into the same
  folder (`museport port --items t1-gelu-backward --baseline`, same caps). The failed attempt
  stays in `summary.json` under `earlier_runs` and in the spend ($0.058), its trace is
  [`trace-api-error.jsonl`](../port-llmc-tier1-baseline-2/t1-gelu-backward/trace-api-error.jsonl),
  and the log records the rerun.
- **New spend:** $4.37 for the four new batches (runs 3 and 4, baseline runs 2 and 3), the
  rerun included.
- `repeats.json` holds every number above. `museport.repeats.collect` rebuilds it from the six
  batches' `summary.json`.
