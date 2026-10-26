# `museport plan` on llm.c `dev/cuda`, streamed: grounded plan

- **When:** 2026-10-26
- **Target:** [karpathy/llm.c](https://github.com/karpathy/llm.c) at `f1e2ace`, `dev/cuda` (22 CUDA files)
- **Command:** `ONEAPI_AGENT_AUTH_FROM_NETWORK=1 museport plan <llm.c>/dev/cuda --max-output 48000 --out evidence/plan-llmc-devcuda-stream`
- **Model:** `muse-spark-1.3`, provider `meta`, default reasoning effort, streamed
- **Worst-case cost before sending:** $0.356 (estimate)

## Result

| | |
|---|---|
| Plan items | 23 (1 shared header, 10 Tier 1, 5 Tier 2, 7 Tier 3) |
| Grounded | **yes**: every file and kernel it names exists |
| Coverage | 22/22 CUDA files placed |
| Grounding issues | **0** |
| Tokens | 135,343 in (0 cached), 22,499 out (mostly reasoning) |
| Cost | **$0.2648** (from the reported usage and Meta's list prices) |
| Wall time | 1 min 37 s |

Files: [`plan.md`](plan.md) (readable), [`plan.json`](plan.json) (plan, usage, grounding),
[`scan.json`](scan.json) (the ground truth it was checked against), [`run.log`](run.log).

## What grounding does and does not show

`ground` checks names, coverage, tiers, library mappings, CUDA-only constructs listed as risks,
and dependency order. It does **not** judge whether an approach is right. Spot check of two
items: `t1-gelu-forward` and `t2-softmax-forward` name the right SYCL constructs (sub-groups,
`local_accessor`, `shift_group_left`) and reuse llm.c's own `validate_result` checks. One
detail is questionable: `joint_reduce` is suggested where `reduce_over_group` is the
sub-group reduction. No item has been ported yet.

## Why a second run

The [first run](../plan-llmc-devcuda/README.md) failed with HTTP 502. Measured since then: a
non-streamed request whose answer takes more than about 30 s gets
`502 upstream request failed` at 30.2 s, while the same request streamed completed in 51 s.
This run took 97 s and only completed because it was streamed. Also, the token estimate
(4 characters per token) undercounted the input by about 12% (121k estimated, 135k actual).
