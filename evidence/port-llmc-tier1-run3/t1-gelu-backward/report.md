# Port report: llmc_gelu_backward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 6 / 12
- **Tokens:** in 44745, out 4111, cache write 0, cache read 29779; est. cost $0.0779
- **Best passing candidate:** iteration 6, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | read_file | ok | gelu_backward_harness.dp.cpp: lines 1-218 of 218 (8431 chars) |
| 3 | read_file | ok | common.h: lines 1-200 of 359 (7436 chars) |
| 4 | write_file | ok | wrote gelu_backward.cpp (6283 chars) |
| 5 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 5 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run3/t1-gelu-backward/2026102... |
| 6 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 6 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 268.33 ms, p95 289.86 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `049675f832bdb75c`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 5, sources `c4843ff7ed131e04`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `c4843ff7ed131e04`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 6, sources `c4843ff7ed131e04`): `ok` — PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05)
- **benchmark** (it 6, sources `c4843ff7ed131e04`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 268.33 ms, p95 289.86 ms

Current sources hash: `c4843ff7ed131e04`. Success requires compile, verify and zero-helper scan all on this hash.
