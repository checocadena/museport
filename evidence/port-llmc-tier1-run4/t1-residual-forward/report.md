# Port report: llmc_residual_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 7 / 12
- **Tokens:** in 28931, out 3195, cache write 0, cache read 50741; est. cost $0.0574
- **Best passing candidate:** iteration 7, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | read_file | ok | residual_forward_harness.dp.cpp: lines 1-196 of 196 (7415 chars) |
| 3 | write_file | ok | wrote residual_forward.cpp (5129 chars) |
| 4 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run4/t1-residual-forward/2026... |
| 5 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 6 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run4/t1-residual-forward/2026... |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| 7 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 281.66 ms, p95 295.57 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `6b9f943dc408d183`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 6, sources `ba9f08b5e35232bd`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `ba9f08b5e35232bd`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `ba9f08b5e35232bd`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **benchmark** (it 7, sources `ba9f08b5e35232bd`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 281.66 ms, p95 295.57 ms

Current sources hash: `ba9f08b5e35232bd`. Success requires compile, verify and zero-helper scan all on this hash.
