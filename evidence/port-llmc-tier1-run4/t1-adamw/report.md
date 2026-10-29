# Port report: llmc_adamw

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 6 / 12
- **Tokens:** in 43578, out 3498, cache write 0, cache read 33589; est. cost $0.0744
- **Best passing candidate:** iteration 6, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | read_file | ok | adamw_harness.dp.cpp: lines 1-282 of 282 (12663 chars) |
| 3 | write_file | ok | wrote adamw_sycl.cpp (5852 chars) |
| 4 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 5 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run4/t1-adamw/20261029T005453... |
| 6 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001) |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 6 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 354.98 ms, p95 375.24 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `6159d8e367cbbd12`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 5, sources `3dee3826b88403cd`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `3dee3826b88403cd`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 6, sources `3dee3826b88403cd`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001)
- **benchmark** (it 6, sources `3dee3826b88403cd`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 354.98 ms, p95 375.24 ms

Current sources hash: `3dee3826b88403cd`. Success requires compile, verify and zero-helper scan all on this hash.
