# Port report: llmc_gelu_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 8 / 12
- **Tokens:** in 53258, out 3644, cache write 0, cache read 49732; est. cost $0.0895
- **Best passing candidate:** iteration 8, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | read_file | ok | gelu_forward_harness.dp.cpp: lines 1-199 of 199 (7343 chars) |
| 3 | read_file | ok | common.h: lines 1-250 of 359 (8760 chars) |
| 4 | write_file | ok | wrote gelu_forward.cpp (3499 chars) |
| 5 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run3/t1-gelu-forward/20261029... |
| 6 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 7 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run3/t1-gelu-forward/20261029... |
| 8 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) |
| 8 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 8 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 296.24 ms, p95 340.17 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `01cba48287b938df`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 7, sources `88c47f1146a76fe5`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 8, sources `88c47f1146a76fe5`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 8, sources `88c47f1146a76fe5`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05)
- **benchmark** (it 8, sources `88c47f1146a76fe5`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 296.24 ms, p95 340.17 ms

Current sources hash: `88c47f1146a76fe5`. Success requires compile, verify and zero-helper scan all on this hash.
