# Port report: llmc_matmul_backward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 8 / 16
- **Tokens:** in 46399, out 10371, cache write 0, cache read 58421; est. cost $0.1108
- **Best passing candidate:** iteration 8, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 19 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x1, DPCT1024x2, DPCT1049x1, DPCT1051x2, DPCT106... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | matmul_backward_harness.dp.cpp: lines 1-120 of 308 (4878 chars) |
| 4 | read_file | ok | matmul_backward_harness.dp.cpp: lines 120-308 of 308 (8334 chars) |
| 5 | write_file | ok | wrote matmul_backward.cpp (8664 chars) |
| 6 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 7 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier23-run2/t3-matmul-backward/2026... |
| 8 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001) |
| 8 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 8 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 317.57 ms, p95 340.51 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5fd88f7cffe671b7`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 19 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x1, DPCT1024x2, DPCT1049x1, DPCT1051x2, DPCT1065x2, DPCT1083x1, DPCT1093x2, DPCT1098x3, DPCT1118x2)
- **compile** (it 7, sources `2e83a1c9f0ca9ea9`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 8, sources `2e83a1c9f0ca9ea9`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 8, sources `2e83a1c9f0ca9ea9`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001)
- **benchmark** (it 8, sources `2e83a1c9f0ca9ea9`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 317.57 ms, p95 340.51 ms

Current sources hash: `2e83a1c9f0ca9ea9`. Success requires compile, verify and zero-helper scan all on this hash.
