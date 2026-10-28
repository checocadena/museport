# Port report: llmc_matmul_backward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 7 / 16
- **Tokens:** in 39128, out 8643, cache write 0, cache read 57397; est. cost $0.0943
- **Best passing candidate:** iteration 7, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 19 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x1, DPCT1024x2, DPCT1049x1, DPCT1051x2, DPCT106... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | matmul_backward_harness.dp.cpp: lines 1-300 of 308 (12919 chars) |
| 4 | write_file | ok | wrote matmul_backward_sycl.cpp (8698 chars) |
| 5 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 6 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/mbruns/t3-matmul-backward/20261028T18510... |
| 7 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001) |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 324.76 ms, p95 357.81 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5fd88f7cffe671b7`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 19 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x1, DPCT1024x2, DPCT1049x1, DPCT1051x2, DPCT1065x2, DPCT1083x1, DPCT1093x2, DPCT1098x3, DPCT1118x2)
- **compile** (it 6, sources `03c7a773ccc2f4a5`): `ok` — compiled 1 source(s) -> work/build/candidate (4 warning(s))
- **scan_helpers** (it 7, sources `03c7a773ccc2f4a5`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `03c7a773ccc2f4a5`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001)
- **benchmark** (it 7, sources `03c7a773ccc2f4a5`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 324.76 ms, p95 357.81 ms

Current sources hash: `03c7a773ccc2f4a5`. Success requires compile, verify and zero-helper scan all on this hash.
