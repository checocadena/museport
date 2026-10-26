# Port report: llmc_gelu_backward

- **Result:** `max_iterations` — reached 12 iterations without meeting success conditions
- **Model:** `muse-spark-1.3`
- **Iterations:** 12 / 12
- **Tokens:** in 59710, out 6825, cache write 0, cache read 170873; est. cost $0.1293
- **Best passing candidate:** iteration 8, 65 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | list_files | ok | 3 file(s) |
| 3 | read_file | ok | gelu_backward_harness.dp.cpp: 8431 chars |
| 4 | read_file | ok | common.h: 16877 chars |
| 5 | write_file | ok | wrote gelu_backward.cpp (6355 chars) |
| 6 | compile | ok | compiled 1 source(s) -> work/build/candidate (0 warning(s)) |
| 7 | scan_helpers | fail | 65 dpct helper use(s) remain in 2 file(s) (macro=31, device_queue_management=24, error_handling=3, events_timing=3) |
| 8 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| 9 | write_file | ok | wrote migrated/gelu_backward_harness.dp.cpp (79 chars) |
| 10 | list_files | ok | 8 file(s) |
| 11 | read_file | ok | MainSourceFiles.yaml: 49391 chars |
| 12 | write_file | ok | wrote migrated/common.h (79 chars) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `049675f832bdb75c`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 6, sources `2dc6e84490e08eb8`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 7, sources `2dc6e84490e08eb8`): `fail` — 65 dpct helper use(s) remain in 2 file(s) (macro=31, device_queue_management=24, error_handling=3, events_timing=3)
- **verify** (it 8, sources `2dc6e84490e08eb8`): `ok` — PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05)
- **benchmark:** not run

Current sources hash: `9e0d306b3bd3eab8`. Success requires compile, verify and zero-helper scan all on this hash.
