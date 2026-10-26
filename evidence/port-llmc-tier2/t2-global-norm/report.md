# Port report: llmc_global_norm

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 13 / 16
- **Tokens:** in 121949, out 16999, cache write 0, cache read 234361; est. cost $0.2598
- **Best passing candidate:** iteration 13, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 31 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x6, DPCT1024x2, DPCT1049x5, DPCT1051x2, DPCT106... |
| 2 | read_file | ok | global_norm_harness.dp.cpp: 18831 chars |
| 3 | list_files | ok | 3 file(s) |
| 4 | read_file | ok | common.h: 17278 chars |
| 5 | list_files | ok | 4 file(s) |
| 6 | write_file | ok | wrote global_norm.cpp (9057 chars) |
| 7 | compile | ok | compiled 1 source(s) -> work/build/candidate (0 warning(s)) |
| 8 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.01) |
| 8 | scan_helpers | fail | 70 dpct helper use(s) remain in 2 file(s) (macro=30, device_queue_management=27, error_handling=3, events_timing=3) |
| 9 | write_file | ok | wrote migrated/global_norm_harness.dp.cpp (77 chars) |
| 10 | read_file | ok | MainSourceFiles.yaml: 77773 chars |
| 11 | write_file | ok | wrote migrated/common.h (77 chars) |
| 12 | compile | ok | compiled 1 source(s) -> work/build/candidate (0 warning(s)) |
| 13 | scan_helpers | ok | no dpct helpers in 3 file(s) |
| 13 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 0.01) |
| 13 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 418.03 ms, p95 438.32 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `823b7ee2cca81471`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 31 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x6, DPCT1024x2, DPCT1049x5, DPCT1051x2, DPCT1065x3, DPCT1093x2, DPCT1096x1, DPCT1098x3, DPCT1101x3, DPCT1118x1)
- **compile** (it 12, sources `445072c10b01477f`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 13, sources `445072c10b01477f`): `ok` — no dpct helpers in 3 file(s)
- **verify** (it 13, sources `445072c10b01477f`): `ok` — PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 0.01)
- **benchmark** (it 13, sources `445072c10b01477f`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 418.03 ms, p95 438.32 ms

Current sources hash: `445072c10b01477f`. Success requires compile, verify and zero-helper scan all on this hash.
