# Port report: llmc_classifier_fused

- **Result:** `budget` — token budget exceeded: 659530 > 600000
- **Model:** `muse-spark-1.3`
- **Iterations:** 14 / 16
- **Tokens:** in 229673, out 25512, cache write 0, cache read 404345; est. cost $0.4562
- **Best passing candidate:** iteration 9, 95 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 38 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x6, DPCT1024x2, DPCT1049x5, DPCT1051x2, DPCT106... |
| 2 | list_files | ok | 3 file(s) |
| 3 | read_file | ok | classifier_fused_harness.dp.cpp: 47901 chars |
| 4 | read_file | ok | classifier_fused_harness.dp.cpp: 47901 chars |
| 5 | list_files | ok | 4 file(s) |
| 6 | read_file | ok | common.h: 17277 chars |
| 7 | write_file | ok | wrote classifier_fused_sycl.cpp (24141 chars) |
| 8 | compile | ok | compiled 1 source(s) -> work/build/candidate (8 warning(s)) |
| 9 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 2.48e-05, atol 0.0001) |
| 10 | scan_helpers | fail | 95 dpct helper use(s) remain in 2 file(s) (macro=43, device_queue_management=37, memory_management=3, error_handling=3) |
| 11 | read_file | ok | classifier_fused_sycl.cpp: 24141 chars |
| 12 | write_file | ok | wrote migrated/classifier_fused_harness.dp.cpp (50 chars) |
| 13 | read_file | ok | MainSourceFiles.yaml: 127384 chars |
| 14 | write_file | ok | wrote migrated/common.h (50 chars) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `1e1ae9bb65e61360`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 38 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x6, DPCT1024x2, DPCT1049x5, DPCT1051x2, DPCT1065x8, DPCT1093x2, DPCT1096x2, DPCT1098x6, DPCT1101x1, DPCT1118x1)
- **compile** (it 8, sources `4328315ba9d33daa`): `ok` — compiled 1 source(s) -> work/build/candidate (8 warning(s))
- **scan_helpers** (it 10, sources `4328315ba9d33daa`): `fail` — 95 dpct helper use(s) remain in 2 file(s) (macro=43, device_queue_management=37, memory_management=3, error_handling=3)
- **verify** (it 9, sources `4328315ba9d33daa`): `ok` — PASS: 1/1 case(s) match (worst max abs err 2.48e-05, atol 0.0001)
- **benchmark:** not run

Current sources hash: `093a491080dc51c0`. Success requires compile, verify and zero-helper scan all on this hash.
