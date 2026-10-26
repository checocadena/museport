# Port report: llmc_fused_residual_forward

- **Result:** `max_iterations` — reached 12 iterations without meeting success conditions
- **Model:** `muse-spark-1.3`
- **Iterations:** 12 / 12
- **Tokens:** in 154841, out 22840, cache write 0, cache read 392012; est. cost $0.3494
- **Best passing candidate:** iteration 11, 98 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT10... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | fused_residual_forward_harness.dp.cpp: 40843 chars |
| 4 | read_file | ok | common.h: 16883 chars |
| 5 | read_file | ok | fused_residual_forward_harness.dp.cpp: 40843 chars |
| 6 | write_file | ok | wrote fused_residual_forward.sycl.cpp (25800 chars) |
| 7 | compile | fail | compile failed with 2 error(s) at work/fused_residual_forw... |
| 8 | read_file | ok | fused_residual_forward.sycl.cpp: 25800 chars |
| 9 | write_file | ok | wrote fused_residual_forward.sycl.cpp (21104 chars) |
| 10 | compile | ok | compiled 1 source(s) -> work/build/candidate (0 warning(s)) |
| 11 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) |
| 11 | scan_helpers | fail | 98 dpct helper use(s) remain in 2 file(s) (macro=44, device_queue_management=44, error_handling=3, events_timing=3) |
| 12 | write_file | ok | wrote migrated/common.h (163 chars) |

## What failed and what fixed it

- it 7: `compile:fused_residual_forward.sycl.cpp:327:use of undeclared identifier 'loc'; did you mean 'log'?` → fixed by iteration 10 after editing fused_residual_forward.sycl.cpp

## Final evidence

- **migrate** (it 1, sources `45600244f09fd8dc`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT1049x9, DPCT1051x2, DPCT1065x4, DPCT1083x2, DPCT1093x2, DPCT1098x3, DPCT1110x3, DPCT1118x3)
- **compile** (it 10, sources `7a69e7b8c737f237`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 11, sources `7a69e7b8c737f237`): `fail` — 98 dpct helper use(s) remain in 2 file(s) (macro=44, device_queue_management=44, error_handling=3, events_timing=3)
- **verify** (it 11, sources `7a69e7b8c737f237`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05)
- **benchmark:** not run

Current sources hash: `a44aa0c91cf68fda`. Success requires compile, verify and zero-helper scan all on this hash.
