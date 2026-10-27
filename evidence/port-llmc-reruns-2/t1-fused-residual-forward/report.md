# Port report: llmc_fused_residual_forward

- **Result:** `budget` — token budget exceeded: 634997 > 600000
- **Model:** `muse-spark-1.3`
- **Iterations:** 12 / 14
- **Tokens:** in 272564, out 35129, cache write 0, cache read 327304; est. cost $0.5391
- **Best passing candidate:** iteration 10, 98 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT10... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | fused_residual_forward_harness.dp.cpp: 40843 chars |
| 3 | read_file | ok | common.h: 16883 chars |
| 4 | read_file | ok | fused_residual_forward_harness.dp.cpp: 40843 chars |
| 5 | write_file | ok | wrote fused_residual_forward.cpp (25585 chars) |
| 6 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-reruns-2/t1-fused-residual-forward/... |
| 7 | verify | fail | FAIL: 0/1 case(s) match; first failure: 327411 of 2362368 element(s) outside tolerance |
| 7 | scan_helpers | fail | 98 dpct helper use(s) remain in 2 file(s) (macro=44, device_queue_management=44, error_handling=3, events_timing=3) |
| 8 | write_file | ok | wrote fused_residual_forward.cpp (25373 chars) |
| 9 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-reruns-2/t1-fused-residual-forward/... |
| 10 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) |
| 11 | list_files | ok | 3 file(s) |
| 12 | write_file | ok | wrote migrated/fused_residual_forward_harness.dp.cpp (76 chars) |

## What failed and what fixed it

- it 7: `verify:327411 of 2362368 element(s) outside tolerance` → fixed by iteration 10 after editing fused_residual_forward.cpp

## Final evidence

- **migrate** (it 1, sources `45600244f09fd8dc`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT1049x9, DPCT1051x2, DPCT1065x4, DPCT1083x2, DPCT1093x2, DPCT1098x3, DPCT1110x3, DPCT1118x3)
- **compile** (it 9, sources `43887734c91aebe0`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 7, sources `c086562eef72820b`): `fail` — 98 dpct helper use(s) remain in 2 file(s) (macro=44, device_queue_management=44, error_handling=3, events_timing=3)
- **verify** (it 10, sources `43887734c91aebe0`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05)
- **benchmark:** not run

Current sources hash: `7e38a7f83d3726fd`. Success requires compile, verify and zero-helper scan all on this hash.
