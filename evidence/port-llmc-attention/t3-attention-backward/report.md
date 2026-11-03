# Port report: llmc_attention_backward

- **Result:** `max_iterations` — reached 20 iterations without meeting success conditions
- **Model:** `muse-spark-1.3`
- **Iterations:** 20 / 20
- **Tokens:** in 355596, out 90576, cache write 0, cache read 625597; est. cost $0.9233
- **Best passing candidate:** iteration 14, 127 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 43 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x4, DPCT1024x2, DPCT1049x13, DPCT1051x2, DPCT10... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | attention_backward_harness.dp.cpp: lines 1-80 of 1398 (3252 chars) |
| 4 | read_file | ok | attention_backward_harness.dp.cpp: lines 80-229 of 1398 (5972 chars) |
| 5 | read_file | ok | attention_backward_harness.dp.cpp: lines 860-1059 of 1398 (9158 chars) |
| 6 | read_file | ok | attention_backward_harness.dp.cpp: lines 1060-1398 of 1398 (16737 chars) |
| 7 | read_file | ok | attention_backward_harness.dp.cpp: lines 230-529 of 1398 (13334 chars) |
| 8 | read_file | ok | attention_backward_harness.dp.cpp: lines 530-729 of 1398 (9246 chars) |
| 9 | read_file | ok | attention_backward_harness.dp.cpp: lines 730-889 of 1398 (6595 chars) |
| 10 | read_file | ok | common.h: lines 1-60 of 480 (2231 chars) |
| 11 | write_file | ok | wrote attention_backward.cpp (30822 chars) |
| 12 | edit_file | ok | edited attention_backward.cpp at line 434 (781 -> 127 chars; file now 698 lines) |
| 13 | compile | ok | compiled 1 source(s) -> work/build/candidate (42 war... |
| 14 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.001) |
| 15 | scan_helpers | fail | 127 dpct helper use(s) remain in 2 file(s) (macro=55, device_queue_management=50, library_wrappers=6, include=5) |
| 16 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 17 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 18 | compile | ok | compiled 1 source(s) -> work/build/candidate (42 war... |
| 19 | read_file | ok | attention_backward.cpp: lines 370-419 of 698 (2005 chars) |
| 20 | edit_file | ok | edited attention_backward.cpp at line 394 (353 -> 387 chars; file now 698 lines) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `1f1b000e806e3b7b`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 43 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x4, DPCT1024x2, DPCT1049x13, DPCT1051x2, DPCT1065x6, DPCT1093x2, DPCT1098x7, DPCT1110x3, DPCT1118x1)
- **compile** (it 18, sources `49be03bd4564016b`): `ok` — compiled 1 source(s) -> work/build/candidate (42 warning(s))
- **scan_helpers** (it 17, sources `49be03bd4564016b`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 14, sources `800818b01edd2de1`): `ok` — PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.001)
- **benchmark:** not run

Current sources hash: `c15fcf3c8a3c10a9`. Success requires compile, verify and zero-helper scan all on this hash.
