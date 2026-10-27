# Port report: llmc_fused_residual_forward

- **Result:** `max_iterations` — reached 12 iterations without meeting success conditions
- **Model:** `muse-spark-1.3`
- **Iterations:** 12 / 12
- **Tokens:** in 100351, out 15397, cache write 0, cache read 217337; est. cost $0.2235
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT10... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | fused_residual_forward_harness.dp.cpp: lines 1-120 of 980 (4188 chars) |
| 4 | read_file | ok | common.h: lines 1-220 of 453 (8174 chars) |
| 5 | write_file | ok | wrote fused_residual_forward.cpp (19729 chars) |
| 6 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 7 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run2/t1-fused-residual-forwar... |
| 8 | read_file | ok | fused_residual_forward.cpp: lines 225-304 of 485 (3182 chars) |
| 9 | read_file | ok | fused_residual_forward.cpp: lines 1-60 of 485 (2358 chars) |
| 10 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 11 | read_file | ok | fused_residual_forward.cpp: lines 60-224 of 485 (7572 chars) |
| 12 | read_file | ok | fused_residual_forward.cpp: lines 304-485 of 485 (6689 chars) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `45600244f09fd8dc`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT1049x9, DPCT1051x2, DPCT1065x4, DPCT1083x2, DPCT1093x2, DPCT1098x3, DPCT1110x3, DPCT1118x3)
- **compile** (it 7, sources `34f8e02c82100e6d`): `ok` — compiled 1 source(s) -> work/build/candidate (4 warning(s))
- **scan_helpers** (it 10, sources `34f8e02c82100e6d`): `ok` — no dpct helpers in 1 file(s)
- **verify:** not run
- **benchmark:** not run

Current sources hash: `34f8e02c82100e6d`. Success requires compile, verify and zero-helper scan all on this hash.
