# Port report: llmc_fused_residual_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 7 / 12
- **Tokens:** in 93757, out 8776, cache write 0, cache read 52036; est. cost $0.1623
- **Best passing candidate:** iteration 7, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT10... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | fused_residual_forward_harness.dp.cpp: lines 1-120 of 980 (4188 chars) |
| 4 | read_file | ok | common.h: lines 1-250 of 359 (8760 chars) |
| 5 | write_file | ok | wrote fused_residual_forward.cpp (15457 chars) |
| 6 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 6 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run3/t1-fused-residual-forwar... |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.91e-06, atol 1e-05) |
| 7 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 837.89 ms, p95 886.74 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `45600244f09fd8dc`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 47 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x10, DPCT1024x2, DPCT1026x2, DPCT1027x2, DPCT1049x9, DPCT1051x2, DPCT1065x4, DPCT1083x2, DPCT1093x2, DPCT1098x3, DPCT1110x3, DPCT1118x3)
- **compile** (it 6, sources `b2ba071b4d9e23e3`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 7, sources `b2ba071b4d9e23e3`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `b2ba071b4d9e23e3`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.91e-06, atol 1e-05)
- **benchmark** (it 7, sources `b2ba071b4d9e23e3`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 837.89 ms, p95 886.74 ms

Current sources hash: `b2ba071b4d9e23e3`. Success requires compile, verify and zero-helper scan all on this hash.
