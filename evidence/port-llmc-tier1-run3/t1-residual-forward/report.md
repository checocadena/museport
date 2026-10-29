# Port report: llmc_residual_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 6 / 12
- **Tokens:** in 33308, out 3713, cache write 0, cache read 27076; est. cost $0.0615
- **Best passing candidate:** iteration 6, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | write_file | ok | wrote residual_forward.cpp (5157 chars) |
| 3 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run3/t1-residual-forward/2026... |
| 4 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 4 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| 5 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run3/t1-residual-forward/2026... |
| 5 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 6 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| 6 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 253.62 ms, p95 283.19 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `6b9f943dc408d183`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 5, sources `e777c02e973312c3`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 5, sources `e777c02e973312c3`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 6, sources `e777c02e973312c3`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **benchmark** (it 6, sources `e777c02e973312c3`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 253.62 ms, p95 283.19 ms

Current sources hash: `e777c02e973312c3`. Success requires compile, verify and zero-helper scan all on this hash.
