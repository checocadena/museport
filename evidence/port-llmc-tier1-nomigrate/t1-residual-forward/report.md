# Port report: llmc_residual_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 5 / 12
- **Tokens:** in 12515, out 2469, cache write 0, cache read 15300; est. cost $0.0284
- **Best passing candidate:** iteration 5, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | list_files | ok | 1 file(s) |
| 2 | read_file | ok | common.h: lines 1-200 of 359 (7436 chars) |
| 3 | write_file | ok | wrote residual_forward_sycl.cpp (4691 chars) |
| 4 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-nomigrate-2/t1-residual-forwa... |
| 5 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 5 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| 5 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 234.27 ms, p95 255.14 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate:** not run
- **compile** (it 4, sources `ccca33e5be128d89`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 5, sources `ccca33e5be128d89`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 5, sources `ccca33e5be128d89`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **benchmark** (it 5, sources `ccca33e5be128d89`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 234.27 ms, p95 255.14 ms

Current sources hash: `ccca33e5be128d89`. Success requires compile, verify and zero-helper scan all on this hash.
