# Port report: llmc_fused_residual_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 4 / 12
- **Tokens:** in 38694, out 9182, cache write 0, cache read 9329; est. cost $0.0888
- **Best passing candidate:** iteration 4, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | list_files | ok | 1 file(s) |
| 2 | write_file | ok | wrote fused_residual_forward_sycl.cpp (13510 chars) |
| 3 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-nomigrate-2/t1-fused-residual... |
| 4 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 4 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 2.03e-06, atol 1e-05) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 922.24 ms, p95 964.32 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate:** not run
- **compile** (it 3, sources `3556aee8a1993128`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 4, sources `3556aee8a1993128`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 4, sources `3556aee8a1993128`): `ok` — PASS: 1/1 case(s) match (worst max abs err 2.03e-06, atol 1e-05)
- **benchmark** (it 4, sources `3556aee8a1993128`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 922.24 ms, p95 964.32 ms

Current sources hash: `3556aee8a1993128`. Success requires compile, verify and zero-helper scan all on this hash.
