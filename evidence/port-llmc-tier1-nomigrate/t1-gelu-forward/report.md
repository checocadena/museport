# Port report: llmc_gelu_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 6 / 12
- **Tokens:** in 26840, out 3395, cache write 0, cache read 12115; est. cost $0.0498
- **Best passing candidate:** iteration 6, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | list_files | ok | 1 file(s) |
| 2 | read_file | ok | common.h: lines 1-200 of 359 (7436 chars) |
| 3 | read_file | ok | common.h: lines 200-359 of 359 (5334 chars) |
| 4 | write_file | ok | wrote gelu_forward.cpp (4618 chars) |
| 5 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-nomigrate-2/t1-gelu-forward/2... |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 6 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) |
| 6 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 313.64 ms, p95 377.29 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate:** not run
- **compile** (it 5, sources `4207920808f8d466`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `4207920808f8d466`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 6, sources `4207920808f8d466`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05)
- **benchmark** (it 6, sources `4207920808f8d466`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 313.64 ms, p95 377.29 ms

Current sources hash: `4207920808f8d466`. Success requires compile, verify and zero-helper scan all on this hash.
