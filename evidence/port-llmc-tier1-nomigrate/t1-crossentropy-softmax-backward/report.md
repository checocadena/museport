# Port report: llmc_crossentropy_softmax_backward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 4 / 12
- **Tokens:** in 10841, out 2087, cache write 0, cache read 6370; est. cost $0.0234
- **Best passing candidate:** iteration 4, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | list_files | ok | 1 file(s) |
| 2 | write_file | ok | wrote candidate.cpp (4005 chars) |
| 3 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-nomigrate-2/t1-crossentropy-s... |
| 4 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 4 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 342.48 ms, p95 370.30 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate:** not run
- **compile** (it 3, sources `69d4d8a35741446a`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 4, sources `69d4d8a35741446a`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 4, sources `69d4d8a35741446a`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **benchmark** (it 4, sources `69d4d8a35741446a`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 342.48 ms, p95 370.30 ms

Current sources hash: `69d4d8a35741446a`. Success requires compile, verify and zero-helper scan all on this hash.
