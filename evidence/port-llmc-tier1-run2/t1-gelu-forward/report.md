# Port report: llmc_gelu_forward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 7 / 12
- **Tokens:** in 57715, out 3356, cache write 0, cache read 23379; est. cost $0.0899
- **Best passing candidate:** iteration 7, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT106... |
| 2 | read_file | ok | gelu_forward_harness.dp.cpp: lines 1-199 of 199 (7343 chars) |
| 3 | read_file | error | path escapes the work dir: /home/user/museport-hackathon/evidence/port-llmc-tier1-run2/t1-gelu-forward/task/common.h |
| 4 | read_file | ok | common.h: lines 1-220 of 453 (8169 chars) |
| 5 | write_file | ok | wrote gelu_forward_sycl.cpp (4513 chars) |
| 6 | delete_file | ok | deleted migrated/ (3 file(s)) |
| 6 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user/61dcfd45-e628-54eb-bcf3-8ee0da15b04b/scratchpad/runs-tier1-run2/t1-gelu-forward/20261027... |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) |
| 7 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 225.23 ms, p95 242.46 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `01cba48287b938df`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 20 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x3, DPCT1024x2, DPCT1049x2, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 6, sources `281aca61e9497720`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 7, sources `281aca61e9497720`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `281aca61e9497720`): `ok` — PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05)
- **benchmark** (it 7, sources `281aca61e9497720`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 225.23 ms, p95 242.46 ms

Current sources hash: `281aca61e9497720`. Success requires compile, verify and zero-helper scan all on this hash.
