# Port report: llmc_crossentropy_softmax_backward

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `muse-spark-1.3`
- **Iterations:** 12 / 12
- **Tokens:** in 124238, out 4983, cache write 0, cache read 100503; est. cost $0.1916
- **Best passing candidate:** iteration 12, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 18 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x2, DPCT1024x2, DPCT1049x1, DPCT1051x2, DPCT106... |
| 2 | list_files | ok | 4 file(s) |
| 3 | read_file | ok | crossentropy_softmax_backward_harness.dp.cpp: 7119 chars |
| 4 | read_file | ok | common.h: 16876 chars |
| 5 | write_file | ok | wrote crossentropy_softmax_backward.cpp (5068 chars) |
| 6 | read_file | ok | MainSourceFiles.yaml: 46969 chars |
| 7 | compile | ok | compiled 1 source(s) -> work/build/candidate (0 war... |
| 8 | write_file | ok | wrote migrated/crossentropy_softmax_backward_harness.dp.cpp (78 chars) |
| 9 | write_file | ok | wrote migrated/common.h (78 chars) |
| 10 | write_file | ok | wrote migrated/MainSourceFiles.yaml (77 chars) |
| 11 | compile | ok | compiled 1 source(s) -> work/build/candidate (0 war... |
| 12 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| 12 | scan_helpers | ok | no dpct helpers in 3 file(s) |
| 12 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 254.76 ms, p95 337.71 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `0a5ef6e04ff11d4f`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 18 DPCT warning(s) (DPCT1005x2, DPCT1009x1, DPCT1010x2, DPCT1024x2, DPCT1049x1, DPCT1051x2, DPCT1065x2, DPCT1093x2, DPCT1098x3, DPCT1118x1)
- **compile** (it 11, sources `cde480d10848ca59`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 12, sources `cde480d10848ca59`): `ok` — no dpct helpers in 3 file(s)
- **verify** (it 12, sources `cde480d10848ca59`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **benchmark** (it 12, sources `cde480d10848ca59`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 254.76 ms, p95 337.71 ms

Current sources hash: `cde480d10848ca59`. Success requires compile, verify and zero-helper scan all on this hash.
