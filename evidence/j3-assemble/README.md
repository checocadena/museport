# J3 step 1: the kernel ports in `llm.c-sycl/dev/sycl`

`museport llmc-sycl assemble` copies the 19 llm.c `dev/cuda` files that Muse ported into
[`llm.c-sycl/dev/sycl`](../../llm.c-sycl/dev/sycl/README.md). They go in unchanged from their
graded runs, each with its provenance.

`museport llmc-sycl check --json evidence/j3-assemble/check.json` grades each one, as copied,
with its own hidden check on the CPU device:

- **Result:** **19 of 19 pass, with 0 SYCLomatic helpers** ([`check.json`](check.json)).
- **When:** run again on Oct 28 to record this file. It took 3 min 41 s and cost $0.
