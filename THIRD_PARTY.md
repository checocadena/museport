# What is pre-existing, and third-party material

## Pre-existing: oneapi-agent-toolkit (the author's own, used as a dependency)

[oneapi-agent-toolkit](https://github.com/checocadena/oneapi-agent-toolkit) (AGPL-3.0) existed
before the hackathon (tag `pre_hackathon`). It is **not part of this repository**: `pip`
installs it from its own repository, pinned in `pyproject.toml`. It provides:
- the evidence-returning tools (migrate, compile, scan_helpers, verify, benchmark);
- the agent loop and its model client;
- the GPU test pipeline (`remote-test`).

Changes made to it during the hackathon are listed in
[`docs/CHANGES-TO-TOOLKIT.md`](docs/CHANGES-TO-TOOLKIT.md) and live in that repository.

The run outputs under `evidence/` were produced during the event by running the toolkit:
reports, traces, and GPU results.

## Third-party: llm.c (MIT)

The ported kernels come from [karpathy/llm.c](https://github.com/karpathy/llm.c) `dev/cuda`
at commit `f1e2ace`, Copyright (c) 2024 Andrej Karpathy, MIT License
([`evidence/LICENSE-llm.c`](evidence/LICENSE-llm.c)). These evidence files contain llm.c
code:
- `evidence/port-llmc-tier*/<item>/task/`:
  - `*_harness.cu`: llm.c's file with museport's mechanical changes;
  - `common.h`: llm.c's, trimmed;
  - `*_ref.cpp`: llm.c's CPU reference functions, extracted verbatim.
- The test fixture `tests/fixtures/llmc_mini/` is not llm.c code: it was written for museport's
  tests in llm.c's layout.

## Third-party: llama2.cu (MIT)

[ankan-ban/llama2.cu](https://github.com/ankan-ban/llama2.cu) at commit `cb0c6fa`, a CUDA port
of Karpathy's llama2.c, MIT License ("Copyright (c) 2023 Andrej")
([`llama2-sycl/LICENSE`](llama2-sycl/LICENSE)). It is used in:
- `llama2-sycl/`: the upstream repo's files, unchanged, except `llama2.cu`, which is replaced
  by Muse's port `llama2.cpp`. Its Makefile has a `llama2` target appended, and its README is
  kept as `README-llama2.c.md`.
- `evidence/j4-llama2/task/`: `llama2.cu` and `tokenizer.bin`, unchanged.
- `evidence/j4-llama2/planted/`: two one-line changes to llama2.c's `run.c`, as diffs.

## Everything else

Everything else in this repository was written during the hackathon:
- the code in `src/museport/` and `tests/`;
- the documents;
- the specs and generated harnesses;
- Muse's ports under `evidence/`, written by `muse-spark-1.3` during the runs.
