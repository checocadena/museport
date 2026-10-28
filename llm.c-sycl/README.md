# llm.c-sycl

[karpathy/llm.c](https://github.com/karpathy/llm.c), ported from CUDA to SYCL 2020 by Meta's
`muse-spark-1.3` with [Muse Port](../README.md). Every piece here passed a hidden check the
model never saw before it was kept. The checks are built from llm.c's own CPU code and
tolerances.

**Status (plan item J3; `docs/J3-PLAN.md`):**

- [x] **`dev/sycl/`:** all 19 llm.c `dev/cuda` kernel files we ported, unchanged from the
  graded runs.
  - `make -C dev/sycl` builds them for the CPU device; `TARGET=nvidia` or `TARGET=amd` builds
    for a GPU.
  - `museport llmc-sycl check` grades every one on the CPU: **19 of 19 pass, with 0 SYCLomatic
    helpers**.
  - Each kernel's NVIDIA and AMD results are in its evidence folder ([PROVENANCE](dev/sycl/PROVENANCE.json)).
- [ ] **`train_gpt2_fp32.cpp`:** llm.c's fp32 trainer, with its real `main` (Muse, next step).
- [ ] **`test_gpt2_fp32.cpp`:** llm.c's own test (`overall okay: 1`) on a CPU, an NVIDIA GPU and
  an AMD GPU.

**`llmc/`** holds llm.c's plain-C host headers, unchanged. `LICENSE` is llm.c's (MIT).
