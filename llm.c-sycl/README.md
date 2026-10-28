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
- [x] **`train_gpt2_fp32.cpp`:** llm.c's fp32 GPT-2 trainer, with llm.c's real `main` (options,
  dataloaders, validation, sampling).
  - **How it was made:** Muse wrapped its proven GPT-2 model port in llm.c's `main`, keeping 94%
    of the proven lines verbatim, in 19 turns for $0.71.
  - **Build:** `make` builds `train_gpt2fp32sycl`.
  - **Check:** on a short tiny-shakespeare run, the validation and training losses it prints
    match llm.c's own CPU trainer within 1e-2 (worst 0.00227)
    ([evidence](../evidence/j3-trainer/README.md)).
- [ ] **`test_gpt2_fp32.cpp`:** llm.c's own test (`overall okay: 1`) on a CPU, an NVIDIA GPU and
  an AMD GPU.

**`llmc/`** holds llm.c's plain-C host headers, unchanged. `LICENSE` is llm.c's (MIT).
