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
- [x] **`test_gpt2_fp32.cpp`:** llm.c's own test.
  - **What it checks:** the logits, the loss and every gradient against PyTorch, then the
    losses of 10 training steps. The tolerances are llm.c's (1e-2).
  - **How it was made:** Muse wrote the test in 24 turns for $0.43, on its third run. It also
    guarded the trainer's `main` with `#ifndef TESTING`, as llm.c does.
  - **Build:** `make test_gpt2fp32sycl`. Run it next to `gpt2_124M.bin` and
    `gpt2_124M_debug_state.bin`.
  - **Result:** it prints `overall okay: 1`, worst loss error 0.000154
    ([evidence](../evidence/j3-test/README.md)).
- [x] **NVIDIA and AMD:** the test and the trainer pass on an NVIDIA RTX 2000 Ada and an AMD
  Instinct MI350X, with oneMath on cuBLAS and on rocBLAS.

  | Program | CPU | NVIDIA | AMD |
  |---|---|---|---|
  | `test_gpt2fp32sycl` (`overall okay: 1`; worst error vs PyTorch) | 0.000154 | 0.000127 | 0.0000236 |
  | `train_gpt2fp32sycl` (worst error vs llm.c's CPU trainer) | 0.00227 | 0.0023 | 0.00242 |

  On NVIDIA, both also match llm.c's own CUDA programs: 7.03e-06 for the test and 0.00147 for the
  trainer. The four GPU runs cost $0.43 in all, including one failed AMD setup. Evidence:
  [test](../evidence/j3-test/README.md), [trainer](../evidence/j3-trainer/README.md).
- **Note:** the GPU runs built the sources the graded runs compiled. The trainer here differs
  from them only by the `#ifndef TESTING` guard.

**`llmc/`** holds llm.c's plain-C host headers, unchanged. `LICENSE` is llm.c's (MIT).
