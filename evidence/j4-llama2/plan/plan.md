# Port plan (muse-spark-1.3)

Single-item port of llama2.cu (8 kernels) to SYCL 2020 for NVIDIA/AMD/Intel GPUs: prove pipeline on mechanical elementwise/convert/RoPE kernels first, then replace CUB BlockReduce/WarpReduce and shared-memory reductions with portable group algorithms without assuming sub-group 32 or 1024 work-group support, verified by per-kernel CPU-reference harnesses and deterministic end-to-end transformer check.

- **Grounded:** yes (files and kernels it names exist in the repository)
- **Coverage:** 1/1 CUDA files placed (1 planned, the rest out of scope)
- **Issues found by the grounding check:** 0
- **Tokens:** 9519 in (0 cached), 3933 out; **cost** $0.0286

## Items

| # | Item | Tier | Files | Kernels | Depends on |
|---|---|---|---|---|---|
| 1 | `item-01-llama2-full-sycl` Port llama2.cu Llama-2 inference to SYCL 2020 | 3 | llama2.cu | 8 | - |
