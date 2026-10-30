# Port plan (muse-spark-1.3)

Migrate foundation llmc/cuda_utils.cuh first (all kernels include it) then Tier1 pointwise kernels easiest to hardest (gelu, global_norm, fused_classifier, adamw, encoder, layernorm), then Tier3 library items (matmul oneMath+manual GELU epilogue, attention fallback, zero oneCCL, train orchestration with oneMath/oneDNN/oneCCL), finally test harness. Two ordering exceptions are intentional: Tier2 foundation precedes Tier1 dependents, and Tier1 test runs last as end-to-end validation.

- **Grounded:** NO (files and kernels it names exist in the repository)
- **Coverage:** 12/12 CUDA files placed (12 planned, the rest out of scope)
- **Issues found by the grounding check:** 1
- **Tokens:** 88393 in (0 cached), 16221 out; **cost** $0.1794

## Items

| # | Item | Tier | Files | Kernels | Depends on |
|---|---|---|---|---|---|
| 1 | `s01-cuda-utils` SYCL foundation utils and reductions | 2 | llmc/cuda_utils.cuh | 2 | - |
| 2 | `s02-gelu` GeLU pointwise kernels | 1 | llmc/gelu.cuh | 2 | s01-cuda-utils |
| 3 | `s03-global-norm` Global norm reductions | 1 | llmc/global_norm.cuh | 3 | s01-cuda-utils |
| 4 | `s04-fused-classifier` Fused classifier softmax+loss | 1 | llmc/fused_classifier.cuh | 1 | s01-cuda-utils |
| 5 | `s05-adamw` AdamW optimizer kernels | 1 | llmc/adamw.cuh | 2 | s01-cuda-utils |
| 6 | `s06-encoder` Encoder token+position kernels | 1 | llmc/encoder.cuh | 3 | s01-cuda-utils |
| 7 | `s07-layernorm` LayerNorm and residual kernels | 1 | llmc/layernorm.cuh | 5 | s01-cuda-utils |
| 8 | `s08-matmul` MatMul with oneMath epilogues | 3 | llmc/matmul.cuh | 2 | s01-cuda-utils, s02-gelu |
| 9 | `s09-attention` Attention fallback kernels | 3 | llmc/attention.cuh | 6 | s01-cuda-utils, s08-matmul |
| 10 | `s10-zero` ZeRO sharding with oneCCL | 3 | llmc/zero.cuh | 0 | s01-cuda-utils |
| 11 | `s11-train` Training loop orchestration | 3 | train_gpt2.cu | 0 | s01-cuda-utils, s02-gelu, s03-global-norm, s04-fused-classifier, s05-adamw, s06-encoder, s07-layernorm, s08-matmul, s09-attention, s10-zero |
| 12 | `s12-test` SYCL correctness harness | 1 | test_gpt2.cu | 0 | s11-train |

## Grounding findings

- **unknown_kernels**: [{"item": "s03-global-norm", "kernel": "global_sum_single_block_kernel"}]
