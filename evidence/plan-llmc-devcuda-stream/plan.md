# Port plan (muse-spark-1.3)

Migrate 22 CUDA files via SYCL 2020 in dependency order: common scaffolding first, then 10 Tier1 plain kernels, then 5 Tier2 sub-group/cooperative/bf16 designs, then 7 Tier3 oneMath/oneDNN/oneCCL replacements with NCCL last; each item verified by its existing CPU reference and validate_result tolerances.

- **Grounded:** yes (files and kernels it names exist in the repository)
- **Coverage:** 22/22 CUDA files placed (22 planned, the rest out of scope)
- **Issues found by the grounding check:** 0
- **Tokens:** 135343 in (0 cached), 22499 out; **cost** $0.2648

## Items

| # | Item | Tier | Files | Kernels | Depends on |
|---|---|---|---|---|---|
| 1 | `common` SYCL common scaffolding | 1 | common.h | 0 | - |
| 2 | `t1-crossentropy-forward` Tier1 crossentropy forward | 1 | crossentropy_forward.cu | 1 | common |
| 3 | `t1-crossentropy-softmax-backward` Tier1 crossentropy softmax backward | 1 | crossentropy_softmax_backward.cu | 1 | common |
| 4 | `t1-permute` Tier1 permute | 1 | permute.cu | 1 | common |
| 5 | `t1-residual-forward` Tier1 residual forward | 1 | residual_forward.cu | 2 | common |
| 6 | `t1-gelu-forward` Tier1 GELU forward | 1 | gelu_forward.cu | 2 | common |
| 7 | `t1-gelu-backward` Tier1 GELU backward | 1 | gelu_backward.cu | 2 | common |
| 8 | `t1-encoder-forward` Tier1 encoder forward | 1 | encoder_forward.cu | 3 | common |
| 9 | `t1-encoder-backward` Tier1 encoder backward | 1 | encoder_backward.cu | 2 | common |
| 10 | `t1-adamw` Tier1 AdamW | 1 | adamw.cu | 2 | common |
| 11 | `t1-fused-residual-forward` Tier1 fused residual forward | 1 | fused_residual_forward.cu | 7 | common |
| 12 | `t2-global-norm` Tier2 global norm | 2 | global_norm.cu | 5 | common |
| 13 | `t2-layernorm-forward` Tier2 layernorm forward | 2 | layernorm_forward.cu | 8 | common |
| 14 | `t2-softmax-forward` Tier2 softmax forward | 2 | softmax_forward.cu | 8 | common |
| 15 | `t2-classifier-fused` Tier2 fused classifier | 2 | classifier_fused.cu | 5 | common |
| 16 | `t2-layernorm-backward` Tier2 layernorm backward | 2 | layernorm_backward.cu | 11 | common |
| 17 | `t3-matmul-backward` Tier3 matmul backward | 3 | matmul_backward.cu | 2 | common |
| 18 | `t3-trimat-forward` Tier3 triangular matmul forward | 3 | trimat_forward.cu | 2 | common |
| 19 | `t3-matmul-backward-bias` Tier3 matmul backward bias | 3 | matmul_backward_bias.cu | 10 | common |
| 20 | `t3-matmul-forward` Tier3 matmul forward | 3 | matmul_forward.cu | 3 | common |
| 21 | `t3-attention-backward` Tier3 attention backward | 3 | attention_backward.cu | 13 | common |
| 22 | `t3-attention-forward` Tier3 attention forward | 3 | attention_forward.cu | 15 | common |
| 23 | `t3-nccl-all-reduce` Tier3 NCCL all-reduce | 3 | nccl_all_reduce.cu | 1 | common |
