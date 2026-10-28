# Harness specs written by Muse and proven

Model `muse-spark-1.3`, spent $1.0035 (GPU proofs $0.011). Worked examples shown to Muse (not graded): gelu_forward.cu, global_norm.cu.

| file | proven | attempts | variants | same as hand-written: variants / CPU functions / sizes / tol | cost |
|---|---|---|---|---|---|
| gelu_backward.cu | yes (CPU) | 1 | [1, 2] | yes / yes / yes / no | $0.0321 |
| residual_forward.cu | yes (CPU) | 1 | [1, 2] | yes / yes / yes / no | $0.0285 |
| permute.cu | yes (CPU) | 1 | [1] | yes / yes / no / yes | $0.0220 |
| crossentropy_forward.cu | yes (CPU) | 1 | [1] | yes / yes / no / yes | $0.0215 |
| crossentropy_softmax_backward.cu | yes (CPU) | 1 | [1] | yes / yes / no / yes | $0.0264 |
| encoder_forward.cu | yes (CPU) | 1 | [1, 2, 3] | yes / yes / no / yes | $0.0309 |
| encoder_backward.cu | yes (CPU) | 1 | [1, 2] | yes / yes / no / yes | $0.0204 |
| adamw.cu | yes (CPU) | 1 | [1, 2] | yes / yes / no / yes | $0.0413 |
| fused_residual_forward.cu | yes (CPU) | 1 | [1, 2, 3, 4, 5, 6] | yes / yes / no / yes | $0.0399 |
| layernorm_forward.cu | yes (CPU) | 1 | [1, 2, 3, 4, 5, 6] | yes / yes / no / yes | $0.0283 |
| softmax_forward.cu | no | 3 |  |  | $0.1289 |
| classifier_fused.cu | no | 3 |  |  | $0.1941 |
| layernorm_backward.cu | yes (CPU) | 3 | [1, 2, 3, 5, 6, 7, 8, 9, 10] | yes / no / no / yes | $0.2026 |
| matmul_forward.cu | yes (GPU) | 1 | [1, 2, 3, 4] | yes / yes / no / yes | $0.0320 |
| matmul_backward.cu | yes (CPU) | 1 | [1] | yes / yes / no / yes | $0.0352 |
| trimat_forward.cu | no | 2 |  |  | $0.1195 |
