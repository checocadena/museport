# Harness specs written by Muse and proven

Model `muse-spark-1.3`, spent $0.0469 (GPU proofs $0.000). Worked examples shown to Muse (not graded): gelu_forward.cu, global_norm.cu.

| file | proven | attempts | variants | same as hand-written: variants / CPU functions / sizes / tol | cost |
|---|---|---|---|---|---|
| gelu_backward.cu | yes (CPU) | 1 | [1, 2] | yes / yes / yes / yes | $0.0229 |
| residual_forward.cu | yes (CPU) | 1 | [1, 2] | yes / yes / yes / yes | $0.0240 |
