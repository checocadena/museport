# Harness specs written by Muse and proven

Model `muse-spark-1.3`, spent $0.1614. Worked examples shown to Muse (not graded): gelu_forward.cu, global_norm.cu.

| file | proven | attempts | variants | same as hand-written: variants / CPU functions / sizes / tol | cost |
|---|---|---|---|---|---|
| gelu_backward.cu | yes | 1 | [1, 2] | yes / yes / yes / no | $0.0321 |
| residual_forward.cu | yes | 1 | [1, 2] | yes / yes / yes / no | $0.0285 |
| permute.cu | yes | 1 | [1] | yes / yes / no / yes | $0.0220 |
| crossentropy_forward.cu | yes | 1 | [1] | yes / yes / no / yes | $0.0215 |
| crossentropy_softmax_backward.cu | yes | 1 | [1] | yes / yes / no / yes | $0.0264 |
| encoder_forward.cu | yes | 1 | [1, 2, 3] | yes / yes / no / yes | $0.0309 |
