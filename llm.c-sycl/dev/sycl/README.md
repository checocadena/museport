# dev/sycl: llm.c's kernel files, ported to SYCL by Muse

Each folder holds Muse's port of one llm.c `dev/cuda` file, exactly as Muse last compiled it and as the hidden check graded it. `make` builds them all; ports that call oneMath need `ONEMATH_PREFIX`.

| kernel | source | hidden grade | Muse turns | cost | evidence |
|---|---|---|---|---|---|
| `gelu_forward` | gelu_forward_sycl.cpp | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) | 7 | $0.0899 | `evidence/port-llmc-tier1-run2/t1-gelu-forward` |
| `gelu_backward` | gelu_backward.cpp | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) | 11 | $0.1502 | `evidence/port-llmc-tier1-run2/t1-gelu-backward` |
| `residual_forward` | residual_forward.cpp | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | 9 | $0.0887 | `evidence/port-llmc-tier1-run2/t1-residual-forward` |
| `permute` | permute.cpp | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | 7 | $0.0526 | `evidence/port-llmc-tier1-run2/t1-permute` |
| `crossentropy_forward` | crossentropy_forward_sycl.cpp | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05) | 9 | $0.0904 | `evidence/port-llmc-tier1-run2/t1-crossentropy-forward` |
| `crossentropy_softmax_backward` | crossentropy_softmax_backward.cpp | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | 6 | $0.0654 | `evidence/port-llmc-tier1-run2/t1-crossentropy-softmax-backward` |
| `encoder_forward` | encoder_forward.cpp | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) | 7 | $0.0842 | `evidence/port-llmc-tier1-run2/t1-encoder-forward` |
| `encoder_backward` | encoder_backward.cpp | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-05) | 7 | $0.0782 | `evidence/port-llmc-tier1-run2/t1-encoder-backward` |
| `adamw` | adamw_sycl.cpp | PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001) | 7 | $0.0832 | `evidence/port-llmc-tier1-run2/t1-adamw` |
| `fused_residual_forward` | fused_residual_forward.cpp | PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 1e-05) | 12 | $0.2235 | `evidence/port-llmc-tier1-run2/t1-fused-residual-forward` |
| `global_norm` | global_norm.cpp | PASS: 1/1 case(s) match (worst max abs err 1.31e-06, atol 0.01) | 16 | $0.3230 | `evidence/port-llmc-tier23-run2/t2-global-norm` |
| `layernorm_forward` | layernorm.cpp | PASS: 1/1 case(s) match (worst max abs err 1.07e-06, atol 1e-05) | 7 | $0.1524 | `evidence/port-llmc-tier23-run2/t2-layernorm-forward` |
| `classifier_fused` | classifier_fused.cpp | PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001) | 7 | $0.1397 | `evidence/port-llmc-tier23-run2/t2-classifier-fused` |
| `layernorm_backward` | layernorm_backward_sycl.cpp | PASS: 1/1 case(s) match (worst max abs err 1.91e-05, atol 0.001) | 14 | $0.4848 | `evidence/port-llmc-tier23-run2/t2-layernorm-backward` |
| `matmul_forward` | matmul_forward.cpp | PASS: 1/1 case(s) match (worst max abs err 5.72e-05, atol 0.1) | 12 | $0.2449 | `evidence/port-llmc-tier23-run2/t3-matmul-forward` |
| `trimat_forward` | trimat_forward.cpp | PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.0001) | 16 | $0.3740 | `evidence/port-llmc-tier23-run2/t3-trimat-forward` |
| `matmul_backward_bias` | matmul_backward_bias.cpp | PASS: 1/1 case(s) match (worst max abs err 7.63e-06, atol 0.005) | 13 | $0.3278 | `evidence/port-llmc-tier23-run2/t3-matmul-backward-bias` |
| `softmax_forward` | softmax_forward.cpp | PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001) | 23 | $0.9063 | `evidence/port-llmc-softmax-rerun-2/t2-softmax-forward` |
| `matmul_backward` | matmul_backward_sycl.cpp | PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001) | 7 | $0.0943 | `evidence/port-llmc-matmul-backward-colmajor/t3-matmul-backward` |
