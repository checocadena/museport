# museport port (muse-spark-1.3)

Spent: $2.7433 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t2-global-norm` | global_norm.cu | max_iterations | 16 | $0.3230 | PASS: 1/1 case(s) match (worst max abs err 1.67e-06, atol 0.01) (70 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.31e-06, atol 0.01); 0 helpers | 5 / 4 |
| `t2-layernorm-forward` | layernorm_forward.cu | success | 7 | $0.1524 | PASS: 1/1 case(s) match (worst max abs err 1.07e-06, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.07e-06, atol 1e-05); 0 helpers | 8 / 6 |
| `t2-softmax-forward` | softmax_forward.cu | max_iterations | 16 | $0.5857 | - | verify fail: FAIL: 0/1 case(s) match; first failure: 36435 of 25731584 element(s) outside tolerance; 0 helpers | 6 / 8 **duplicated** |
| `t2-classifier-fused` | classifier_fused.cu | success | 7 | $0.1397 | PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001); 0 helpers | 5 / 5 |
| `t2-layernorm-backward` | layernorm_backward.cu | success | 14 | $0.4848 | PASS: 1/1 case(s) match (worst max abs err 2.1e-05, atol 0.001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.91e-05, atol 0.001); 0 helpers | 8 / 9 **duplicated** |
| `t3-matmul-backward` | matmul_backward.cu | success | 8 | $0.1108 | PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001); 0 helpers | 2 + 2 oneMath / 1 |
| `t3-trimat-forward` | trimat_forward.cu | max_iterations | 16 | $0.3740 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.0001); 0 helpers | 5 + 1 oneMath / 5 |
| `t3-matmul-backward-bias` | matmul_backward_bias.cu | success | 13 | $0.3278 | PASS: 1/1 case(s) match (worst max abs err 7.63e-06, atol 0.005) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 7.63e-06, atol 0.005); 0 helpers | 9 / 8 |
| `t3-matmul-forward` | matmul_forward.cu | success | 12 | $0.2449 | PASS: 1/1 case(s) match (worst max abs err 5.72e-05, atol 0.1) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 5.72e-05, atol 0.1); 0 helpers | 3 + 2 oneMath / 4 |

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
