# museport port (muse-spark-1.3)

Spent: $0.4759 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t1-crossentropy-forward` | crossentropy_forward.cu | success | 4 | $0.0191 | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-crossentropy-softmax-backward` | crossentropy_softmax_backward.cu | success | 4 | $0.0234 | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-permute` | permute.cu | success | 5 | $0.0207 | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-residual-forward` | residual_forward.cu | success | 5 | $0.0284 | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-gelu-forward` | gelu_forward.cu | success | 6 | $0.0498 | PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05); 0 helpers | 3 / 2 |
| `t1-gelu-backward` | gelu_backward.cu | success | 4 | $0.0382 | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-encoder-forward` | encoder_forward.cu | success | 6 | $0.0654 | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 3 / 3 |
| `t1-encoder-backward` | encoder_backward.cu | success | 8 | $0.0689 | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-adamw` | adamw.cu | success | 7 | $0.0732 | PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001); 0 helpers | 2 / 2 |
| `t1-fused-residual-forward` | fused_residual_forward.cu | success | 4 | $0.0888 | PASS: 1/1 case(s) match (worst max abs err 2.03e-06, atol 1e-05) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 2.03e-06, atol 1e-05); 0 helpers | 7 / 6 |

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
