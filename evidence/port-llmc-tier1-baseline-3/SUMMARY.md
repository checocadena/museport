# museport port (muse-spark-1.3)

Spent: $0.7245 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t1-crossentropy-forward` | crossentropy_forward.cu | model_finished | 7 | $0.0649 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-crossentropy-softmax-backward` | crossentropy_softmax_backward.cu | model_finished | 11 | $0.1066 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-permute` | permute.cu | model_finished | 7 | $0.0581 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-residual-forward` | residual_forward.cu | max_iterations | 12 | $0.1355 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-gelu-forward` | gelu_forward.cu | model_finished | 10 | $0.1454 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-gelu-backward` | gelu_backward.cu | model_finished | 11 | $0.0968 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-encoder-forward` | encoder_forward.cu | max_iterations | 12 | $0.1171 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 3 / 3 |

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
