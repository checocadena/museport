# museport port (muse-spark-1.3)

Spent: $1.1803 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t1-crossentropy-forward` | crossentropy_forward.cu | model_finished | 6 | $0.0673 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05); 0 helpers | 1 / 1 (recomputed; the run recorded 0 / 1, a false flag) |
| `t1-crossentropy-softmax-backward` | crossentropy_softmax_backward.cu | model_finished | 9 | $0.0968 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-permute` | permute.cu | model_finished | 8 | $0.0774 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 1 / 1 |
| `t1-residual-forward` | residual_forward.cu | model_finished | 7 | $0.0801 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-gelu-forward` | gelu_forward.cu | model_finished | 8 | $0.0859 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.19e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-gelu-backward` | gelu_backward.cu | max_iterations | 12 | $0.1488 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-encoder-forward` | encoder_forward.cu | max_iterations | 12 | $0.1667 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); 0 helpers | 3 / 3 |
| `t1-encoder-backward` | encoder_backward.cu | model_finished | 9 | $0.0890 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-05); 0 helpers | 2 / 2 |
| `t1-adamw` | adamw.cu | model_finished | 10 | $0.1205 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.79e-07, atol 0.0001); 0 helpers | 2 / 2 |
| `t1-fused-residual-forward` | fused_residual_forward.cu | model_finished | 10 | $0.2478 | - | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05); 0 helpers | 5 / 6 **duplicated** |

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
