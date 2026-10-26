# museport port (muse-spark-1.3)

Spent: $2.4454 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t2-global-norm` | global_norm.cu | success | 13 | $0.2598 | PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 0.01) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.01); 0 helpers | 5 / 4 |
| `t2-layernorm-forward` | layernorm_forward.cu | budget | 13 | $0.4512 | PASS: 1/1 case(s) match (worst max abs err 1.07e-06, atol 1e-05) (88 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.07e-06, atol 1e-05); 0 helpers | 8 / 6 |
| `t2-softmax-forward` | softmax_forward.cu | budget | 14 | $0.5402 | PASS: 1/1 case(s) match (worst max abs err 5.77e-05, atol 0.0001) (85 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 5.77e-05, atol 0.0001); 45 helpers | 8 / 8 |
| `t2-classifier-fused` | classifier_fused.cu | budget | 14 | $0.4562 | PASS: 1/1 case(s) match (worst max abs err 2.48e-05, atol 0.0001) (95 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 2.48e-05, atol 0.0001); 0 helpers | 5 / 5 |
| `t2-layernorm-backward` | layernorm_backward.cu | success | 11 | $0.4797 | PASS: 1/1 case(s) match (worst max abs err 1.14e-05, atol 0.001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.14e-05, atol 0.001); 0 helpers | 2 / 9 **duplicated** |

Earlier runs (counted in the spend):

- `t2-layernorm-backward`: error (KeyError: 'path' (toolkit Toolbox.t_write_file; a write_file call cut off by the output limit)), 6 turns, $0.2583

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
