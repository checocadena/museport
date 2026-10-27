# museport port (muse-spark-1.3)

Spent: $1.4502 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t1-fused-residual-forward` | fused_residual_forward.cu | budget | 12 | $0.5391 | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) (98 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05); 45 helpers | 7 / 6 |
| `t2-layernorm-backward` | layernorm_backward.cu | budget | 14 | $0.7144 | PASS: 1/1 case(s) match (worst max abs err 1.53e-05, atol 0.001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.53e-05, atol 0.001); 0 helpers | 10 / 9 |

Earlier runs (counted in the spend):

- `t2-layernorm-backward`: error (ConnectionResetError: [Errno 104] Connection reset by peer), 6 turns, $0.1967

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
