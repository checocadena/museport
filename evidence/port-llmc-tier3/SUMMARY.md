# museport port (muse-spark-1.3)

Spent: $1.3960 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t3-matmul-backward` | matmul_backward.cu | max_iterations | 16 | $0.2795 | PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001) (105 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001); 0 helpers | 2 + 2 oneMath / 1 |
| `t3-trimat-forward` | trimat_forward.cu | success | 9 | $0.2112 | PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.0001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.0001); 0 helpers | 5 + 1 oneMath / 5 |
| `t3-matmul-backward-bias` | matmul_backward_bias.cu | budget | 14 | $0.5007 | PASS: 1/1 case(s) match (worst max abs err 1.53e-05, atol 0.005) (95 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 1.53e-05, atol 0.005); 0 helpers | 12 / 8 |
| `t3-matmul-forward` | matmul_forward.cu | success | 15 | $0.4046 | PASS: 1/1 case(s) match (worst max abs err 5.72e-05, atol 0.1) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 5.72e-05, atol 0.1); 0 helpers | 3 + 2 oneMath / 4 |

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
