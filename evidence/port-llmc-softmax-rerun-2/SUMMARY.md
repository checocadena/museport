# museport port (muse-spark-1.3)

Spent: $0.9063 (every run, including earlier runs of an item and runs that ended in an error, counted from their reported usage)

| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of final sources | Distinct kernels / variants |
|---|---|---|---|---|---|---|---|
| `t2-softmax-forward` | softmax_forward.cu | success | 23 | $0.9063 | PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001) (0 helpers) | verify ok: PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001); 0 helpers | 8 / 8 |

`success`: compile, verify against the hidden reference and a zero-helper scan of the whole work tree, all on the same final sources (the toolkit's rule).
Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources (never shown to the model).
Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, which replace cuBLAS variants) in the final sources against the variants the port must keep; fewer means variants were copied (outputs can still be correct).
