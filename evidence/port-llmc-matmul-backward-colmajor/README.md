# matmul_backward re-ported with column-major oneMath calls

Run 2's port of `matmul_backward` ([`../port-llmc-tier23-run2/t3-matmul-backward`](../port-llmc-tier23-run2/t3-matmul-backward))
passed its hidden check on the CPU device and on an AMD MI350X. On an NVIDIA GPU it aborted:
it calls `oneapi::math::blas::row_major::gemm`, and oneMath's cuBLAS backend does not implement
row-major gemm ([D2](../d2-onemath-gpu/README.md)).

The port goal's oneMath line now asks for `oneapi::math::blas::column_major`, and says why. With
that goal, the same task and the same hidden check, Muse ported the file again
(`museport port ... --items t3-matmul-backward --max-iterations 16 --max-cost 0.70`, as in
run 2; [`run.log`](run.log)).

| | Run 2 | This run |
|---|---|---|
| Loop | success in 8 turns | success in 7 turns |
| Muse cost | $0.11 | $0.094 |
| Hidden grade (CPU) | PASS, worst 0.00021, 0 helpers | PASS, worst 0.00021, 0 helpers |
| oneMath calls | 2 × `row_major::gemm` | 2 × `column_major::gemm` |
| NVIDIA | aborted (RTX A4500) | PASS vs reference (worst 0.000198) and vs llm.c's CUDA (identical), RTX 2000 Ada |

GPU result: [`gpu/nvidia`](gpu/nvidia). This port is the `matmul_backward` that
`llm.c-sycl/dev/sycl` uses. The README's headline table still counts run 2's port, as it does
for every item.
