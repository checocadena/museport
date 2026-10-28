# D2: the kernel ports that call oneMath, on NVIDIA and AMD

Three of llm.c's Tier 3 kernel files call cuBLAS, so Muse's ports call oneMath. They are
`matmul_forward`, `matmul_backward` and `trimat_forward`. All three passed their hidden checks
on the CPU device, where oneMath uses MKL ([run 2](../port-llmc-tier23-run2/README.md)). Until
now they were never run on a GPU: `museport gpu-tasks` skipped them, because the GPU machines had
no oneMath.

`museport gpu-tasks --vendor nvidia|amd` now gives these tasks the same `gpu_setup` as the GPT-2
and K1 runs. It builds oneMath on the machine: with cuBLAS on NVIDIA and with rocBLAS on AMD.

## Results

| Port | NVIDIA vs hidden reference | NVIDIA vs llm.c's CUDA | AMD MI350X vs hidden reference |
|---|---|---|---|
| `matmul_forward` | PASS, worst 6.29e-05 (RTX 2000 Ada) | PASS, worst 0.0122 (atol 0.1) | PASS, worst 3.43e-05 |
| `trimat_forward` | PASS, worst 5.96e-07 (RTX A4500) | PASS, worst 0 | PASS, worst 5.96e-07 |
| `matmul_backward`, run 2's port | **aborted** (RTX A4500) | aborted | PASS, worst 0.000214 |
| `matmul_backward`, re-ported | PASS, worst 0.000198 (RTX 2000 Ada) | PASS, worst 0 | (not rerun) |

- **AMD:** llm.c's CUDA does not hipify into something that builds, so on AMD the ports are
  checked against the hidden reference only, as everywhere else in this project.
- **Results:** NVIDIA in [`nvidia/`](nvidia), AMD in [`amd/`](amd), and the re-ported
  `matmul_backward` on NVIDIA in
  [`../port-llmc-matmul-backward-colmajor/gpu/nvidia`](../port-llmc-matmul-backward-colmajor/gpu/nvidia).

## matmul_backward: a CPU pass that a GPU caught

Run 2's port of `matmul_backward` called `oneapi::math::blas::row_major::gemm`.
- **CPU and AMD:** oneMath's MKL backend implements row-major gemm, and so does its rocBLAS
  backend. So the port passed on the CPU and on the AMD GPU.
- **NVIDIA:** its cuBLAS backend does not, and the program aborted before writing any output:

```
terminate called after throwing an instance of 'oneapi::math::unimplemented'
  what():  oneMath: blas/gemm: function is not implemented for row_major layout
```

The other two ports, and the J3 GPT-2 trainer, use `column_major`, which is cuBLAS's own layout.

**The fix:**
- **Prompt:** the oneMath part of the port goal now tells Muse to use
  `oneapi::math::blas::column_major` (swapping operands for row-major data), and says why.
- **Rerun:** Muse re-ported `matmul_backward` with the same task and hidden check
  ([evidence](../port-llmc-matmul-backward-colmajor/SUMMARY.md)).
  - **Result:** success in 7 turns for $0.094. Worst error 0.00021 on the CPU, 0 helpers.
  - **API:** two `column_major::gemm` calls, and no `row_major`.
  - **On NVIDIA:** it now passes against the reference, and matches llm.c's CUDA exactly.

`llm.c-sycl/dev/sycl` still holds run 2's `matmul_backward`, as it was graded. Switching it to the
re-ported version is a separate change to `museport llmc-sycl assemble`.

## Cost

| | Runs | Cost |
|---|---|---|
| NVIDIA, Runpod | 3 ports + the re-ported `matmul_backward` | $0.028 |
| AMD, DigitalOcean MI350X | 3 ports, oneMath built each time | $0.425 |
| Muse (the rerun) | 1 | $0.094 |

All machines were confirmed deleted ([`logs/`](logs)).
