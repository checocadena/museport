# Tiers 2-3 run-2 ports on NVIDIA GPUs

- **Date:** 2026-10-27, afternoon UTC.
- **Machines:** Runpod; one pod per port. Each was created and deleted by `oneapi-agent remote-test --provider runpod --vendor nvidia --max-usd 0.5 --max-minutes 20`, run from the cloud session. Every log ends with `deleted runpod machine <id>: confirmed`.
- **GPUs:** whichever Runpod had in stock: RTX 2000 Ada Generation, RTX A4500, L4 and GeForce RTX 4090.
- **What each run does:** it builds the hidden C++ reference, llm.c's original CUDA harness (with `nvcc`) and Muse's SYCL port (as last compiled by Muse, unchanged). It checks the port against both the reference and the CUDA.
- **Which ports:** the 5 of the 8 correct ports that make no oneMath call. `museport gpu-tasks` skipped `softmax_forward` (it fails the hidden grade) and the three ports that call oneMath (`matmul_forward`, `matmul_backward`, `trimat_forward`), because oneMath is not built for the GPU hosts yet.

## Results: 5/5 pass against the reference and llm.c's CUDA

| Port | GPU | vs reference (worst max abs err) | vs llm.c's CUDA (worst max abs err) | atol |
|---|---|---|---|---|
| `t2-classifier-fused` | GeForce RTX 4090 | PASS (3.81e-06) | PASS (1.14e-05) | 0.0001 |
| `t2-global-norm` | RTX 2000 Ada Generation | PASS (2.86e-06) | PASS (2.5e-06) | 0.01 |
| `t2-layernorm-backward` | L4 | PASS (2.48e-05) | PASS (2.1e-05) | 0.001 |
| `t2-layernorm-forward` | RTX A4500 | PASS (9.54e-07) | PASS (3.58e-07) | 1e-05 |
| `t3-matmul-backward-bias` | RTX 2000 Ada Generation | PASS (5.72e-06) | PASS (1.14e-05) | 0.005 |

`layernorm_backward` passes, but it still copies one implementation across its kernel
variants (see the [README](../README.md)).

## Cost

About **$0.04** in total: the sum of the `est_cost_usd` the tool recorded for each run
($0.018, $0.003, $0.010, $0.004 and $0.004). Logs: [`logs/`](logs/).
