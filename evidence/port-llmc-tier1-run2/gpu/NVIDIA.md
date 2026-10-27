# Tier 1 run-2 ports on NVIDIA GPUs

- **Date:** 2026-10-27, 03:31-03:42 UTC.
- **Machines:** Runpod; one pod per port. Each was created and deleted by `oneapi-agent remote-test --provider runpod --vendor nvidia --max-usd 0.5`, run from the cloud session. Every log ends with `deleted runpod machine <id>: confirmed`. The Runpod API listed no pods afterwards.
- **GPUs:** NVIDIA RTX 2000 Ada Generation ($0.24/h) for the first 4 ports and RTX A4500 ($0.25/h) for the other 6, whichever Runpod had in stock.
- **What each run does:** it builds the hidden C++ reference, llm.c's original CUDA harness (with `nvcc`) and Muse's SYCL port (as last compiled by Muse, unchanged). It checks the port against both the reference and the CUDA.

## Results: 10/10 pass against the reference and llm.c's CUDA

| Port | GPU | vs reference (worst max abs err) | vs llm.c's CUDA (worst max abs err) | atol |
|---|---|---|---|---|
| `t1-adamw` | RTX 2000 Ada Generation | PASS (5.96e-08) | PASS (5.96e-08) | 0.0001 |
| `t1-crossentropy-forward` | RTX 2000 Ada Generation | PASS (1.19e-07) | PASS (0) | 1e-05 |
| `t1-crossentropy-softmax-backward` | RTX 2000 Ada Generation | PASS (0) | PASS (0) | 1e-05 |
| `t1-encoder-backward` | RTX 2000 Ada Generation | PASS (0) | PASS (0) | 1e-05 |
| `t1-encoder-forward` | RTX A4500 | PASS (0) | PASS (0) | 1e-05 |
| `t1-fused-residual-forward` | RTX A4500 | PASS (1.43e-06) | PASS (1.43e-06) | 1e-05 |
| `t1-gelu-backward` | RTX A4500 | PASS (2.38e-07) | PASS (1.79e-07) | 1e-05 |
| `t1-gelu-forward` | RTX A4500 | PASS (1.19e-07) | PASS (1.19e-07) | 1e-05 |
| `t1-permute` | RTX A4500 | PASS (0) | PASS (0) | 1e-05 |
| `t1-residual-forward` | RTX A4500 | PASS (0) | PASS (0) | 1e-05 |

## Cost

The tool does not record a cost for Runpod runs. The 10 runs took about 10.5 minutes in all,
at $0.24-0.25 an hour. That is about $0.05 in total (estimate from the run logs' times and the
hourly prices). Logs: [`logs/`](logs/).
