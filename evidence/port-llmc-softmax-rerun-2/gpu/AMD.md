# softmax_forward (rerun 2) on an AMD GPU

- **Date:** 2026-10-28, early UTC.
- **Device:** AMD Instinct MI350X VF, gfx950 (`hip:0`), on a DigitalOcean `gpu-mi350x1-288gb-spot` droplet ($3.00/h). It was created and deleted by `oneapi-agent remote-test --provider digitalocean --vendor amd --public-tls --max-usd 1.0`, run from the cloud session. This was the first DigitalOcean run made from the cloud.
- **Check:** Muse's port, unchanged, against the hidden C++ reference. llm.c's CUDA does not hipify into something that builds, so there is no comparison with the CUDA on AMD.

| Port | vs reference (worst max abs err) | atol | minutes | est_cost_usd |
|---|---|---|---|---|
| `t2-softmax-forward` | PASS (2.86e-05) | 0.0001 | 2.9 | 0.143 |

Muse's code review ([review-muse](../../review-muse/README.md)) found divergent `group_barrier`s in
variants 6 and 8 of this port, at sizes the hidden check does not use. This pass does not cover
them. Log: [`logs/t2-softmax-forward-amd.log`](logs/t2-softmax-forward-amd.log).
