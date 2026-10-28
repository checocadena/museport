# Tiers 2-3 run-2 ports on an AMD GPU

- **Date:** 2026-10-28, early UTC.
- **Device:** AMD Instinct MI350X VF, gfx950 (`AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85]`, SYCL selector `hip:0`).
- **Machine:** a DigitalOcean GPU droplet (`gpu-mi350x1-288gb-spot`, $3.00/h), one droplet per port. Each was created and deleted by `oneapi-agent remote-test --provider digitalocean --vendor amd --public-tls --max-usd 1.0 --max-minutes 30`, **run from the cloud session**; that became possible with `--public-tls` ([oneapi-agent-toolkit#25](https://github.com/checocadena/oneapi-agent-toolkit/pull/25)). Every log ends with `deleted digitalocean machine <id>: confirmed`.
- **Toolkit:** oneapi-agent-toolkit main at `bb12d1d` (#25 merged), or the `record-native` branch on top of it, whose only change applies to recording tasks.
- **SYCL compiler:** `sycl-hip-nightly-2026-10-14-rocm7.0-r3` (sha256 `e9bf82ee…c8fc3`).
- **Which ports:** the same 5 as on [NVIDIA](NVIDIA.md): the correct ports that make no oneMath call.
- **Check:** each SYCL port, as Muse last compiled it, unchanged, against the hidden C++ reference. llm.c's CUDA does not hipify into something that builds (`native_build` fails, as on Tier 1), so there is no comparison with the CUDA on AMD.

## Results: 5/5 pass against the reference

| Port | vs reference (worst max abs err) | atol | sycl_compile | minutes | est_cost_usd |
|---|---|---|---|---|---|
| `t2-classifier-fused` | PASS (3.81e-06) | 0.0001 | ok (0 warnings) | 2.4 | 0.119 |
| `t2-global-norm` | PASS (5.96e-07) | 0.01 | ok (0 warnings) | 2.2 | 0.110 |
| `t2-layernorm-backward` | PASS (1.53e-05) | 0.001 | ok (0 warnings) | 2.3 | 0.114 |
| `t2-layernorm-forward` | PASS (9.54e-07) | 1e-05 | ok (30 warnings) | 2.4 | 0.119 |
| `t3-matmul-backward-bias` | PASS (6.68e-06) | 0.005 | ok (0 warnings) | 2.3 | 0.116 |

Total: **$0.578**, the sum of the `est_cost_usd` the tool recorded. Logs are in
[`logs/`](logs/) (`*-amd.log`); results are in
[`amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/`](amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/).
The tool also ran a benchmark, a rough single-process wall time; it is in each `SUMMARY.md`.
