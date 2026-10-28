# GPT-2 stage 3 (training) on an AMD GPU

This is the same Muse SYCL port of llm.c's GPT-2 training as on [NVIDIA](NVIDIA.md)
([`../run1/port/gpt2_train.cpp`](../run1/port/gpt2_train.cpp), unchanged, as Muse last compiled
it). Here it ran on an AMD GPU, with its matmuls going through oneMath to **rocBLAS**.

- **Date:** 2026-10-28, about 03:15 UTC.
- **Machine:** a DigitalOcean GPU droplet (`gpu-mi350x1-288gb-spot`, $3.00/h) with an AMD Instinct
  MI350X VF, gfx950 (SYCL selector `hip:0`). It was created and deleted by
  `oneapi-agent remote-test --provider digitalocean --vendor amd --public-tls --max-usd 1.5 --max-minutes 30`,
  run from the cloud session. The log ends with `deleted digitalocean machine <id>: confirmed`.
- **Task:** written by `museport gpt2-gpu-task evidence/gpt2-stage3 --vendor amd`. As on NVIDIA,
  the machine fetches llm.c's weights and checks their sha256. It builds oneMath (the same commit
  as the CPU and NVIDIA builds) with its rocBLAS backend for the GPU's gfx target, which is
  detected on the machine. [`setup.log`](amd/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/llmc_gpt2_train/setup.log)
  shows `libonemath_blas_rocblas.so` installed.

## Result: PASS against PyTorch

| Check | Result (10 losses, atol 1e-2) |
|---|---|
| Muse's SYCL port vs PyTorch (hidden reference) | **PASS**, worst max abs err 0.0000246 |
| Muse's SYCL port vs llm.c's CUDA | not run: llm.c's CUDA does not hipify into something that builds (`native_build` fails, as for every llm.c file on AMD) |

The same source, unchanged, now passes on three targets: the CPU (oneMath with MKL), an NVIDIA
RTX 4000 Ada (cuBLAS) and an AMD MI350X (rocBLAS).

Time: 10 runs of the whole program (10 training steps, including loading the 498 MB of weights)
took a median of 732 ms. This is not a kernel speed measurement.

## Getting there: five runs, $0.65

DigitalOcean's MI350X image (ROCm 7.14 prerelease) lacks the development files oneMath's rocBLAS
backend needs. Each run got one step further, and the fixes are in `museport/gpt2.py`
(`ROCBLAS_UNPACK`):

| Run | What stopped it | Fix | est_cost_usd |
|---|---|---|---|
| 1 | no `cmake` (and no `pip`) on the image | install `cmake` and `ninja` with apt when pip is missing | 0.117 |
| 2 | no rocBLAS CMake config | unpack AMD's `amdrocm-blas-dev`, `-host` and `-gfx950` packages into the ROCm tree, at the image runtime's version (the toolkit unpacks the HIP headers the same way) | 0.120 |
| 3 | empty results: the setup script, which the toolkit's runner sources, set `D` and then deleted `$D`, the runner's own folder | prefix the setup's variables with `MP_`; a test checks that they never use the runner's names | 0.126 |
| 4 | `hip-config` needs `AMDDeviceLibs`' CMake config | extract just `lib/llvm/lib/cmake` from `amdrocm-llvm-dev` | 0.128 |
| 5 | **PASS** | | 0.158 |

All five logs are in [`logs/`](logs/) (`gpt2-stage3-amd-attempt*.log`).
