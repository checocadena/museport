# J3: `llm.c-sycl`'s own Makefile on NVIDIA and AMD GPUs

The step 4 GPU results ([test](../j3-test/README.md), [trainer](../j3-trainer/README.md)) came
from the toolkit's `gpu-test` build. This run builds `llm.c-sycl` with **its own Makefile** on
each GPU, and grades that build.

## How

`museport.trainer.make_j3_makefile_task` builds the J3 test's GPU task. Changes from step 4:

- **The repo is copied in:** `llm.c-sycl` as committed (its sources, `llmc/` and the Makefile)
  goes into the task as `llmc_sycl/`.
- **`make` runs in setup:** the task's `gpu_setup` builds oneMath for the vendor as before, then
  runs `make -C llmc_sycl SYCL_CXX=<the machine's SYCL compiler> TARGET=nvidia`, or
  `TARGET=amd AMD_ARCH=<detected arch>`. Its output is archived in `setup.log`.
- **The Makefile's binary is graded:** the candidate wrapper `run_make_test.py` swaps
  `gpu-test`'s own SYCL build for the test binary the Makefile built. llm.c's CUDA runs as
  itself. So the `sycl_*` rows in each `SUMMARY.md` grade the Makefile build.

The chain was first run locally on the CPU ($0). The Makefile build inside the task passed the
hidden check through the wrapper, with a worst error of 0.000154.

## Results

| GPU | `make` (from `setup.log`) | Makefile-built test vs PyTorch | vs llm.c's own CUDA test | Cost |
|---|---|---|---|---|
| NVIDIA RTX 2000 Ada (Runpod) | `TARGET=nvidia`: `-fsycl-targets=nvptx64-nvidia-cuda`, both programs built | **PASS**, `overall okay` 1, worst 0.000126 | **PASS**, worst 5.01e-06 | $0.014 |
| AMD Instinct MI350X (DigitalOcean) | `TARGET=amd AMD_ARCH=gfx950`: `--offload-arch=gfx950`, both programs built | **PASS**, `overall okay` 1, worst 0.0000236 | not run: llm.c's CUDA does not hipify into something that builds | $0.186 |

- **Results and setup logs:**
  - NVIDIA: [results](nvidia/nvidia-rtx-2000-ada-generation/llmc_test_gpt2_fp32_makefile/SUMMARY.md) and
    [setup.log](nvidia/nvidia-rtx-2000-ada-generation/llmc_test_gpt2_fp32_makefile/setup.log).
  - AMD: [results](amd/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/llmc_test_gpt2_fp32_makefile/SUMMARY.md)
    and [setup.log](amd/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/llmc_test_gpt2_fp32_makefile/setup.log).
  - Each `setup.log` ends with `museport: Makefile build ok`.
- **The remote-test logs** are in [`logs/`](logs).
- **The trainer** was built by the Makefile on both GPUs, but only the test was run.
- **These are different binaries from step 4's.** On NVIDIA, the Makefile's generic
  `nvptx64-nvidia-cuda` target differs from `gpu-test`'s `nvidia_gpu_sm_89`. The NVIDIA errors
  differ slightly from step 4's (0.000127 and 7.03e-06). On AMD the flags are the same, and so
  is the result.

Both machines were confirmed deleted. Total $0.20, which brings J3's GPU spend to $0.63 of its
$1 cap.
