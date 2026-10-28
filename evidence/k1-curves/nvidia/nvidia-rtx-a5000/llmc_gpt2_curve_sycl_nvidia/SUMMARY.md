# GPU test: llmc_gpt2_curve_sycl_nvidia

- **Result:** `ok`: PASS on NVIDIA RTX A5000 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.00422, atol 0.01); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.00397, atol 0.01)
- **Device:** NVIDIA RTX A5000 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.i3zht750xj/task /tmp/tmp.i3zht750xj/task/train_gpt2_fp32.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00141, atol 0.01) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.i3zht750xj/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00422, atol 0.01) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00397, atol 0.01) |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
