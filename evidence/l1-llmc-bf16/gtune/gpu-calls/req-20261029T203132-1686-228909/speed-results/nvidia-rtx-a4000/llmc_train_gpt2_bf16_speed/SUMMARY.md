# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `ok`: PASS on NVIDIA RTX A4000 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.0192, atol 0.1); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.0243, atol 0.1)
- **Device:** NVIDIA RTX A4000 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.SoJGQh8HzP/task /tmp/tmp.SoJGQh8HzP/task/train_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0151, atol 0.1) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.SoJGQh8HzP/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0192, atol 0.1) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0243, atol 0.1) |
| sycl_benchmark | `ok` | 2 run(s) each; cuda:0: median 9111.89 ms, p95 9163.24 ms |
| native_benchmark | `ok` | 2 run(s) each; none: median 4414.29 ms, p95 4436.16 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
