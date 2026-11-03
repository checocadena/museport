# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `ok`: PASS on NVIDIA H100 80GB HBM3 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.0156, atol 0.1); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.0176, atol 0.1)
- **Device:** NVIDIA H100 80GB HBM3 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_90`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.Qwc6MxiK6s/task /tmp/tmp.Qwc6MxiK6s/task/train_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0115, atol 0.1) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.Qwc6MxiK6s/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0156, atol 0.1) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0176, atol 0.1) |
| sycl_benchmark | `ok` | 3 run(s) each; cuda:0: median 2333.39 ms, p95 2353.00 ms |
| native_benchmark | `ok` | 3 run(s) each; none: median 1353.98 ms, p95 1364.03 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
