# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `fail`: FAIL on NVIDIA GeForce RTX 4090 (cuda:0, nvidia): sycl_verify_vs_reference: candidate exited with code 1; sycl_verify_vs_native: candidate exited with code 1
- **Device:** NVIDIA GeForce RTX 4090 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.D4jDii8z6S/task /tmp/tmp.D4jDii8z6S/task/train_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0119, atol 0.1) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.D4jDii8z6S/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `fail` | candidate exited with code 1 |
| sycl_verify_vs_native | `fail` | candidate exited with code 1 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| native_benchmark | `ok` | 2 run(s) each; none: median 2210.50 ms, p95 2219.93 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
