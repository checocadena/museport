# GPU test: llmc_train_gpt2_fp32

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.0023, atol 0.01); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.00147, atol 0.01)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.5BCwjVYN6p/task /tmp/tmp.5BCwjVYN6p/task/train_gpt2_fp32.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00386, atol 0.01) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.5BCwjVYN6p/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0023, atol 0.01) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00147, atol 0.01) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 1708.89 ms, p95 1766.17 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 1602.18 ms, p95 1653.35 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
