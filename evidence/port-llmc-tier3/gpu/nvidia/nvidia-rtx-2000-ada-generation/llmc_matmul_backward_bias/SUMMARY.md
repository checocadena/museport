# GPU test: llmc_matmul_backward_bias

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 7.75e-06, atol 0.005); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.005)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.aWxZTmwrat/task/matmul_backward_bias_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.aWxZTmwrat/task /tmp/tmp.aWxZTmwrat/task/matmul_backward_bias_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.005) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.aWxZTmwrat/out/work/build/sycl (24 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.75e-06, atol 0.005) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.005) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 321.25 ms, p95 341.79 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 321.22 ms, p95 340.80 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
