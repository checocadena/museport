# GPU test: llmc_matmul_forward

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 6.29e-05, atol 0.1); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.0122, atol 0.1)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.zz9hzkUhrT/task/matmul_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.zz9hzkUhrT/task /tmp/tmp.zz9hzkUhrT/task/matmul_forward_harness.cu -lcublas -lcublasLt -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0121, atol 0.1) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.zz9hzkUhrT/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 6.29e-05, atol 0.1) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0122, atol 0.1) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 421.66 ms, p95 440.49 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 390.83 ms, p95 440.66 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
