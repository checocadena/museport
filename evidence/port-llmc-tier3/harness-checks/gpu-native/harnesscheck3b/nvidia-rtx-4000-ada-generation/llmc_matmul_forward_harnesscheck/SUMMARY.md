# GPU test: llmc_matmul_forward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX 4000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.1); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.0121, atol 0.1)
- **Device:** NVIDIA RTX 4000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.cM12bid1fL/task/matmul_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.cM12bid1fL/task /tmp/tmp.cM12bid1fL/task/matmul_forward_harness.cu -lcublas -lcublasLt -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0121, atol 0.1) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.cM12bid1fL/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.1) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0121, atol 0.1) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 555.93 ms, p95 558.06 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 519.41 ms, p95 578.14 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
