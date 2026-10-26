# GPU test: llmc_matmul_forward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.1)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.EApzIu2PtR/task/matmul_forward_ref.cpp -o build/reference |
| native_build | `fail` | exit 1: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.EApzIu2PtR/task /tmp/tmp.EApzIu2PtR/task/matmul_forward_harness.cu -o build/native |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.EApzIu2PtR/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.1) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 460.55 ms, p95 464.67 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
