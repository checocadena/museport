# GPU test: llmc_trimat_forward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.26IjeXxwlB/task/trimat_forward_ref.cpp -o build/reference |
| native_build | `fail` | exit 1: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.26IjeXxwlB/task /tmp/tmp.26IjeXxwlB/task/trimat_forward_harness.cu -o build/native |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.26IjeXxwlB/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 14.25 ms, p95 14.87 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
