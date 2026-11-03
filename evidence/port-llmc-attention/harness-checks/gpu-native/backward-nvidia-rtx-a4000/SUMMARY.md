# GPU test: llmc_attention_backward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX A4000 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.001)
- **Device:** NVIDIA RTX A4000 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.osChO8EpHC/task/attention_backward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.osChO8EpHC/task /tmp/tmp.osChO8EpHC/task/attention_backward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.osChO8EpHC/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 543.03 ms, p95 591.56 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 800.94 ms, p95 925.53 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
