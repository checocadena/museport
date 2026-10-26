# GPU test: llmc_crossentropy_softmax_backward

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.c7SkWYkhjq/task/crossentropy_softmax_backward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.c7SkWYkhjq/task /tmp/tmp.c7SkWYkhjq/task/crossentropy_softmax_backward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.c7SkWYkhjq/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 349.13 ms, p95 364.27 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 321.36 ms, p95 351.66 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
