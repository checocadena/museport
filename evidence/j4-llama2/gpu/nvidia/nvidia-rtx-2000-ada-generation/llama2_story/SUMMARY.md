# GPU test: llama2_story

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 2/2 case(s) match (worst max abs err 0, atol 0); sycl_verify_vs_native: PASS: 2/2 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.2wblkj0G8O/task /tmp/tmp.2wblkj0G8O/task/llama2.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 2/2 case(s) match (worst max abs err 0, atol 0) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.2wblkj0G8O/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 2/2 case(s) match (worst max abs err 0, atol 0) |
| sycl_verify_vs_native | `ok` | PASS: 2/2 case(s) match (worst max abs err 0, atol 0) |
| sycl_benchmark | `ok` | 3 run(s) each; cuda:0: median 430.38 ms, p95 433.32 ms |
| native_benchmark | `ok` | 3 run(s) each; none: median 378.20 ms, p95 390.55 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
