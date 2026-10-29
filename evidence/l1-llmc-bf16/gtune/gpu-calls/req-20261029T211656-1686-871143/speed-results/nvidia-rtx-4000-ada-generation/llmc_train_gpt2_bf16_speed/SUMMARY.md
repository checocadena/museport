# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `ok`: PASS on NVIDIA RTX 4000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.02, atol 0.1); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0.0179, atol 0.1)
- **Device:** NVIDIA RTX 4000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.1i0vdfuOqc/task /tmp/tmp.1i0vdfuOqc/task/train_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00924, atol 0.1) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.1i0vdfuOqc/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.02, atol 0.1) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0179, atol 0.1) |
| sycl_benchmark | `ok` | 1 run(s) each; cuda:0: median 5334.61 ms, p95 5334.61 ms |
| native_benchmark | `ok` | 1 run(s) each; none: median 4200.80 ms, p95 4200.80 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
