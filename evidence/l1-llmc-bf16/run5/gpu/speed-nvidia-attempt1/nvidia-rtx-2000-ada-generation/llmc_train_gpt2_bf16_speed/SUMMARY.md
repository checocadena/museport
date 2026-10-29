# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `fail`: SYCL port did not compile for NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): compile failed with 3 error(s) at train_gpt2.cpp:None: undefined reference to `nvmlInit_v2'
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.pF3nVUC8ef/task /tmp/tmp.pF3nVUC8ef/task/train_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0131, atol 0.1) |
| sycl_compile | `fail` | compile failed with 3 error(s) at train_gpt2.cpp:None: undefined reference to `nvmlInit_v2' |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
