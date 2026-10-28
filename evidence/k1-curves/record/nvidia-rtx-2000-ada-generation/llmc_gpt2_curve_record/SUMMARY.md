# GPU test: llmc_gpt2_curve_record

- **Result:** `ok`: recorded 1 output(s) of the original CUDA on NVIDIA RTX 2000 Ada Generation (nvidia); two runs differ: FAIL: 0/1 case(s) match; first failure: 308 of 311 element(s) outside tolerance
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `(JIT)`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.I4LCXzuPTQ/task /tmp/tmp.I4LCXzuPTQ/task/train_gpt2_fp32.cu -lcublas -o build/native |
| native_repeat | `fail` | FAIL: 0/1 case(s) match; first failure: 308 of 311 element(s) outside tolerance |

Limitations:
- A recorded reference is the original program's own output on one GPU: it is only as right as that program, and a non-deterministic program needs a tolerance at least as wide as its run-to-run difference.
