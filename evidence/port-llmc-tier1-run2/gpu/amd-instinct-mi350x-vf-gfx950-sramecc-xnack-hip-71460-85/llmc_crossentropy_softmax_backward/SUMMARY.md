# GPU test: llmc_crossentropy_softmax_backward

- **Result:** `ok`: PASS on AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (hip:0, amd): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05)
- **Device:** AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.cRgyawOZw7/task/crossentropy_softmax_backward_ref.cpp -o build/reference |
| native_build | `fail` | exit 1: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.cRgyawOZw7/task /tmp/tmp.cRgyawOZw7/out/work/crossentropy_softmax_backward_harness.hip.cpp -o build/native |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.cRgyawOZw7/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; hip:0: median 251.52 ms, p95 255.15 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
