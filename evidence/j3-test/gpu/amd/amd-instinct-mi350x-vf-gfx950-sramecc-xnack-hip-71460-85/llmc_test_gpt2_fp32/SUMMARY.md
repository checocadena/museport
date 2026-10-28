# GPU test: llmc_test_gpt2_fp32

- **Result:** `ok`: PASS on AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (hip:0, amd): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 2.36e-05, atol 0.01)
- **Device:** AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `fail` | exit 1: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.82tChDCHUO/task /tmp/tmp.82tChDCHUO/out/work/test_gpt2_fp32.hip.cpp -o build/native |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.82tChDCHUO/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.36e-05, atol 0.01) |
| sycl_benchmark | `ok` | 10 run(s) each; hip:0: median 999.81 ms, p95 1014.69 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
