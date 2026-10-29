# GPU test: llmc_test_gpt2_bf16

- **Result:** `fail`: FAIL on AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (hip:0, amd): sycl_verify_vs_reference: FAIL: 0/1 case(s) match; first failure: 1 of 12 element(s) outside tolerance
- **Device:** AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `fail` | exit 1: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.7vy243iDbm/task /tmp/tmp.7vy243iDbm/out/work/test_gpt2.hip.cpp -o build/native |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.7vy243iDbm/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `fail` | FAIL: 0/1 case(s) match; first failure: 1 of 12 element(s) outside tolerance |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
