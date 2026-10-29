# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `fail`: SYCL port did not compile for AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (hip:0, amd): compile failed with 2 error(s) at /tmp/tmp.fRVHsIcZ76/task/agent_port/llmc/batched_gemm.hpp:99: no matching function for call to 'get_native'
- **Device:** AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `fail` | exit 1: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.fRVHsIcZ76/task /tmp/tmp.fRVHsIcZ76/out/work/train_gpt2.hip.cpp -o build/native |
| sycl_compile | `fail` | compile failed with 2 error(s) at /tmp/tmp.fRVHsIcZ76/task/agent_port/llmc/batched_gemm.hpp:99: no matching function for call to 'get_native' |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
