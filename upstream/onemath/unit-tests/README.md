# oneMath BLAS unit tests on cuBLAS, before and after the patch

[`onemath-cublas-blas-tests.log`](onemath-cublas-blas-tests.log) is the log of
[`../gpu-task/run_tests.sh`](../gpu-task/run_tests.sh), run once on a rented machine.

## Setup

| | |
|---|---|
| GPU | NVIDIA L4, driver 595.91.07 (CUDA 13.2) |
| Compiler | intel/llvm DPC++ nightly 2026-10-30 (`clang++`) |
| oneMath | `develop` at `585e610`; `-DENABLE_CUBLAS_BACKEND=ON -DTARGET_DOMAINS=blas -DBUILD_FUNCTIONAL_TESTS=ON`, other backends off |
| Reference BLAS | Reference-LAPACK v3.12.1, CBLAS, 32- and 64-bit index |
| Devices | `ONEAPI_DEVICE_SELECTOR=cuda:*` |
| Time and cost | 38 min, $0.37 |

## Results: the whole BLAS suite, `--gtest_filter='*'`

| | Passed | Failed | Skipped |
|---|---|---|---|
| Unpatched, compile-time API (`test_main_blas_ct`) | 323 | 88 | 575 |
| Patched, compile-time API | **325** | 88 | 573 |
| Unpatched, run-time API (`test_main_blas_rt`) | 323 | 88 | 575 |
| Patched, run-time API | **325** | 88 | 573 |

**What the patch changes, in both APIs:**

- `GemmTests.Bfloat16Bfloat16FloatPrecision/Column_Major` goes from SKIPPED to **OK**.
- `GemmUsmTests.Bfloat16Bfloat16FloatPrecision/Column_Major` goes from SKIPPED to **OK**.
- Nothing else changes. The 88 failing tests are the same set before and after the patch, and the
  row-major bf16 cases stay SKIPPED.

## The 88 failures, which are not caused by the patch

- **Which tests:** all 88 are level-1 routines, such as `Scal`, `Copy`, `Swap`, `Nrm2`, `Asum`,
  `Axpy`, `Iamax` and `Dot`. Each fails within about 1 ms.
- **Not specific to level 1:** in a separate run filtered to `*Gemm*`, the gemm tests failed
  in the same way, including fp32 gemm and row-major cases that otherwise skip. In the
  full-suite run those same gemm tests pass.
- **Looks like start-up:** whatever runs first in a test process fails. That points to this
  machine's start-up (compiler nightly, driver, or cuBLAS initialisation), not to oneMath's
  routines.
- **Not investigated further:** the gemm-only sections of the log (`unpatched-gemm`,
  `patched-gemm`) show it. Their bf16 lines are failures of this kind, not results of the patch.
- **For the PR:** state this, and offer to rerun on another setup if the maintainers want a
  clean baseline.
