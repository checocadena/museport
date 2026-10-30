# oneMath: what Muse Port found, ready to send upstream

Muse's SYCL port of llm.c's bf16 trainer ([L1](../../evidence/l1-llmc-bf16/README.md)) found
two things in [uxlfoundation/oneMath](https://github.com/uxlfoundation/oneMath). This folder has
both written up for the author to send. **Nothing here has been sent.**

| | What | Send as | Files |
|---|---|---|---|
| 1 | The cuBLAS backend throws `unimplemented` for `gemm` with bf16 inputs and fp32 output, though the API declares it and the rocBLAS backend implements it | a pull request (bug fix) | [`onemath-cublas-bf16-gemm.patch`](onemath-cublas-bf16-gemm.patch), [`PR.md`](PR.md), [`unit-tests/`](unit-tests) |
| 2 | oneMath's API has no bf16 `gemm_batch` and no bf16-output `gemm`; it has both for fp16 | an issue (feature request) | [`ISSUE-bf16-gemm-forms.md`](ISSUE-bf16-gemm-forms.md) |

**Who wrote what:**

- **Muse:** the port, which found both gaps.
- **Claude, the AI assistant running this project:** the patch and both drafts. Say so when you
  send them; the drafts already do.

## The patch

[`onemath-cublas-bf16-gemm.patch`](onemath-cublas-bf16-gemm.patch) is against oneMath `develop`
at `585e610` (2026-10-23).

- **What it does:** it adds one `GEMM_EX_LAUNCHER` line for the buffer API and one
  `GEMM_EX_LAUNCHER_USM` line for USM, beside the existing fp16 lines, and removes the two
  functions that threw. The bf16 case then calls `cublasGemmEx` with `CUDA_R_16BF`, as the fp16
  case calls it with `CUDA_R_16F`.
- **Only column-major:** the row-major versions keep throwing, as every row-major level-3 call in
  the cuBLAS backend does.
- **What the evidence used:** this repo's GPU runs applied an earlier, equivalent version of the
  fix ([`patches/onemath-cublas-bf16-gemm.patch`](../../patches/onemath-cublas-bf16-gemm.patch),
  against oneMath `1308c5e`). It calls the same `gemm_ex` helper directly instead of through the
  macro.

**Tests:** oneMath already has the regression tests, `GemmTests.Bfloat16Bfloat16FloatPrecision`
and `GemmUsmTests.Bfloat16Bfloat16FloatPrecision`. On cuBLAS they report SKIPPED today, because
the call throws `unimplemented`. [`unit-tests/`](unit-tests) has oneMath's BLAS unit tests run on
an NVIDIA GPU before and after the patch, with logs.

## How to send them

1. **Check for duplicates.** Search oneMath's issues and pull requests for "bfloat16 cublas"
   and "bf16 gemm_batch"; this session could not search them. The git history of `develop` has
   one earlier bf16 commit, which is unrelated (#679, preprocessor macros).
2. **The pull request:** fork oneMath and branch from `develop`. Apply the patch with `git apply
   onemath-cublas-bf16-gemm.patch` and commit it under your name. Open the PR with the text in
   [`PR.md`](PR.md), attach the logs from [`unit-tests/`](unit-tests), and request the
   `@uxlfoundation/onemath-gpu-nvidia` and `@uxlfoundation/onemath-blas-write` teams.
3. **The issue:** open it with the "Request a feature" template, using
   [`ISSUE-bf16-gemm-forms.md`](ISSUE-bf16-gemm-forms.md). It asks the maintainers whether new
   interfaces need an RFC first; per oneMath's CONTRIBUTING.md, they do.

oneMath asks for no sign-off line; check CONTRIBUTING.md again when you send.
