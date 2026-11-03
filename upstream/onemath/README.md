# oneMath: what Muse Port found, and what was sent upstream

Muse's SYCL port of llm.c's bf16 trainer ([L1](../../evidence/l1-llmc-bf16/README.md)) found
two things in [uxlfoundation/oneMath](https://github.com/uxlfoundation/oneMath). This folder has
both written up.

**Status (Nov 3, 2026):**

- **The bug fix is submitted** as [uxlfoundation/oneMath#793](https://github.com/uxlfoundation/oneMath/pull/793), from
  the author's fork (`checocadena/oneMath`, branch `cublas-bf16-gemm`, commit `727d516`), on top
  of `develop` at `c430a94`.
  - oneMath's `pre-commit` (clang-format 19.1.0) passes.
  - GitHub reports it mergeable.
- **The feature request was not filed.** Before sending, we found two open upstream PRs that add
  exactly these forms: [#785](https://github.com/uxlfoundation/oneMath/pull/785) (bf16
  `gemm_batch`) and [#770](https://github.com/uxlfoundation/oneMath/pull/770) (bf16-output
  `gemm`). Neither touches the bf16 → fp32 `gemm` that #793 fixes; #793's description explains
  how they relate.

| | What | Send as | Files |
|---|---|---|---|
| 1 | The cuBLAS backend throws `unimplemented` for `gemm` with bf16 inputs and fp32 output, though the API declares it and the rocBLAS backend implements it | a pull request (bug fix) | [`onemath-cublas-bf16-gemm.patch`](onemath-cublas-bf16-gemm.patch), [`PR.md`](PR.md), [`unit-tests/`](unit-tests) |
| 2 | oneMath's released API has no bf16 `gemm_batch` and no bf16-output `gemm`; it has both for fp16 | not filed: already in review upstream (#785, #770) | [`ISSUE-bf16-gemm-forms.md`](ISSUE-bf16-gemm-forms.md) |

**Who wrote what:**

- **Muse:** the port, which found both gaps.
- **Claude, the AI assistant running this project:** the patch and both drafts. #793 says so in
  its description, and its commit has a `Co-Authored-By` line.

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

## How it was sent

1. **Duplicates checked:** we searched oneMath's PRs and found #785 and #770 (above).
2. **Branch:** the patch was applied to a branch of the author's fork, on today's `develop`, and
   checked with oneMath's `pre-commit`. The commit carries a `Co-Authored-By` line for Claude.
3. **The PR** was opened with the text in [`PR.md`](PR.md), plus a "Related PRs" note. It links
   the test log in [`unit-tests/`](unit-tests) and the script that produced it.
