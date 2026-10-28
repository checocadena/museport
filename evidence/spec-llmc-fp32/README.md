# Harness specs: rerun with the fp32 build stated in the prompt

The first J2 run ([`../spec-llmc`](../spec-llmc/README.md)) proved 16 of 17 specs, but three of them
were looser than llm.c's own check. llm.c's mains pick the tolerance with
`#if defined(ENABLE_BF16) ... #else` (or `std::is_same_v<floatX, float>`), and Muse took the bf16
side. In `matmul_backward_bias` it also dropped variant 5, which exists only in the fp32 build.

**The change:** one paragraph added to the prompt (`src/museport/spec.py`, `SYSTEM`). It states a
fact about museport's build, nothing about any file:

> The build is fp32: ENABLE_BF16 is not defined and floatX is float. Wherever the file or its main
> branches on it (...), take the fp32 side: the fp32 tolerance, and keep the variants under
> `#ifndef ENABLE_BF16` (they exist only in the fp32 build).

**The rerun (Oct 28):** the three loose files and `trimat_forward`, with the GPU proof on.

## Results ([SUMMARY.md](SUMMARY.md), [summary.json](summary.json))

| File | First prompt | fp32 prompt |
|---|---|---|
| `gelu_backward` | proven, tol 1e-2 (llm.c fp32: 1e-5) | **proven, tol 1e-5**: CPU proof, first attempt, $0.0229 |
| `residual_forward` | proven, tol 1e-2 (1e-5) | **proven, tol 1e-5**: CPU proof, first attempt, $0.0240 |
| `matmul_backward_bias` | proven, tol 1.0 (5e-3), variant 5 dropped | **proven, tol 5e-3, all 8 fp32 variants**: GPU proof, first attempt, $0.0553 + GPU $0.003 |
| `trimat_forward` | not proven | **not proven**, $0.1992 + GPU $0.004 |

`trimat_forward` failed the same way as before:
- **Attempt 1:** the spec does not zero the causal-masked triangle. llm.c's CPU code writes NaN
  there and its `validate_result` skips those values. The GPU proof found exactly the same
  4,112,984 of 7,864,320 values out of tolerance as in the first run.
- **Attempts 2 and 3:** empty answers (`finish_reason=stop`, no content, no usage). Each is
  counted at its worst case ($0.084).

Across three runs of this file, each first spec made the masked-triangle mistake. After that
feedback, four answers were empty and one spec's reference did not compile. More output budget,
or a hint about masked outputs, would be the next things to try.

## All 17 files, taking each file's latest run

| | Files |
|---|---|
| Spec proven | **16 of 17** |
| Proven and as strict as llm.c's own check (its fp32 tolerance, every fp32 variant) | **16 of 16** |
| Not proven | `trimat_forward` |

- The other 13 proven specs are from the first run; the prompt change does not touch them.
- `softmax_forward` and `classifier_fused` were proven by a GPU proof of Muse's first spec after
  that run (see [`../spec-llmc`](../spec-llmc/README.md)).

## Cost

- **Muse:** $0.3014. The two empty answers are counted at their worst case.
- **GPU:** $0.007.
