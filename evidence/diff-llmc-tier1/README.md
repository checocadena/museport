# Differential checks on Tier 1 (plan item J1)

Until now each llm.c kernel's hidden reference was a C++ program written for it from llm.c's
CPU code. A differential check needs no such program. The hidden reference is **the original
CUDA's own output**, recorded on an NVIDIA GPU. This lets a port be checked when nobody wrote a
CPU version.

## What was done (Oct 28)

1. `museport record-tasks evidence/port-llmc-tier1-run2 --out <dir>` wrote one recording task
   per Tier 1 kernel. A recording task builds llm.c's CUDA harness with `nvcc` and runs it, with
   no port involved; this is the toolkit's `record_native`, from
   [oneapi-agent-toolkit#26](https://github.com/checocadena/oneapi-agent-toolkit/pull/26).
2. `oneapi-agent remote-test <task> --provider runpod --vendor nvidia` ran each one on a rented
   NVIDIA GPU (an RTX A4500 or an RTX 2000 Ada). Each program ran twice, and the two runs were
   **bit-identical for all 10 kernels**. Logs are in `logs/`, results in `recordings/`.
   **Cost: $0.034 for all 10.**
3. `museport diff-check evidence/port-llmc-tier1-run2 <recordings> --out evidence/diff-llmc-tier1`
   ran on a CPU, with no model and no GPU, at $0. For each kernel it:
   - built the differential task: the recorded outputs sit in the task directory, which the
     model's tools cannot read, and are served by `golden_ref.py`;
   - checked the recording against the old CPU reference, with the same inputs and tolerance;
   - graded Muse's port from that run against the recording, using the same steps as the hidden
     grade (compile with the model's last arguments, verify, scan for helpers).

## Results ([SUMMARY.md](SUMMARY.md), [summary.json](summary.json))

| | result |
|---|---|
| recordings agree with llm.c's CPU reference | **10 / 10** (worst error 1.43e-06) |
| Muse's ports pass against the recorded CUDA output | **10 / 10**, the same verdicts as against the CPU reference |
| planted bug, dropped digit (`0.044715f` → `0.04715f` in gelu_forward) | **fails** both references (1,062,644 of 1,572,864 values out of tolerance) |
| planted bug, subtle (`0.044715f` → `0.0447f`) | **passes** both references (worst error 3.2e-06 against atol 1e-05) |

The subtle typo passing shows the limit of any tolerance-based check, a differential one
included: an error smaller than the tolerance is invalid code that still passes. The planted
sources and their grades are in [planted/](planted/).

## What is not committed

The recorded outputs themselves are about 83 MB. Each one's sha256 is in `summary.json` and in
`recordings/*/summary.json`, and re-recording costs about $0.03. `museport diff-task` rebuilds a
differential task from a recording directory.

## Limits

- A recording is only as right as the original program on one GPU. Here every run was
  deterministic and every recording matched an independent CPU reference, so these 10 references
  are as trustworthy as the old ones. That second check needs a CPU reference, and a project
  without one gets only the first.
- The ports checked here come from a run graded against the CPU reference. No Muse run has yet
  been graded against a recording from the start.
- The harness that feeds the CUDA and the port the same inputs is still written per file.
  Generating it is plan item J2.
