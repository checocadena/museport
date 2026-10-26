# Build plan (Oct 27 – Nov 4, 2026)

Track: **Indie Developer**. Problem statement: **Push the Frontier**. Submission due Wed Nov 4, 9:00 pm ET (target:
midday). Demo video: 3 minutes, new work only.

## The product

1. **`museport plan <repo>`**: Muse reads a CUDA repository and writes a migration plan:
   every kernel and `.cu` file, NVIDIA library calls mapped to oneAPI (cuBLAS → oneMath
   BLAS, Thrust → oneDPL, cuFFT → oneMath DFT, cuRAND → oneMath RNG), CUDA-only
   assumptions (warp size 32, bf16, inline PTX, cooperative groups), and an ordered work
   list with a check for each item. The plan is grounded: its claims are cross-checked
   against what SYCLomatic and `scan_helpers` actually report.
2. **`museport port <plan>`**: for each work item, the toolkit's agent loop, driven by Muse,
   ports it; success only when the hidden check passes and no migration helpers remain.
3. **`museport report`**: a project-level report and replay pages: what was ported and
   proven, what was not, cost, and GPU results (NVIDIA and AMD).

Target: llm.c's `dev/cuda` kernels (each ships with its own CPU reference check).
Backup: thundersvm.

## Days

| Day | Goal |
|---|---|
| Oct 27–28 | Meta API smoke test; Muse runs through the existing comparison (supporting evidence); repository scanner |
| Oct 29–30 | `plan`: Muse planning over a real repo, plan schema, grounding against c2s/scan_helpers |
| Oct 31–Nov 1 | `port`: per-item tasks for llm.c kernels; run with Muse |
| Nov 2 | GPU runs (NVIDIA + AMD); `report`; open-weights variant (Muse Spark Open via a partner) if time |
| Nov 3 | Demo video, submission text, X post |
| Nov 4 | Submit by midday |

## Rules this project follows

- The toolkit is a **pinned, open-source dependency** (`oneapi-agent-toolkit` at tag
  `pre_hackathon`). Organizers confirmed in Discord (Oct 26) that an open-source dependency
  is fine as long as the demo and README mark it as pre-existing.
- Not a dashboard or Streamlit app (both on the anti-project list): the product is the
  agent workflow; reports and replays are its evidence.

- Public repository; new work only, and the demo shows only new work.
- Meta's models do the planning and the porting.
- No result is claimed until it is measured, with its evidence committed.
- Every paid step (API calls, GPU time) has a cost estimate and a cap before it runs.
  **Caps set by the author (2026-10-26):** $5 for development and smoke tests, $20 for the
  Muse comparison runs, at most $1 per run. `museport plan` refuses a request whose
  worst-case cost exceeds `--max-cost` (default $1) before sending it.
- No secrets in the repository; keys come from environment variables.
