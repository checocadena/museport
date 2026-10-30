# Lessons: mistakes made during the hackathon, and the guard that now prevents each

Every entry was a real mistake in this project, by us (Claude, the assistant running it) unless
it says otherwise. Each has the guard that now stops it from happening again.

## Security

| What happened | Cost | Guard now |
|---|---|---|
| L1 run 4's shell tool printed `ONEAPI_AGENT_API_KEY` (an Anthropic key) into Muse's context and trace, so it went to Meta's API, and GitHub push protection blocked the commit. | A key to rotate; one commit redone. | Runs launch through [`scripts/clean-env.sh`](../scripts/clean-env.sh) (`env -i` and a whitelist), and the loop process is checked for credential variables after launch. Evidence passes [`museport.redact`](../src/museport/redact.py) and `secret_hits` before commit. GPU keys live only in the broker, outside the model's process. |

## Losing runs or results

| What happened | Cost | Guard now |
|---|---|---|
| Background commands are stopped after 2 h, which ended long runs. | A run restarted. | Long runs start with `setsid nohup` and write a `DONE` file. |
| A session restart killed `remote-test`'s local process. Its rented machine kept running, and nothing could collect its results. | 8 min of GPU, deleted by hand. | [`scripts/sweep-pods.py`](../scripts/sweep-pods.py) deletes `oneapi-gputest-*` machines when no local GPU job is alive, or when one has outlived the longest job limit. Each overnight run calls it when it ends. |
| A GPU setup script was *sourced* by the runner, and what it set leaked into the runner. We first blamed `set -u`; the real cause was a variable named `HERE`, the runner's own output directory, so the results archive went where the agent never looks. | Two oneMath test runs (about 40 min and $0.15 each) lost their logs. | A sourced setup script runs its whole body in a subshell `( ... )`. And before rerunning a failure, find the cause in the runner's code rather than guessing. |
| The first wrapper kept only the last 3,000 characters of a failing test, so the failing line was lost (AMD attempt 2). Then its failure filter matched "error" in "rel error" and listed passing lines. | One GPU rerun. | The wrapper lists failed checks first, matching only `NOT OK`, `MISMATCH` and `Nondeterminism`. A test keeps the committed wrapper equal to the code's. |
| The GPU broker showed Muse the tail of a failing test and cut off its failed checks. A crashing trainer's stderr was dropped. | Muse spent turns guessing. | The broker puts failed checks first and passes stderr through; both have tests. |

## Wrong or weak claims

| What happened | Cost | Guard now |
|---|---|---|
| We quoted run 5's NVIDIA speed as 2850 vs 79 ms. The recorded medians are 2891 vs 80.3 (the same 36x). | Docs and a tuning goal carried the wrong figures. | Report and demo numbers come from loaders over the evidence (`museport.report.load_l1`), checked by tests, and are never typed by hand. |
| We wrote that oneMath's *cuBLAS backend* lacks bf16 `gemm_batch` and bf16-output `gemm`. In fact oneMath's *API* has neither, on any backend. | The README, submission and report needed fixing. | Before a claim about another project goes into a document, read that project's source at a pinned commit. Here that was oneMath `develop` at `585e610`. |
| We compared a pure-oneMath speed from one GPU model with an interop speed from others. | The 1.13-1.14x headline missed the RTX A4500's 1.21x. | Speed comparisons are made as same-GPU pairs (`evidence/l1-llmc-bf16/gtune/diagnostics/pairs`). Every number states its GPU and its number of runs. |
| The CPU-tuned port was described from intentions, not measurements, before any GPU time existed. | Expectations set too high. | No speed claim without a GPU measurement in the evidence. |
| Muse's GPU-tuned port sent small shapes to different kernels than training uses, so llm.c's test did not run the fast paths. *(Muse's design.)* | A caveat on the best result. | We forced the fast paths on as a diagnostic, and llm.c's test passes on NVIDIA and AMD. The pure-oneAPI run's goal forbids size-based code paths. |

## The agent's tools

| What happened | Cost | Guard now |
|---|---|---|
| Muse tuned for speed with no GPU (CPU only). | One tuning run of guessing. | The `gpu_test` tool (`museport.gpubroker`) gives Muse real GPU timing, tests and profiles. |
| Muse used vendor libraries (cuBLAS through `dlopen`) to get the best NVIDIA number. *(Muse's choice; allowed by that run's goal.)* | The headline needed an interop caveat. | `museport gpu-broker --pure-oneapi` refuses such ports before renting a GPU. Goals now say what is allowed. |
| The loop's stuck rule (3 identical failures) fired on failed shell commands during normal debugging (run 4). | A $5.57 run stopped mid-port. | Failed `run` commands no longer count toward the stuck rule. |
| The broker's budget check reserves the worst case per call, so it refused a call that a small budget could have afforded. | One diagnostic retried. | Give a broker a budget well above its expected spend. |
| Our launcher sent a diagnostic to the wrong vendor (an edit to it did not apply, and we did not check). | $0.02 of duplicate GPU time. | Print and check the exact command before launching paid work. |
