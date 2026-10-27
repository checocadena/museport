# Start here (handoff, updated Oct 26, night)

## Context
- Muse Port is the Meta Global AI Developer Hackathon entry (Indie track, "Push the Frontier").
  Submit by midday Nov 4 (deadline 9:00 pm ET). Plan: [`PLAN.md`](PLAN.md).
- **Toolkit dependency.** It depends on
  [oneapi-agent-toolkit](https://github.com/checocadena/oneapi-agent-toolkit), pinned in
  `pyproject.toml` (`pre_hackathon` = `0e8f41d`, plus the changes in
  [`CHANGES-TO-TOOLKIT.md`](CHANGES-TO-TOOLKIT.md)). Organizers confirmed
  an open-source dependency is fine if marked as pre-existing. Toolkit changes go in
  [`CHANGES-TO-TOOLKIT.md`](CHANGES-TO-TOOLKIT.md).
- **Meta API key.** It is an environment **network secret**: the proxy adds the
  `Authorization` header for `api.meta.ai`, and the key is not in the environment. Set
  `ONEAPI_AGENT_AUTH_FROM_NETWORK=1`. Never print or commit keys.

## Rebuilding the environment (a fresh container has none of this)
1. Install both packages:
   - `pip install -e '.[dev]'` in museport;
   - `pip install --no-deps -e .` in a toolkit clone on branch `claude/affectionate-cray-vk73rn`
     until toolkit #20 is merged and pinned, because Tier 3 GPU checks need `native_flags`.
2. In the toolkit, install the toolchain:
   - `bash scripts/install-toolchain.sh /opt` (`icpx`, SYCLomatic, CUDA 12.3 headers,
     OpenCL CPU device);
   - for Tier 3, oneMKL and oneMath: the script's `WITH_ONEMATH=1` block (about 10 minutes).

   Then `source scripts/oneapi-env.sh`; `oneapi-agent --summary doctor` must say "all tools
   found".
3. `git clone --depth 1 https://github.com/karpathy/llm.c` (`f1e2ace`); the tasks use
   `llm.c/dev/cuda`.
4. **GPU keys** (`RUNPOD_API_KEY` and others) are in the environment;
   `oneapi-agent providers` checks them (read-only). DigitalOcean needs `ssh-keygen`
   (`apt-get install openssh-client`).

## Status

| Step | Result | Evidence |
|---|---|---|
| `museport plan` on llm.c `dev/cuda` | grounded, 23 items, 22/22 files, 0 issues, $0.2648 (after the 502 fix: streaming) | [plan](../evidence/plan-llmc-devcuda-stream/README.md), [first failure](../evidence/plan-llmc-devcuda/README.md) |
| `museport loop` smoke tests on the toolkit's own saxpy example | worked (verify PASS at turn 6; GPU PASS on an RTX 2000 Ada). Raw files are not in this repository, because they contain the toolkit's pre-existing example | - |
| Tier 1 (10 items) | 9 correct with 0 helpers, 4 loop success, 9 pass on NVIDIA, $1.67 | [tier 1](../evidence/port-llmc-tier1/README.md) |
| Tier 1 again, with the loop fixes (`delete_file`, ranged `read_file`, 32k output) | 10 correct with 0 helpers, 9 loop success, $1.01; not run on a GPU yet | [tier 1 run 2](../evidence/port-llmc-tier1-run2/README.md) |
| Tier 2 (5 items) | 4 correct with 0 helpers, 2 loop success, 4 pass on NVIDIA, $2.45 | [tier 2](../evidence/port-llmc-tier2/README.md) |
| Tier 3 (4 of 7) | 4 correct with 0 helpers, 2 loop success, 1 pass on NVIDIA (3 call oneMath: CPU only), $1.40 | [tier 3](../evidence/port-llmc-tier3/README.md) |
| AMD | not run: DigitalOcean's bare-IP agent is unreachable from this cloud session; Runpod had no AMD stock | tier 1 `gpu/logs/` |

**Open issues in the ports:**
- `t1-fused-residual-forward` and `t2-softmax-forward`: correct, but c2s helpers remain (reruns:
  45 helpers left; fails verify).
- `t2-layernorm-backward`: copies one implementation into 8 of its 9 variants (fixed by the
  round-2 rerun: correct, 0 helpers, a distinct kernel per variant).

**Not attempted:** `attention_forward`, `attention_backward`, `nccl_all_reduce`.

## Repository
- This repository (`checocadena/museport`) is a standalone extract of the hackathon work, made
  on Oct 26 from the development repository with no earlier history. Development continues
  here, on branches with PRs.
- The development repository was renamed `checocadena/museport-dev` and made private. Do not
  push it here: it carries early toolkit-derived files in its history.
- The toolkit's `native_flags` (toolkit #20) and the loop's `delete_file` and ranged
  `read_file` (toolkit #21) and the AMD native build's header hipify (toolkit #22) are
  merged; museport pins #22's merge commit `23fdcff`.

## Spend (known; caps: $5 development, $20 comparison runs, $1 per run)

| Bucket | Spent |
|---|---|
| Development and smoke tests | about $0.55: plan $0.2648, mini-fixture plan $0.0502, saxpy loop $0.0592, smoke test $0.0072, probes about $0.17. The first, failed plan is unknown: at most $1.15 if all 4 attempts were billed |
| Port runs (comparison) | $8.3585: Tier 1 $1.6698, Tier 2 $2.4454 (incl. one crashed run), Tier 3 $1.3960, reruns round 1 $1.3971, round 2 $1.4502, Tier 1 run 2 $1.0064 (total $9.3649) |
| GPUs | NVIDIA about $0.13 (all pods deleted and confirmed); DigitalOcean AMD attempt about $1.0 (two droplets, deleted, no SSH key left) |

## Known toolkit behaviour (handled in museport, toolkit unchanged)
- The OpenAI-compatible client does not stream; museport passes its own streaming transport.
- `Toolbox` raises `KeyError` on a tool call missing an argument; museport's `SafeToolbox`
  turns it into an error result.
- A missing `ssh-keygen` on DigitalOcean is reported as "usually means no stock".

## Next steps
The full list, with costs and order, is in [`MASTER-PLAN.md`](MASTER-PLAN.md).

## Working rules (from the author)
- No paid API calls or GPU rentals without the author's explicit OK and a stated cost
  estimate; hard cap per run and per batch (caps above).
- Never commit keys. Ask before pushing to `main` or merging; small commits on branches with PRs.
- No Meta result is claimed until measured, with the evidence committed.
- Demo and README show only work done during the hackathon.
- If something is blocked or unclear, stop and ask.
