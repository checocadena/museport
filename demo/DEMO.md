# Demo video: script and shot list (3:00 max)

This is plan item I1. Every number on screen comes from `demo/data.json`, which
`demo/make_data.py` builds from the committed evidence. The terminal shots replay real output
that `demo/capture.py` recorded.

## Build the shots

```bash
python3 demo/make_data.py > demo/data.json         # numbers from evidence/
python3 demo/capture.py <dir holding llm.c> > demo/sessions.json   # needs the toolchain
node demo/render.mjs demo/out                      # 1920x1080 H.264 MP4 + PNG per shot (Playwright, ffmpeg)
```

`demo/timeline.json` lists the shots in order, with durations, voiceover and on-screen notes. A
DaVinci Resolve MCP can build the edit from it (see [`RESOLVE.md`](RESOLVE.md)).

## Script

Times are approximate; the voiceover is the author's. **(rec)** marks a live screen recording to
capture on the laptop; every other shot is a rendered clip from `demo/out/`.

| # | Time | Shot | Voiceover |
|---|---|---|---|
| 1 | 0:00-0:08 | `disclosure` | "Pre-existing: my open-source oneapi-agent-toolkit, the SYCL tools and agent loop, used here as a pinned dependency. Built this week: Muse Port. Meta's Muse plans and ports CUDA code against hidden checks, then tunes it on real NVIDIA and AMD GPUs: llm.c's bf16 trainer, GPT-2 and llama2.cu." **This is the required opening line** (same as `docs/SUBMISSION.md`). |
| 2 | 0:08-0:14 | `open` | "CUDA locks code to NVIDIA. SYCL is the open standard. An AI port is only worth something if you can prove it's right." |
| 3 | 0:14-0:23 | `term_scan` | "Muse Port starts by reading the repository: Karpathy's llm.c, 22 CUDA files, 106 kernels, cuBLAS and cuDNN calls." |
| 4 | 0:23-0:30 | `pipeline` | "Muse writes a plan, checked against the scan: 23 items, 22 of 22 files, 0 issues. Then it ports each item in an agent loop." |
| 5 | 0:30-0:39 | `term_bug` | "Nothing counts as done until a check the model never sees agrees. It passes Muse's port, and catches a one-term slip: x cubed written as x squared." |
| 6 | 0:39-0:56 | `replay` | "Here Muse ports GPT-2's training step, with the real weights. 907 lines in one go. The check fails; Muse finds its own transpose mistake, fixes it, and all 10 steps match PyTorch." |
| 7 | 0:56-1:05 | `check` | "The checks are strict: SYCLomatic's fast-math output fails on 14 rows; the precise version passes." |
| 8 | 1:05-1:35 | `l1` | "The biggest test: llm.c's production trainer. bf16, cuBLASLt, warp reductions, 4,884 lines. Muse ported it in one run, 125 turns, under six dollars, and llm.c's own test passes. Then we gave Muse a tool that runs its code on rented GPUs, with a profiler. It went from 36 times CUDA's time to 1.5 with pure SYCL and oneMath, and 1.14 when it also calls cuBLAS. On an AMD MI350X: from 1,274 milliseconds a step to 63." |
| 9 | 1:35-1:42 | `gpus` | "The kernels, GPT-2 and llm.c's own test run from the same SYCL source on NVIDIA, against llm.c's CUDA, and on AMD." |
| 10 | 1:42-2:02 | **(rec)** the report site | Scroll the overview, open the "bf16 trainer" section and one port page's turn-by-turn replay. "Every result is in a report built only from committed evidence, down to each GPU call Muse made." |
| 11 | 2:02-2:09 | `numbers` | "18 of 19 kernel files correct, 17 passing on NVIDIA, 18 on AMD, and GPT-2 training ported for about three dollars." |
| 12 | 2:09-2:21 | **(rec)** `llm.c-sycl`: `make && ./test_gpt2fp32sycl`, sped up, ending on `overall okay: 1` | "The pieces go back together as llm.c-sycl. One make, and llm.c's own test prints 'overall okay'." |
| 13 | 2:21-2:39 | **(rec)** `llama2-sycl`: `make llama2 && ./llama2 stories15M.bin 0.0 64 "Once upon a time"`, the story streaming out | "And a repo it had never seen: llama2.cu, one 772-line CUDA file. Ported in one run, 20 turns, 45 cents. The story is llama2.cu's own, byte for byte, on a CPU, an NVIDIA GPU and an AMD GPU." |
| 14 | 2:39-2:53 | limits card (text on screen) | "Limits: the kernels are fp32 at reduced sizes, one run per item. On NVIDIA the trainer's 1.14 uses cuBLAS through SYCL interop; llm.c's test runs its fast paths only when we force them on, and passes; GPU numbers are single runs on rented machines." |
| 15 | 2:53-2:59 | `end` | "github.com/checocadena/museport." |

The `fixes` clip (the loop fixes, 8 to 15 loop successes) is still rendered but left out of the
timeline to make room for `l1`; the report and README keep that result.

Before recording, re-check every spoken number against `demo/data.json`, then re-run
`make_data.py` and `render.mjs` once the evidence changes.

**Oct 28 (evening): the clips match the evidence again.** `make_data.py` and `render.mjs` were
re-run for `gpus` (9) and `numbers` (11):
- **`gpus`:** it adds rows for GPT-2 training and llm.c's own test (`llm.c-sycl`), each on NVIDIA
  against llm.c's CUDA and on the AMD MI350X.
- **`numbers`:** it shows 17 ports on NVIDIA and 18 on AMD (with D2's oneMath runs), and GPT-2
  matching PyTorch on the CPU, an NVIDIA GPU and an AMD GPU.

`demo/out/` is not in git, so re-render it on the editing machine.

**J4 (Oct 28): shot 13 is now the `llama2-sycl` recording.** It replaces the optional live
`museport loop`; `timeline.json` moves the limits card to 2:37. Its spoken numbers are in
`data.json` under `j4`: 1 CUDA file of 772 lines, 20 turns, $0.45, identical on the CPU, the RTX
2000 Ada (also against llama2.cu itself) and the MI350X, and from a fresh clone.

Three new results have no rendered clip of their own:
- J1's recorded-CUDA checks (`evidence/diff-llmc-tier1`);
- J2's Muse-written checks (`evidence/spec-llmc`, `evidence/spec-llmc-fp32`);
- J3's `llm.c-sycl` (`evidence/j3-*`).

The report shows all three, so the report recording (shot 10) can cover them: "Differential
checks", "Hidden checks written by Muse" and "llm.c-sycl". J3 also gets its own recording,
shot 12.

**L1 (Oct 30): shot 8 is `l1`,** llm.c's production bf16 trainer, 30 s. Its numbers are in
`data.json` under `l1`, all read from `evidence/l1-llmc-bf16` by `museport.report.load_l1`:
- **Run 5:** 125 turns, $5.67.
- **NVIDIA:** 36x CUDA's time (run 5, RTX A5000), 5.65x (tuned with no GPU), 1.52x (pure SYCL +
  oneMath, our diagnostic) and 1.13-1.14x (with cuBLAS interop, the calls that built the final
  port).
- **AMD MI350X:** 1274, 578 and 63.5 ms per step.

To make room, `term_scan`, `term_bug`, `replay`, `check`, `gpus`, the report recording, the
llm.c-sycl recording, llama2-sycl and the limits card are shorter, and `fixes` is out. Total
2:59. The rendered `l1` and `disclosure` were checked frame by frame; re-render all clips with
`render.mjs` on the editing machine.

## Shots that need the author

- **The report site (10):** a screen recording of the published report. It must be shared
  publicly first.
- **`llm.c-sycl` (12):** in a fresh clone, run `cd llm.c-sycl && ./dev/download_starter_pack.sh`
  beforehand (10 s), then record `make && ./test_gpt2fp32sycl`. On the CPU device that takes
  about a minute in all (`make` 26 s, test 34 s: `evidence/j3-fresh-clone`), so speed it up.
  End on `overall okay: 1`. The NVIDIA and AMD results are in the report, not in this recording.
- **`llama2-sycl` (13):** in a fresh clone, run `cd llama2-sycl && wget
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin` beforehand, then
  record `make llama2 && ./llama2 stories15M.bin 0.0 64 "Once upon a time"`. On the CPU device
  `make llama2` took 7 s and the run under half a second (`evidence/j4-fresh-clone`), so it
  needs no speed-up. End on the story. The GPU results are in the report, not in this
  recording.
- **A live run (dropped for J4):** `museport loop` on a Tier 1 task with a $0.50 cap, sped up
  4-8×. If time allows, it can replace part of shot 10.
- **Voiceover:** the author's voice. Record it per shot so the edit can slip each clip to fit.
