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
| 1 | 0:00-0:08 | `disclosure` | "Pre-existing: my open-source oneapi-agent-toolkit, the tools and agent loop. Built this week: Muse Port, where Meta's Muse plans and ports CUDA code, with hidden checks, GPU runs and a GPT-2 port." **This is the required opening line.** |
| 2 | 0:08-0:14 | `open` | "NVIDIA's CUDA locks code to one vendor. SYCL is the open standard. Moving code across is slow, and an AI port is only worth something if you can prove it's right." |
| 3 | 0:14-0:23 | `term_scan` | "Muse Port starts by reading the repository: Karpathy's llm.c, 22 CUDA files, 106 kernels, cuBLAS and cuDNN calls." |
| 4 | 0:23-0:31 | `pipeline` | "Muse Spark writes a migration plan, and every file and kernel it names is checked against the scan: 23 items, 22 of 22 files, 0 issues. Then Muse ports each item inside an agent loop." |
| 5 | 0:31-0:40 | `term_bug` | "Nothing counts as done until a check the model never sees agrees. Here it passes Muse's port, then catches a one-term slip: x cubed written as x squared." |
| 6 | 0:40-0:59 | `replay` | "This is Muse porting GPT-2's training, llm.c's real code with the real 124-million-parameter weights: forward, backward and the AdamW update. It reads the source and writes 907 lines of SYCL in one go. The hidden check fails: the losses are off. Muse finds its own matrix-transpose mistake, fixes it, and all 10 training steps match PyTorch." |
| 7 | 0:59-1:11 | `check` | "The check compares against PyTorch's own numbers, at llm.c's tolerance. It is strict: SYCLomatic's fast-math output fails on 14 rows, and the precise version passes." |
| 8 | 1:11-1:19 | `fixes` | "We improved the loop itself. Same 19 kernels: loop success from 8 to 15, correct from 17 to 18, and cost down a third." |
| 9 | 1:19-1:27 | `gpus` | "And the same SYCL source runs on both vendors: NVIDIA, checked against llm.c's own CUDA, and an AMD MI350X." |
| 10 | 1:27-1:52 | **(rec)** the report site | Scroll the overview, open one port page and its turn-by-turn replay. "Every result is in a report built only from committed evidence." |
| 11 | 1:52-1:58 | `numbers` | "18 of 19 kernel files correct, 15 on NVIDIA against CUDA, 10 of 10 on AMD, and GPT-2 training ported for about three dollars." |
| 12 | 1:58-2:40 | *(optional)* **(rec)** a live `museport loop` on one kernel, sped up | "And it runs live: about 10 cents per kernel." |
| 13 | 2:40-2:55 | limits card (text on screen) | "Limits: fp32, reduced sizes, one run per item. GPT-2's backward pass is next." |
| 14 | 2:55-3:00 | `end` | "github.com/checocadena/museport." |

Before recording, re-check every spoken number against `demo/data.json`, then re-run
`make_data.py` and `render.mjs` once the evidence changes: for example, GPT-2 stage 2 adds a
line to shots 6 and 11.

## Shots that need the author

- **The report site (10):** a screen recording of the published report. It must be shared
  publicly first.
- **A live run (12):** `museport loop` on a Tier 1 task with a $0.50 cap. Speed it up 4-8×
  in the edit.
- **Voiceover:** the author's voice. Record it per shot so the edit can slip each clip to fit.
