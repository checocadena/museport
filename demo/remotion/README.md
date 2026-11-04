# The Muse Port film (Remotion)

One Remotion composition, `Film` (`src/Film.tsx`), built from the voiceover, in the same way
as the BALI film:

- each of the 15 lines in [`public/lines.json`](public/lines.json) has its own shot;
- each line's take sets how long its shot runs (lead + take + tail), so re-recording a line
  moves every shot after it;
- captions follow Whisper's word timings, or share the line by word count without them;
- an optional music bed dips 4 dB under the voice with 0.5 s raised-cosine ramps.

Every number on screen comes from [`public/film-data.json`](public/film-data.json), which
`make_film_data.py` reads from the committed evidence, and `tests/test_film.py` checks the
spoken numbers in `lines.json` against it. The two terminal shots replay committed logs
(`evidence/j3-fresh-clone/test.log`, `evidence/j4-fresh-clone/session.log`), labeled
"real output, time compressed", so nothing needs recording by hand.

## Shots

| # | Shot | Made by |
|---|---|---|
| 1 | `disclosure` (the required opening line) | clip from `demo/render.mjs` |
| 2 | `hook`: the trainer, NVIDIA / AMD / CPU, `overall okay: 1` | Remotion card |
| 3-7 | `term_scan`, `pipeline`, `term_bug`, `replay`, `l1` | clips from `demo/render.mjs` |
| 8 | `speed`: portable step time vs CUDA (L4, A4000, H100) and AMD 1274 → 63 ms | Remotion card |
| 9 | `kernels`: 21 of 22 files, 20 correct, 19 on NVIDIA | Remotion card |
| 10 | `j3`: llm.c-sycl's test to `overall okay: 1` | replay of a committed log |
| 11 | `j4`: llama2-sycl's story | replay of a committed log |
| 12 | `onemath`: uxlfoundation/oneMath#793, tests 323 → 325 | Remotion card |
| 13 | `report`: a slow pan down the report | Playwright capture of `museport report` |
| 14 | `limits` | Remotion card |
| 15 | `end` | Remotion card |

## Make it (on the laptop)

Needs Node 20+, ffmpeg, Playwright with Chromium, and the repo's Python package
(`pip install -e .`).

```bash
cd demo/remotion
npm ci
scripts/prepare-assets.sh          # numbers, the six clips, the report capture (about 2 min)
npm run studio                     # preview: lines without a take use an estimated length
```

**Voiceover.** Record with [Cuecard](../prompter/index.html), the prompter in `demo/prompter/`:
open `demo/prompter/index.html` in Chrome (no install, no server), read each line off the
screen, record, play it back, keep your best take, then **Download kept takes**. It already
has these 15 lines; the zip holds `01.webm` … `15.webm`. Unzip it into `public/vo-raw/`.
(Any recorder works too: one file per line, named by its `n`, such as `01.m4a`.) Then:

```bash
scripts/prepare-voice.sh           # cleans each take into public/vo/vo-NN.wav, measures it
# with word timings for the captions (whisper.cpp), either locally or on a machine that has it:
WHISPER_SSH=deploy@<droplet> WHISPER_MODEL=/opt/whisper/ggml-small-q5_1.bin scripts/prepare-voice.sh
```

The cleanup is BALI's chain: left channel to mono, `highpass=f=80`, `afftdn=nf=-60`, a 2.5:1
`acompressor`, silence trimmed at both ends, `loudnorm=I=-16:TP=-1.5:LRA=9`, 48 kHz mono.
Re-run the script after replacing any take.

**Music (optional).** Put a royalty-free track at `public/music.wav` (or `.mp3`/`.m4a`) and
re-run `scripts/prepare-assets.sh` to register it. Don't use a commercial track: a public
video with one risks a takedown.

**Render.**

```bash
scripts/render-film.sh out/museport-film.mp4      # H.264 CRF 17, BT.709, AAC 256k, +faststart
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" scripts/render-film.sh out/museport-film.mp4
```

## Changing it

- **A line's words:** edit `public/lines.json`, re-record that take, re-run
  `prepare-voice.sh`. If the line has a number, `tests/test_film.py` must still pass.
- **A line's timing:** `lead` and `tail` (seconds of air before and after the take).
- **The numbers:** they follow the evidence; re-run `prepare-assets.sh` after it changes.

## Rules

- `public/clips/`, `public/vo*/`, `public/music.*` and `public/report/` are generated or
  personal and are not committed.
- The voice is the author's own; if a take is polished with a voice tool, say so in the
  video's description.
- Keep the total under 3:00 (the hackathon's limit). The estimate at a calm read is about 2:24.
