# Editing the demo in DaVinci Resolve

The rendered clips are ordinary 1920×1080, 30 fps H.264 MP4s, so any editor can use them. This
page covers putting them together in DaVinci Resolve, by hand or through an MCP server driven
by Claude Code on the laptop.

## 1. Get the clips

On the laptop, with Node, Playwright and ffmpeg installed:

```bash
cd museport && git pull
python3 demo/make_data.py > demo/data.json   # refresh numbers if the evidence changed
node demo/render.mjs demo/out                # renders every shot; or name shots to re-render a few
```

`demo/sessions.json` is committed. Re-capturing it needs the oneAPI toolchain; see `capture.py`.

## 2a. By hand (about 20 minutes)

1. **Project:** a new project at 1920×1080, 30 fps.
2. **Media:** import `demo/out/*.mp4` into the Media Pool.
3. **Timeline:** drag the clips onto one track in the order of `demo/timeline.json`. Leave
   gaps for the recorded shots (report site, live loop) and the limits title.
4. **Voiceover:** record it per shot (Fairlight → ADR, or any mic recording) and lay it on
   audio track 1. Trim each clip to its line.
5. **Music:** optional, a quiet bed under it, around −24 dB.
6. **Deliver:** H.264 1080p, "YouTube 1080p" preset.

## 2b. With an MCP server and Claude Code

Community MCP servers drive Resolve through its scripting API:

- [lordhoell/davinci-resolve-mcp](https://github.com/lordhoell/davinci-resolve-mcp): broad API
  coverage, with a Claude Code skill;
- [Tooflex/davinci-resolve-mcp](https://mcpservers.org/servers/Tooflex/davinci-resolve-mcp);
- [apvlv/davinci-resolve-mcp](https://cdn.jsdelivr.net/gh/apvlv/davinci-resolve-mcp@main/README.md).

These are third-party projects, not checked here.

**Before you start:**
- Their listings say external scripting needs **DaVinci Resolve Studio**, the paid version,
  plus Python 3.10 or newer.
- Resolve must be running.
- Under Preferences → System → General, set "External scripting using" to *Local*.

Read the README of the server you pick before installing it.

Then ask Claude Code on the laptop something like:

> Using the DaVinci Resolve MCP, create a project "Muse Port demo" at 1920×1080, 30 fps. Import
> every MP4 in demo/out/ into a bin "shots". Create a timeline "demo v1" and append the clips in
> the order and durations of demo/timeline.json. For clips of kind "record" or "title", leave a
> gap of `duration_s`. Add a marker at each clip's `start_s`, named after the clip. Don't render
> yet; tell me the timeline's total length.

Then record the voiceover and the two screen recordings, drop them into the gaps, and deliver.

## Polish worth the time

- **Speed ramps:** in the `replay` shot, ramp up through the reading turns (1-15) and slow down
  at the first compile and the final PASS.
- **Dip to color:** a short dip to colour between sections (setup, GPT-2, results); avoid fancy
  transitions.
- **Zoom:** a slow push-in on the `check` grid when the red rows appear (Transform → Zoom
  keyframes, 1.00 to 1.06).
- **Captions:** burn them in (Subtitles track) for viewers watching with the sound off.
