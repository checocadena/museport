// Like render.mjs, but lossless: the browser streams every composited frame as a PNG
// (Chrome DevTools screencast), and ffmpeg assembles them at 30 fps by their timestamps.
// render.mjs's recordVideo is a ~1 Mbps VP8 stream, which blurs text and thin colored lines.
// usage: node demo/render-hq.mjs <outDir> [shot ...] [--scale 2]
import { createRequire } from 'node:module';
import { readFileSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = createRequire(import.meta.url)('playwright');
const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const si = args.indexOf('--scale');
const scale = si >= 0 ? Number(args.splice(si, 2)[1]) : 1;
const out = resolve(args[0] || join(here, 'out'));
const data = readFileSync(join(here, 'data.json'), 'utf8');
const sessions = readFileSync(join(here, 'sessions.json'), 'utf8');
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const probe = await browser.newPage();
await probe.goto('file://' + join(here, 'shots.html'));
const durations = await probe.evaluate(() => window.SHOT_DURATIONS);
await probe.close();
const names = args.slice(1).length ? args.slice(1) : Object.keys(durations);

for (const name of names) {
  const tmp = join(out, `.hq-${name}`);
  rmSync(tmp, { recursive: true, force: true }); mkdirSync(tmp);
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: scale });
  const page = await ctx.newPage();
  await page.addInitScript(`window.DATA = ${data}; window.SESSIONS = ${sessions};`);
  await page.goto('file://' + join(here, 'shots.html'));
  const cdp = await ctx.newCDPSession(page);
  const frames = [];
  let t0 = null;
  cdp.on('Page.screencastFrame', async ({ data: png, metadata, sessionId }) => {
    const ts = metadata.timestamp;
    if (t0 === null) t0 = ts;
    const file = join(tmp, `f${String(frames.length).padStart(5, '0')}.png`);
    writeFileSync(file, Buffer.from(png, 'base64'));
    frames.push({ file, t: ts - t0 });
    await cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
  });
  await cdp.send('Page.startScreencast', { format: 'png', everyNthFrame: 1,
    maxWidth: 1920 * scale, maxHeight: 1080 * scale });
  // keep the page repainting, so the screencast sends frames even while nothing moves
  await page.evaluate(() => { const d = document.createElement('div');
    d.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;opacity:.01';
    document.body.append(d); let k = 0; (function tick() { d.style.opacity = (k++ % 2) ? '.01' : '.02';
      requestAnimationFrame(tick); })(); });
  const start = Date.now();
  await page.evaluate(n => window.play(n), name);
  await page.waitForTimeout(Math.max(0, durations[name] * 1000 - (Date.now() - start)) + 300);
  await cdp.send('Page.stopScreencast');
  await ctx.close();
  // a concat list with each frame's own duration, then constant 30 fps
  const lines = [];
  for (let i = 0; i < frames.length; i++) {
    const d = i + 1 < frames.length ? frames[i + 1].t - frames[i].t : 1 / 30;
    lines.push(`file '${frames[i].file}'`, `duration ${Math.max(0.001, d).toFixed(4)}`);
  }
  lines.push(`file '${frames[frames.length - 1].file}'`);
  writeFileSync(join(tmp, 'list.txt'), lines.join('\n'));
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', join(tmp, 'list.txt'),
    '-t', String(durations[name]), '-vf', `fps=30,scale=${1920 * scale}:${1080 * scale}:flags=lanczos`,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '10', '-pix_fmt', 'yuv420p', join(out, `${name}.mp4`)]);
  const fps = frames.length / (frames.length ? frames[frames.length - 1].t : 1);
  rmSync(tmp, { recursive: true, force: true });
  console.log(`${name}: ${durations[name]} s, ${frames.length} frames captured (${fps.toFixed(1)}/s) -> ${name}.mp4`);
}
await browser.close();
