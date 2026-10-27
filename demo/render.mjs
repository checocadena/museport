// Render each demo shot to a 1920x1080 MP4 (and a PNG of its last frame) with Playwright.
// usage: node demo/render.mjs [outDir] [shot ...]   (needs ffmpeg on PATH)
import { createRequire } from 'node:module';
import { readFileSync, mkdirSync, renameSync, rmSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// require() also finds a globally installed Playwright, which an ESM import does not
const { chromium } = createRequire(import.meta.url)('playwright');
const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(process.argv[2] || join(here, 'out'));
const data = readFileSync(join(here, 'data.json'), 'utf8');
const sessions = readFileSync(join(here, 'sessions.json'), 'utf8');
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const probe = await browser.newPage();
await probe.goto('file://' + join(here, 'shots.html'));
const durations = await probe.evaluate(() => window.SHOT_DURATIONS);
await probe.close();
const names = process.argv.slice(3).length ? process.argv.slice(3) : Object.keys(durations);

for (const name of names) {
  const tmp = join(out, `.tmp-${name}`);
  rmSync(tmp, { recursive: true, force: true });
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 },
    recordVideo: { dir: tmp, size: { width: 1920, height: 1080 } } });
  const page = await ctx.newPage();
  await page.addInitScript(`window.DATA = ${data}; window.SESSIONS = ${sessions};`);
  await page.goto('file://' + join(here, 'shots.html'));
  const t0 = Date.now();
  await page.evaluate(n => window.play(n), name);
  await page.waitForTimeout(Math.max(0, durations[name] * 1000 - (Date.now() - t0)));
  await page.screenshot({ path: join(out, `${name}.png`) });
  await ctx.close();
  const webm = join(tmp, readdirSync(tmp).find(f => f.endsWith('.webm')));
  // trim Chromium's first blank frames, then H.264 at 30 fps for any editor
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', '0.25', '-i', webm, '-r', '30',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', join(out, `${name}.mp4`)]);
  rmSync(tmp, { recursive: true, force: true });
  console.log(`${name}: ${durations[name]} s -> ${join(out, name + '.mp4')}`);
}
await browser.close();
