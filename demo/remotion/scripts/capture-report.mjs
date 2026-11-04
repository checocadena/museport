// Full-page capture of the report's overview, for the film's slow pan (shot "report").
// usage: node scripts/capture-report.mjs <site dir> <out png>
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

const { chromium } = createRequire(import.meta.url)('playwright');
const [site, out] = process.argv.slice(2).map(p => resolve(p));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1,
  colorScheme: 'dark' });
await page.goto('file://' + site + '/index.html');
await page.waitForTimeout(500);
await page.screenshot({ path: out, fullPage: true });
await browser.close();
console.log(`report: ${out}`);
