// Same-pose captures of any world in headless Chromium on the real GPU, for looking at a change without the in-app pane
// (hidden, throttled, and reloaded by every edit): node shoot.mjs <variant> <out dir> <manifest path> <poses.json>
// poses.json: {"name": [[x, y, z], [x, y, z]], ...} as from / look-at in metric ENU (the viewer's __view.pose).
import { createRequire } from 'node:module';
import path from 'node:path';
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || '/Users/palash/Documents/reflex/code/soma', 'package.json'));
const { chromium } = require('playwright');
const [variant, out, manifest, posesFile] = process.argv.slice(2);
const poses = JSON.parse(readFileSync(posesFile, 'utf8')); mkdirSync(out, { recursive: true });
const port = process.env.BENCH_PORT || 3107, W = Number(process.env.SHOT_W || 1600), H = Number(process.env.SHOT_H || 1000);
const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })).newPage();
const logs = []; page.on('pageerror', e => logs.push('pageerror: ' + String(e.message).slice(0, 300))); page.on('console', m => { if (m.type() === 'error' || m.text().startsWith('T:')) logs.push(m.text().slice(0, 300)); });
const t0 = Date.now();
await page.goto(`http://127.0.0.1:${port}/?variant=${variant}&manifest=${encodeURIComponent(manifest)}`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__bench, null, { timeout: 300_000 });
const boot = await page.evaluate(() => window.__bench.booted());
if (boot.loadError) { console.log(boot.loadError, logs); await browser.close(); process.exit(1); }
await page.evaluate(() => window.__bench.frames(20));
for (const [name, [from, at]] of Object.entries(poses)) {
  await page.evaluate(p => window.__view.pose(p[0], p[1]), [from, at]);
  await page.evaluate(() => window.__bench.frames(4));
  const url = await page.evaluate(([w, h]) => window.__bench.capture(w, h), [W, H]);
  writeFileSync(path.join(out, `${name}.png`), Buffer.from(url.split(',')[1], 'base64'));
}
console.log(JSON.stringify({ shots: Object.keys(poses).length, loadS: Math.round((Date.now() - t0) / 1000), heapMB: boot.heapMB, logs: logs.slice(0, 30), progress: (boot.progress || []).map(p => [Math.round(p.t / 1000), p.m.slice(0, 40)]) }));
await browser.close();
