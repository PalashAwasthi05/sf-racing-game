// Drive one variant in headless Chromium on the real GPU (ANGLE Metal) and write <out>/report.json, plus same-pose
// captures with --captures. Compare two variants with `python3 pixdiff.py <outA> <outB>`.
//
//   node run.mjs <variant> <out> [--captures] [--scenarios=street,crossing,aerial_a,aerial_b,street_corner,car_idle,car_drive]
//
// Playwright is not a dependency of the viewer; set PLAYWRIGHT_FROM to any directory whose node_modules has it.
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || '/Users/palash/Documents/reflex/code/soma', 'package.json'));
const { chromium } = require('playwright');

const [variant = 'live', out = `out-${variant}`] = process.argv.slice(2).filter(a => !a.startsWith('--'));
const captures = process.argv.includes('--captures');
const only = process.argv.find(a => a.startsWith('--scenarios='))?.split('=')[1].split(',');
const port = Number(process.env.BENCH_PORT || 3107);
mkdirSync(out, { recursive: true });

// Metric ENU (x east, y north, z up), as ?viewFrom/?viewAt take them.
const POSES = { street_corner: [[52, -34, 2], [76, -12, 20]], crossing: [[13, -58, 1.6], [-5, -44, 1]], aerial_a: [[80, -75, 95], [5, 30, 18]], aerial_b: [[185, 82, 65], [105, 139, 14]] };
const SCENARIOS = only || ['street', ...Object.keys(POSES), 'car_idle', 'car_drive'];

// 1440x900 CSS px at device pixel ratio 2 is the viewer's 1.25 render scale on a retina laptop screen.
const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-precise-memory-info'] });
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 })).newPage();
const logs = [];
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`${m.type()}: ${m.text()}`.slice(0, 300)); });
page.on('pageerror', e => logs.push('pageerror: ' + String(e.message).slice(0, 300)));

const t0 = Date.now();
await page.goto(`http://127.0.0.1:${port}/?variant=${variant}`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__bench, null, { timeout: 60_000 });
const boot = await page.evaluate(() => window.__bench.booted());
const report = { variant, when: new Date().toISOString(), loadWallMs: Date.now() - t0, boot, scenarios: {} };
if (boot.loadError) { console.log(JSON.stringify(report, null, 1)); await browser.close(); process.exit(1); }
report.firstFrames = await page.evaluate(() => window.__bench.frames(40));

const capture = async name => {
  if (!captures) return;
  const url = await page.evaluate(() => window.__bench.capture(1600, 1000));
  writeFileSync(path.join(out, `${name}.png`), Buffer.from(url.split(',')[1], 'base64'));
};
const measure = async (opts = {}) => {
  await page.evaluate(o => window.__bench.frames(30, o), opts);
  const r = await page.evaluate(o => window.__bench.frames(o.n || 120, o), opts);
  return { cpu: r.cpu, total: r.total, info: r.info };
};
for (const name of SCENARIOS) {
  if (POSES[name]) { await page.evaluate(p => window.__view.pose(p[0], p[1]), POSES[name]); report.scenarios[name] = await measure(); await capture(name); }
  else if (name === 'street') { report.scenarios.street = await measure(); await capture('street'); }
  else if (name === 'car_idle') { await page.evaluate(() => window.__bench.api().car(true)); report.scenarios.car_idle = await measure(); await capture('car_idle'); }
  else if (name === 'car_drive') { await page.evaluate(() => window.__bench.api().car(true)); report.scenarios.car_drive = await measure({ keys: ['KeyW'], n: 360 }); }
}
report.memory = await page.evaluate(() => window.__bench.memory());
report.logs = logs.slice(0, 40);
writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 1));
console.log(`${variant}: load ${report.loadWallMs} ms, first frame ${report.firstFrames.cpu.max} ms`);
for (const [k, v] of Object.entries(report.scenarios)) console.log(`  ${k.padEnd(14)} p50 ${String(v.total.p50).padStart(6)} ms  p90 ${String(v.total.p90).padStart(6)}  max ${String(v.total.max).padStart(7)}  draws ${v.info.calls}`);
if (logs.length) console.log('  console:', logs.slice(0, 5).join(' | '));
await browser.close();
