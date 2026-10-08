// Real-time A/B of live switches: the page runs free (no vsync, no frame-rate limit) at a fixed render scale; at each
// street view the baseline and each configuration alternate (base, A, base, B, ...) so drift cancels; prints each
// configuration's mean frame interval minus its neighbouring baselines'.
//   node ab.mjs <variant> '<json: {name: [onJS, offJS]}>' [--w --h --dpr --scale --secs --rounds]
import { createRequire } from 'node:module';
import path from 'node:path';
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || '/Users/palash/Documents/reflex/code/soma', 'package.json'));
const { chromium } = require('playwright');
const args = process.argv.slice(2), opt = k => args.find(a => a.startsWith(`--${k}=`))?.split('=').slice(1).join('=');
const [variant = 'live', cfgJson = '{}'] = args.filter(a => !a.startsWith('--'));
const W = Number(opt('w') || 2560), H = Number(opt('h') || 1330), DPR = Number(opt('dpr') || 2), SCALE = Number(opt('scale') || 1.25), SECS = Number(opt('secs') || 2.5), ROUNDS = Number(opt('rounds') || 2);
const configs = JSON.parse(cfgJson);
const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const page = await (await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: DPR })).newPage();
const logs = []; page.on('pageerror', e => logs.push('pageerror: ' + String(e.message).slice(0, 300)));
await page.goto(`http://127.0.0.1:${process.env.BENCH_PORT || 3107}/?variant=${variant}&free=1`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__bench, null, { timeout: 120_000 });
const boot = await page.evaluate(() => window.__bench.booted()); if (boot.loadError) { console.log(boot.loadError); process.exit(1); }
const res = await page.evaluate(async ({ SCALE, SECS, ROUNDS, configs }) => {
  const v = window.__view; v.quality(SCALE, true);
  const POSES = { king_ne: [[73.74, 35.03, 2.03], [103.16, 62.13, 0.99]], south_ne: [[-30.91, -72.58, 1.99], [-2.74, -44.18, 1.16]], east_ne: [[143.83, 61.46, 1.62], [168.95, 92.58, 0.63]] };
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const measure = () => new Promise(done => { const t = []; let last = 0; const start = performance.now();
    const tick = now => { if (last) t.push(now - last); last = now; if (now - start < SECS * 1000) requestAnimationFrame(tick); else done(t.reduce((a, b) => a + b, 0) / t.length); }; requestAnimationFrame(tick); });
  const out = { base: [] };
  for (const [from, at] of Object.values(POSES)) {
    v.pose(from, at); await wait(2500);
    for (let r = 0; r < ROUNDS; r++) for (const [name, [on, off]] of Object.entries(configs)) {
      const b0 = await measure(); (0, eval)(on); await wait(600); const x = await measure(); (0, eval)(off); await wait(400); const b1 = await measure();
      out.base.push((b0 + b1) / 2); (out[name] ??= []).push(x - (b0 + b1) / 2);
    }
  }
  return out;
}, { SCALE, SECS, ROUNDS, configs });
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
console.log(`${variant} ${W}x${H}@${DPR} scale ${SCALE}: base mean frame ${mean(res.base).toFixed(2)} ms (${(1000 / mean(res.base)).toFixed(0)} fps)`);
for (const [k, vals] of Object.entries(res)) if (k !== 'base') console.log(`  ${k.padEnd(16)} ${(mean(vals) >= 0 ? '+' : '') + mean(vals).toFixed(2)} ms   [${vals.map(x => x.toFixed(1)).join(' ')}]`);
if (logs.length) console.log(logs.join('\n'));
await browser.close();
