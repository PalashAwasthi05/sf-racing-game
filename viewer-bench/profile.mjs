// CPU profile of the page while the autopilot drives (free running, uncapped): top functions by self time.
//   node profile.mjs <variant> [--w --h --dpr] [--seconds=8] [--js=...]
import { createRequire } from 'node:module';
import path from 'node:path';
import { writeFileSync } from 'node:fs';
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || '/Users/palash/Documents/reflex/code/soma', 'package.json'));
const { chromium } = require('playwright');
const args = process.argv.slice(2), opt = k => args.find(a => a.startsWith(`--${k}=`))?.split('=').slice(1).join('=');
const [variant = 'live', out = ''] = args.filter(a => !a.startsWith('--'));
const W = Number(opt('w') || 2560), H = Number(opt('h') || 1330), DPR = Number(opt('dpr') || 2), SECONDS = Number(opt('seconds') || 8);
const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: DPR }); const page = await ctx.newPage();
await page.goto(`http://127.0.0.1:${process.env.BENCH_PORT || 3107}/?variant=${variant}&free=1`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__bench, null, { timeout: 120_000 });
const boot = await page.evaluate(() => window.__bench.booted()); if (boot.loadError) { console.log(boot.loadError); process.exit(1); }
if (opt('js')) await page.evaluate(code => (0, eval)(code), opt('js'));
const cdp = await ctx.newCDPSession(page);
await page.evaluate(async () => {
  const B = window.__bench, car = window.__car.car; B.api().car(true); try { car.stop(); } catch {}
  const route = await fetch('/review-routes.json').then(r => r.json()).then(d => d.routes[0].map(p => [p[0], -p[1]]));
  let idx = 0; { const [x, z] = car.position(); let bd = Infinity; route.forEach((p, i) => { const d = (p[0] - x) ** 2 + (p[1] - z) ** 2; if (d < bd) { bd = d; idx = i; } }); }
  const held = new Set(), wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  const tick = () => {
    const [x, z, yaw] = car.position(), v = car.telemetry().speedKmh;
    for (let k = 0; k < 12; k++) { const j = (idx + 1) % route.length; if ((route[j][0] - x) ** 2 + (route[j][1] - z) ** 2 < (route[idx][0] - x) ** 2 + (route[idx][1] - z) ** 2) idx = j; else break; }
    let a = idx, run = 0; while (run < 10) { const b = (a + 1) % route.length; run += Math.hypot(route[b][0] - route[a][0], route[b][1] - route[a][1]); a = b; }
    const err = wrap(Math.atan2(-(route[a][0] - x), -(route[a][1] - z)) - yaw), keys = [];
    if (err > .04) keys.push('KeyA'); else if (err < -.04) keys.push('KeyD');
    if (v < 36 - 20 * Math.min(1, Math.abs(err))) keys.push('KeyW');
    for (const k of ['KeyA', 'KeyD', 'KeyW']) { const want = keys.includes(k); if (want !== held.has(k)) { B.key(k, want); if (want) held.add(k); else held.delete(k); } }
    window.__pilot = requestAnimationFrame(tick);
  };
  tick(); await new Promise(r => setTimeout(r, 2500));
});
await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 200 }); await cdp.send('Profiler.start');
await page.waitForTimeout(SECONDS * 1000);
const { profile } = await cdp.send('Profiler.stop');
if (out) writeFileSync(out, JSON.stringify(profile));
const byId = new Map(profile.nodes.map(n => [n.id, n])), self = new Map(), total = profile.samples.length;
const dt = (profile.endTime - profile.startTime) / 1000 / total;
for (const id of profile.samples) { const n = byId.get(id); const k = `${n.callFrame.functionName || '(anon)'} ${n.callFrame.url.split('/').pop().split('?')[0]}:${n.callFrame.lineNumber}`; self.set(k, (self.get(k) || 0) + 1); }
const rows = [...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 40);
console.log(`samples ${total}, ${(dt).toFixed(3)} ms each; top self time:`);
for (const [k, c] of rows) console.log(`${(100 * c / total).toFixed(1).padStart(5)}%  ${k}`);
await browser.close();
