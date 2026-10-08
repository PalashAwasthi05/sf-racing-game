// Drive the car round the review route (an autopilot steering for a point ahead on the route at city speed) and record
// every frame, at a real screen size: the number that matters is whether driving holds 60 fps.
//
//   node drive.mjs <variant> <out> [--w=2560 --h=1330 --dpr=2] [--seconds=40] [--free] [--manifest=/model.json] [--trace=t.json]
//
// Stepped (default): the animation loop is stepped at 60 Hz and the GPU drained after every frame, so each frame's time
// is its whole serial cost (JS + GPU); a frame over 16.7 ms cannot be shown at 60 fps. --free runs the page's own
// requestAnimationFrame and records the intervals between frames as the browser paces them (CPU and GPU overlap, as in
// use). 2560x1330 at dpr 2 is a maximised browser window on a 5K display; 1512x860 the laptop's own screen.
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || '/Users/palash/Documents/reflex/code/soma', 'package.json'));
const { chromium } = require('playwright');

const args = process.argv.slice(2), opt = k => args.find(a => a.startsWith(`--${k}=`))?.split('=')[1];
const [variant = 'live', out = `drive-${variant}`] = args.filter(a => !a.startsWith('--'));
const W = Number(opt('w') || 2560), H = Number(opt('h') || 1330), DPR = Number(opt('dpr') || 2), SECONDS = Number(opt('seconds') || 40);
const free = args.includes('--free'), manifest = opt('manifest') || '', hz = Number(opt('hz') || 0);   // --hz=60 with --free: paced as on a 60 Hz display
const port = Number(process.env.BENCH_PORT || 3107);
mkdirSync(out, { recursive: true });

// --uncapped: no vsync and no frame-rate limit, so a free-running page shows its real throughput (above 60 when it can)
const uncapped = args.includes('--uncapped');
const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-precise-memory-info', '--autoplay-policy=no-user-gesture-required', ...(uncapped ? ['--disable-gpu-vsync', '--disable-frame-rate-limit'] : [])] });
const page = await (await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: DPR })).newPage();
const logs = [];
page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`${m.type()}: ${m.text()}`.slice(0, 300)); });
page.on('pageerror', e => logs.push('pageerror: ' + String(e.message).slice(0, 300)));
await page.goto(`http://127.0.0.1:${port}/?variant=${variant}${free ? '&free=1' : ''}${manifest ? '&manifest=' + encodeURIComponent(manifest) : ''}`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__bench, null, { timeout: 120_000 });
const boot = await page.evaluate(() => window.__bench.booted());
if (boot.loadError) { console.log(boot.loadError); await browser.close(); process.exit(1); }
// --js='...': run after load (live switches, e.g. __view.tune({...}) or __view.quality(1.25, true))
if (opt('js')) await page.evaluate(code => (0, eval)(code), args.find(a => a.startsWith('--js=')).slice(5));

// --gc: a full garbage collection after load, before the drive (the load leaves ~1 GB of garbage that V8 otherwise
// collects some seconds into the drive)
if (args.includes('--gc')) { const c = await page.context().newCDPSession(page); await c.send('HeapProfiler.collectGarbage'); await page.waitForTimeout(500); }
// --trace=<file.json>: a Chrome trace of the drive, every process (what the GPU process was doing during a long frame)
const trace = opt('trace');
if (trace) await browser.startTracing(page, { path: trace, categories: ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'disabled-by-default-devtools.timeline.frame', 'toplevel', 'blink', 'v8', 'disabled-by-default-v8.gc', 'gpu', 'gpu.angle', 'disabled-by-default-gpu.service', 'disabled-by-default-gpu.decoder', 'viz', 'cc', 'benchmark'] });
const result = await page.evaluate(async ({ SECONDS, free, hz }) => {
  const B = window.__bench, api = B.api(), C = window.__car, car = C.car;
  const route = await fetch('/review-routes.json').then(r => r.json()).then(d => d.routes[0].map(p => [p[0], -p[1]]));   // three x, z
  // the engine sound is not part of the measure
  try { car.stop(); } catch {}
  api.car(true);
  const near = (x, z) => { let b = 0, bd = Infinity; for (let i = 0; i < route.length; i++) { const d = (route[i][0] - x) ** 2 + (route[i][1] - z) ** 2; if (d < bd) { bd = d; b = i; } } return b; };
  let idx = near(...car.position().slice(0, 2)), stuck = 0;
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  // keys for this frame: steer for a point ~10 m ahead on the route, hold about 40 km/h
  const pilot = () => {
    const [x, z, yaw] = car.position(), v = car.telemetry().speedKmh;
    for (let k = 0; k < 12; k++) { const j = (idx + 1) % route.length; if ((route[j][0] - x) ** 2 + (route[j][1] - z) ** 2 < (route[idx][0] - x) ** 2 + (route[idx][1] - z) ** 2) idx = j; else break; }
    let a = idx, run = 0; while (run < 10) { const b = (a + 1) % route.length; run += Math.hypot(route[b][0] - route[a][0], route[b][1] - route[a][1]); a = b; }
    const want = Math.atan2(-(route[a][0] - x), -(route[a][1] - z)), err = wrap(want - yaw);
    const keys = [];
    if (err > .04) keys.push('KeyA'); else if (err < -.04) keys.push('KeyD');
    if (v < 36 - 20 * Math.min(1, Math.abs(err))) keys.push('KeyW'); else if (v > 50) keys.push('KeyS');
    stuck = v < 3 ? stuck + 1 : 0;
    return { keys, v, x, z };
  };
  const frames = [], samples = [];
  const total = Math.round(SECONDS * 60);
  let resets = 0;
  if (!free) {
    for (let i = 0; i < total; i++) {
      const p = pilot();
      if (stuck > 120) { B.key('KeyR', true); B.key('KeyR', false); stuck = 0; resets++; }
      const r = await B.frames(1, { keys: p.keys });
      frames.push({ t: r.total.max, cpu: r.cpu.max, v: +p.v.toFixed(1), x: +p.x.toFixed(1), z: +p.z.toFixed(1), calls: r.info.calls });
      if (i % 60 === 0) samples.push(window.__view.info());
    }
  } else {
    // the page paces itself: the pilot runs in its own animation frame callback, after the viewer's (which is timed)
    const held = new Set(), R = window.__view.renderer; let js = 0;
    { const raf = window.requestAnimationFrame, timed = cb => t => { const a = performance.now(); cb(t); js = performance.now() - a; };
      // hz=60: a callback every second vsync, as on a 60 Hz display (headless paces at 120 Hz here)
      window.requestAnimationFrame = hz === 60 ? cb => raf(() => raf(timed(cb))) : cb => raf(timed(cb)); }
    await new Promise(done => {
      let n = 0, last = performance.now();
      const tick = now => {
        const p = pilot();
        for (const k of ['KeyA', 'KeyD', 'KeyW', 'KeyS']) { const want = p.keys.includes(k); if (want !== held.has(k)) { B.key(k, want); if (want) held.add(k); else held.delete(k); } }
        if (stuck > 120) { B.key('KeyR', true); B.key('KeyR', false); stuck = 0; resets++; }
        if (n > 0) frames.push({ t: +(now - last).toFixed(2), js: +js.toFixed(2), calls: R.info.render.calls, tris: R.info.render.triangles, pf: window.__view.probe?.() ?? null, heap: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null, progs: R.info.programs?.length, v: +p.v.toFixed(1), x: +p.x.toFixed(1), z: +p.z.toFixed(1) });
        last = now; n++;
        if (n % 60 === 0) samples.push(window.__view.info());
        if (performance.now() - start < SECONDS * 1000) requestAnimationFrame(tick); else done();
      };
      const start = performance.now(); requestAnimationFrame(() => requestAnimationFrame(tick));
    });
    for (const k of held) B.key(k, false);
  }
  return { frames, samples, resets, distance: frames.reduce((d, f, i) => i ? d + Math.hypot(f.x - frames[i - 1].x, f.z - frames[i - 1].z) : 0, 0) };
}, { SECONDS, free, hz });

if (trace) await browser.stopTracing();
const t = result.frames.map(f => f.t).sort((a, b) => a - b), q = p => t[Math.min(t.length - 1, Math.floor(p * t.length))];
const over = ms => result.frames.filter(f => f.t > ms).length;
const summary = { variant, size: `${W}x${H}@${DPR}`, mode: free ? (uncapped ? 'free-uncapped' : 'free') : 'stepped', frames: t.length, distanceM: Math.round(result.distance), resets: result.resets,
  p50: q(.5), p90: q(.9), p95: q(.95), p99: q(.99), max: t[t.length - 1], over16_7: over(16.7), over20: over(20), over33: over(33),
  fps: free ? +(1000 / (result.frames.reduce((a, f) => a + f.t, 0) / t.length)).toFixed(1) : null,
  jsP50: free ? (() => { const c = result.frames.map(f => f.js).sort((a, b) => a - b); return c[Math.floor(c.length / 2)]; })() : null,
  cpuP50: free ? null : (() => { const c = result.frames.map(f => f.cpu).sort((a, b) => a - b); return c[Math.floor(c.length / 2)]; })(),
  quality: result.samples.slice(-3), logs: logs.slice(0, 10) };
writeFileSync(path.join(out, 'drive.json'), JSON.stringify({ summary, frames: result.frames, samples: result.samples }, null, 0));
console.log(JSON.stringify(summary));
await browser.close();
