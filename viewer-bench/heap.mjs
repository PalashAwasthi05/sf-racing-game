// Where the viewer's memory goes while it loads: a sampling heap profile of every allocation made from navigation to
// the end of load (collected ones included), then the heap before and after a full garbage collection.
//
//   node heap.mjs <variant> <out-dir> [--w=2560 --h=1330 --dpr=2] [--manifest=/model.json] [--interval=65536]
//
// Writes <out-dir>/heap.json (summary: top allocating functions, used and live heap) and load.heapprofile (open it in
// DevTools' Memory panel). The heap sizes are V8's own heap; ArrayBuffer contents (geometry, the downloaded model) live
// outside it and show in the renderer's footprint instead.
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || '/Users/palash/Documents/reflex/code/soma', 'package.json'));
const { chromium } = require('playwright');

const args = process.argv.slice(2), opt = k => args.find(a => a.startsWith(`--${k}=`))?.split('=')[1];
const [variant = 'live', out = `heap-${variant}`] = args.filter(a => !a.startsWith('--'));
const W = Number(opt('w') || 2560), H = Number(opt('h') || 1330), DPR = Number(opt('dpr') || 2), manifest = opt('manifest') || '';
const interval = Number(opt('interval') || 65536), port = Number(process.env.BENCH_PORT || 3107);
mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ headless: true, args: ['--headless=new', '--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-precise-memory-info'] });
const page = await (await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: DPR })).newPage();
const cdp = await page.context().newCDPSession(page);
await cdp.send('HeapProfiler.enable');
await cdp.send('HeapProfiler.startSampling', { samplingInterval: interval, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true });
const t0 = Date.now();
await page.goto(`http://127.0.0.1:${port}/?variant=${variant}${manifest ? '&manifest=' + encodeURIComponent(manifest) : ''}`, { waitUntil: 'load' });
await page.waitForFunction(() => !!window.__bench, null, { timeout: 180_000 });
const boot = await page.evaluate(() => window.__bench.booted());
const loadS = (Date.now() - t0) / 1000;
const { profile } = await cdp.send('HeapProfiler.stopSampling');
const heap = () => page.evaluate(() => Math.round(performance.memory.usedJSHeapSize / 1e6));
// the renderer process's footprint (V8 heap, ArrayBuffers, decoded images, the model blob if it is kept in process)
const footprint = () => { try { return execFileSync('ps', ['-axo', 'rss,command']).toString().split('\n').filter(l => l.includes('--type=renderer') && l.includes('headless')).map(l => Math.round(parseInt(l) / 1024)); } catch { return []; } };
const used = await heap(), rssBefore = footprint();
await cdp.send('HeapProfiler.collectGarbage');
const live = await heap(), rssAfter = footprint();

// self size per allocating function (name, script, line), all allocations during load
const by = new Map();
const walk = (n, stack) => {
  const f = n.callFrame, key = `${f.functionName || '(anonymous)'} ${f.url.split('/').slice(-1)[0]}:${f.lineNumber + 1}`;
  if (n.selfSize) { const e = by.get(key) ?? { self: 0, stacks: new Map() }; e.self += n.selfSize; const s = stack.slice(-4).join(' < '); e.stacks.set(s, (e.stacks.get(s) ?? 0) + n.selfSize); by.set(key, e); }
  for (const c of n.children) walk(c, [...stack, key]);
};
walk(profile.head, []);
const total = [...by.values()].reduce((a, e) => a + e.self, 0);
const top = [...by.entries()].sort((a, b) => b[1].self - a[1].self).slice(0, 40).map(([k, e]) => ({ fn: k, MB: +(e.self / 1e6).toFixed(1), from: [...e.stacks.entries()].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([s, v]) => `${(v / 1e6).toFixed(1)} MB: ${s}`) }));
const summary = { variant, loadS, loadError: boot.loadError ?? null, allocatedMB: +(total / 1e6).toFixed(0), usedHeapMB: used, liveHeapMB: live, rendererRssMB: { beforeGC: rssBefore, afterGC: rssAfter }, top };
writeFileSync(path.join(out, 'heap.json'), JSON.stringify(summary, null, 1));
writeFileSync(path.join(out, 'load.heapprofile'), JSON.stringify(profile));
console.log(JSON.stringify({ ...summary, top: summary.top.slice(0, 12).map(t => `${t.MB} MB ${t.fn}`) }, null, 1));
await browser.close();
