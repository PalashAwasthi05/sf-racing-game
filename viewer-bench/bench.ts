// Benchmark harness for the King Street viewer (work/viewer-bench; nothing here is part of the app). It loads a copy of
// the viewer modules from variants/<name>/ (?variant=<name>), drives the animation loop deterministically (stepped
// requestAnimationFrame with fixed timestamps) and measures per-frame CPU time and CPU+GPU time (gl.finish), draw
// calls per render target, and same-pose captures. GPU timer queries are available too but carry a floor of a few
// milliseconds per query on ANGLE Metal, so attribute cost by toggling features instead.
//
// Deterministic runs: Math.random is seeded and the clock can be frozen for captures, so two captures of one pose differ
// only by what the code under test changed.
{ let seed = 0x2f6b1a3d; Math.random = () => { seed |= 0; seed = seed + 0x6d2b79f5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const realNow = performance.now.bind(performance);
let frozenNow: number | null = null;
performance.now = () => frozenNow ?? realNow();
type Seg = { label: string; q: WebGLQuery };
const params = new URLSearchParams(location.search);
const variant = /^[a-z_0-9-]+$/.test(params.get('variant') || '') ? params.get('variant')! : 'live';
const realRaf = window.requestAnimationFrame.bind(window);
const realCaf = window.cancelAnimationFrame.bind(window);
let stepped = params.get('free') !== '1';
let queue: { id: number; cb: FrameRequestCallback }[] = [];
let nextId = 1;
window.requestAnimationFrame = (cb: FrameRequestCallback) => {
  if (!stepped) return realRaf(cb);
  const id = nextId++; queue.push({ id, cb }); return id;
};
window.cancelAnimationFrame = (id: number) => { queue = queue.filter(e => e.id !== id); realCaf(id); };

const progress: { t: number; m: string }[] = [];
const t0 = performance.now();
let view: any = null, api: any = null, loadError: string | null = null;
let clock = performance.now() + 1e6;

async function boot() {
  // ?manifest=/compare/x.json (as in the viewer): another area's build, whose manifest names its own world data
  const requested = params.get('manifest'), same = (p: unknown) => typeof p === 'string' && /^\/compare\/[\w.-]+\.json$/.test(p);
  let worldPath = '/world.json';
  if (same(requested)) { const m = await fetch(requested!, { cache: 'no-cache' }).then(r => r.json()).catch(() => null); if (same(m?.world)) worldPath = m.world; }
  const world = await fetch(worldPath, { cache: 'no-cache' }).then(r => r.json());
  const mod = await import(`./variants/${variant}/world-view.ts`);
  const host = document.getElementById('host') as HTMLDivElement;
  api = await mod.createWorldView(host, world, () => {}, (m: string) => {
    const last = progress[progress.length - 1];
    if (!last || last.m.slice(0, 18) !== m.slice(0, 18) || !m.startsWith('Loading')) progress.push({ t: performance.now() - t0, m });
  });
  progress.push({ t: performance.now() - t0, m: 'ready' });
  view = (window as any).__view;
  // GTAO's denoise noise is drawn from Math.random, which three.js also spends on every object's UUID: any change in how
  // many objects are created before the pass shifts the pattern (in the app it is simply random per page load). Redraw
  // it from a fixed seed so variants are compared on the same noise.
  { const ao = view.composer.passes.find((p: any) => p.pdMaterial); let seed = 0x51f15e;
    const rnd = Math.random; Math.random = () => { seed |= 0; seed = seed + 0x6d2b79f5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
    try { const tex = ao._generateNoise(); ao.pdNoiseTexture.dispose(); ao.pdNoiseTexture = tex; ao.pdMaterial.uniforms.tNoise.value = tex; } finally { Math.random = rnd; } }
}
const booted = boot().catch(e => { loadError = String(e?.stack || e); });

// ---- GPU timing by render-target segment -------------------------------------------------------------------
let gl: WebGL2RenderingContext | null = null, ext: any = null;
let timing = false, segs: Seg[] = [], current: Seg | null = null;
let counting = false, callSegs: { label: string; calls: number; tris: number }[] = [], callCur: { label: string; calls: number; tris: number; c0: number; t0: number } | null = null;
const labels = new Map<any, string>();
function labelOf(target: any): string {
  if (target === null) return 'canvas';
  const known = labels.get(target); if (known) return known;
  const name = target.texture?.name || '';
  if (name) return name;
  if (target.isWebGLCubeRenderTarget) return 'cube-probe';
  return `rt:${target.width}x${target.height}${target.samples ? 'x' + target.samples : ''}${target.texture?.generateMipmaps ? '+mip' : ''}`;
}
function mark(label: string) {
  if (!timing || !gl || !ext) return;
  if (current) gl.endQuery(ext.TIME_ELAPSED_EXT);
  const q = gl.createQuery()!; gl.beginQuery(ext.TIME_ELAPSED_EXT, q); current = { label, q }; segs.push(current);
}
function stopTiming() { if (current && gl && ext) { gl.endQuery(ext.TIME_ELAPSED_EXT); current = null; } }
function instrument() {
  const renderer = view.renderer; gl = renderer.getContext(); ext = gl!.getExtension('EXT_disjoint_timer_query_webgl2');
  const csm = view.atmosphere?.csm; if (csm) csm.lights.forEach((l: any, i: number) => { if (l.shadow.map) labels.set(l.shadow.map, 'shadow:cascade' + i); });
  const setRT = renderer.setRenderTarget.bind(renderer);
  renderer.setRenderTarget = (t: any, face?: number, mip?: number) => {
    // shadow maps are created lazily: label on first sight
    if (csm) csm.lights.forEach((l: any, i: number) => { if (l.shadow.map && !labels.has(l.shadow.map)) labels.set(l.shadow.map, 'shadow:cascade' + i); });
    const from = renderer.getRenderTarget(); const res = setRT(t, face, mip);
    if (t !== from) { mark(labelOf(t)); if (counting) { const info = renderer.info.render; if (callCur) callSegs.push({ label: callCur.label, calls: info.calls - callCur.c0, tris: info.triangles - callCur.t0 }); callCur = { label: labelOf(t), calls: 0, tris: 0, c0: info.calls, t0: info.triangles }; } }
    return res;
  };
}
// WebGL makes query results available only after control returns to the event loop, so this awaits a task.
async function readSegments(): Promise<Record<string, number>> {
  const out: Record<string, number> = {};
  if (!gl || !ext) return out;
  gl.finish();
  const list = segs; segs = [];
  for (let tries = 0; tries < 200; tries++) {
    await new Promise(r => setTimeout(r, 0));
    if (list.every(s => gl!.getQueryParameter(s.q, gl!.QUERY_RESULT_AVAILABLE))) break;
  }
  const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
  for (const s of list) {
    const ok = !!gl.getQueryParameter(s.q, gl.QUERY_RESULT_AVAILABLE);
    const ns = ok && !disjoint ? Number(gl.getQueryParameter(s.q, gl.QUERY_RESULT)) : NaN;
    out[s.label] = (out[s.label] ?? 0) + ns / 1e6; gl.deleteQuery(s.q);
  }
  return out;
}

// ---- stepped frames ---------------------------------------------------------------------------------------
function key(code: string, down: boolean) { window.dispatchEvent(new KeyboardEvent(down ? 'keydown' : 'keyup', { code, bubbles: true })); }
function runOneFrame(dtMs: number) {
  clock += dtMs; const batch = queue; queue = [];
  for (const e of batch) e.cb(clock);
}
async function frames(n: number, opts: { dtMs?: number; keys?: string[]; gpu?: boolean; warm?: number } = {}) {
  await booted; if (!gl) instrument();
  const dt = opts.dtMs ?? 1000 / 60; for (const k of opts.keys ?? []) key(k, true);
  const cpu: number[] = [], total: number[] = [], gpu: Record<string, number>[] = [];
  for (let i = 0; i < (opts.warm ?? 0); i++) { runOneFrame(dt); gl!.finish(); }
  for (let i = 0; i < n; i++) {
    const a = performance.now();
    if (opts.gpu) { timing = true; mark('pre'); }
    runOneFrame(dt);
    const b = performance.now();
    if (opts.gpu) { stopTiming(); timing = false; gl!.finish(); const c = performance.now(); cpu.push(b - a); total.push(c - a); gpu.push(await readSegments()); continue; }
    gl!.finish();
    const c = performance.now(); cpu.push(b - a); total.push(c - a);
  }
  for (const k of opts.keys ?? []) key(k, false);
  const stat = (v: number[]) => { if (!v.length) return null; const s = [...v].sort((x, y) => x - y); const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))]; return { p50: +q(.5).toFixed(2), p90: +q(.9).toFixed(2), max: +s[s.length - 1].toFixed(2), mean: +(v.reduce((x, y) => x + y, 0) / v.length).toFixed(2) }; };
  let gpuSummary: Record<string, number> | null = null;
  if (opts.gpu && gpu.length) {
    gpuSummary = {}; const keys = new Set(gpu.flatMap(g => Object.keys(g)));
    for (const k of keys) { const v = gpu.map(g => g[k] ?? 0).filter(Number.isFinite).sort((x, y) => x - y); gpuSummary[k] = v.length ? +v[Math.floor(v.length / 2)].toFixed(3) : NaN; }
    const sums = gpu.map(g => Object.values(g).filter(Number.isFinite).reduce((x, y) => x + y, 0)).sort((x, y) => x - y);
    gpuSummary.__sum_p50 = sums.length ? +sums[Math.floor(sums.length / 2)].toFixed(3) : NaN;
  }
  return { cpu: stat(cpu), total: stat(total), gpu: gpuSummary, info: view.info() };
}

async function capture(w = 1600, h = 1000) {
  // capture() reads performance.now() for the cloud drift and the probe clock: freeze it so every capture of a pose
  // sees the same sky
  await booted; frozenNow = 5e6; let blob: Blob;
  try { blob = await view.capture(w, h); } finally { frozenNow = null; }
  return await new Promise<string>(r => { const fr = new FileReader(); fr.onload = () => r(String(fr.result)); fr.readAsDataURL(blob); });
}

async function callBreakdown() {
  await booted; if (!gl) instrument();
  const renderer = view.renderer; counting = true; callSegs = []; callCur = null;
  // info is reset at the start of animate(); count from a fresh frame
  runOneFrame(1000 / 60);
  const info = renderer.info.render; if (callCur) callSegs.push({ label: callCur.label, calls: info.calls - callCur.c0, tris: info.triangles - callCur.t0 });
  counting = false; callCur = null; gl!.finish();
  const agg: Record<string, { calls: number; tris: number; n: number }> = {};
  for (const c of callSegs) { const a = agg[c.label] ?? (agg[c.label] = { calls: 0, tris: 0, n: 0 }); a.calls += c.calls; a.tris += c.tris; a.n++; }
  return { total: { calls: info.calls, tris: info.triangles }, bySegment: agg, order: callSegs.map(c => c.label + ':' + c.calls) };
}
// Like view.capture, but with a chosen set of post passes (capture itself forces occlusion and bloom on).
async function rawCapture(w: number, h: number, passes: Record<number, boolean>) {
  await booted; frozenNow = 5e6;
  try {
    const r = view.renderer, c = view.composer, cam = view.camera, a = view.atmosphere;
    const saved = c.passes.map((p: any) => p.enabled);
    for (const [i, on] of Object.entries(passes)) c.passes[Number(i)].enabled = on;
    r.setPixelRatio(1); r.setSize(w, h, false); c.setPixelRatio(1); c.setSize(w, h); c.passes[1].setSize(Math.round(w * .6), Math.round(h * .6));
    cam.aspect = w / h; cam.updateProjectionMatrix(); a.resize(); a.update(performance.now() + 1000); a.csm.lights[1].shadow.needsUpdate = true;
    c.render(1 / 60);
    const blob: Blob = await new Promise(res => r.domElement.toBlob((b: Blob) => res(b), 'image/png'));
    c.passes.forEach((p: any, i: number) => { p.enabled = saved[i]; });
    return await new Promise<string>(res => { const fr = new FileReader(); fr.onload = () => res(String(fr.result)); fr.readAsDataURL(blob); });
  } finally { frozenNow = null; }
}
(window as any).__bench = {
  rawCapture,
  callBreakdown, eval: (src: string) => (0, eval)(src),
  variant, booted: () => booted.then(() => ({ loadError, progress, heapMB: (performance as any).memory ? Math.round((performance as any).memory.usedJSHeapSize / 1e6) : null })),
  frames, capture, key,
  api: () => api, view: () => view,
  setStepped: (on: boolean) => { stepped = on; if (!on) { const b = queue; queue = []; for (const e of b) realRaf(e.cb); } },
  memory: () => { const r = view.renderer.info.memory; return { geometries: r.geometries, textures: r.textures, programs: view.renderer.info.programs?.length, heapMB: (performance as any).memory ? Math.round((performance as any).memory.usedJSHeapSize / 1e6) : null }; },
};
