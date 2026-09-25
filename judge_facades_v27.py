"""Facade judge: every photographed facade, model against photograph, across builds.

For each wall a cached street panorama sees well (the frontage module's camera choice), the photograph is rectified
square-on at 12 px/m and each build's own geometry is drawn square-on at the same scale (painter's order, back faces
culled as the viewer does). The two are compared on structure, which survives colour, light and texture:
- rows: where horizontal edges fall up the wall (floor lines, sills, bands, fascias, cornices);
- columns: where vertical edges fall along it (bays, piers, jambs);
- cells: a 1 m map of where detail is (blank walls against windowed ones), above the street clutter;
- false openings: the share of the model's openings that sit on wall the photograph shows blank;
- missing detail: the share of the photograph's detailed wall the model leaves plain.
Absolute scores carry the photograph's noise (occluders, sky, neighbours); the same photograph is used for every
build, so a drop between builds is a regression of the model. Nothing here names a place.

Usage: judge_facades_v27.py --walls <build dir> --build label=glb [...] --out <dir> [--max-walls N]
"""
import argparse, json, math, pathlib, re, sys, struct, mmap, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
from auto_read_frontages_v24 import rectify, PPM

OPENING = {'glass', 'glass_dark', 'glass_light', 'glass_blue', 'glass_clear', 'curtain', 'interior', 'recess', 'screen', 'window', 'opening', 'shopfront'}
LAYERS = ('buildings', 'facades', 'roofs')
STREET_M = 3.0          # street clutter (cars, people, kiosks) lives below this height in the photograph


def mesh_class(name):
    return name.split('___')[-1].split('#')[0].split('__')[0].split(' ')[0]


def load_build(path):
    """All buildings/facades/roofs triangles of a GLB in the scene frame (x east, y north, z up), with each
    triangle's class, colour and unit normal. Memory-light: the file is mapped, textures are only averaged."""
    f = open(path, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    ln = struct.unpack('<I', mm[12:16])[0]; j = json.loads(mm[20:20 + ln]); off = 20 + ln + 8
    CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}; NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    def acc(i):
        a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]; n = NC[a['type']]
        arr = np.frombuffer(mm, CT[a['componentType']], count=a['count'] * n, offset=off + bv.get('byteOffset', 0) + a.get('byteOffset', 0))
        return arr.reshape(a['count'], n) if n > 1 else arr
    def mat(node):
        if 'matrix' in node: return np.array(node['matrix'], float).reshape(4, 4).T
        T = np.eye(4)
        if 'rotation' in node:
            x, y, z, w = node['rotation']; T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        if 'scale' in node: T[:3, :3] = T[:3, :3] @ np.diag(node['scale'])
        if 'translation' in node: T[:3, 3] = node['translation']
        return T
    parent = {c: i for i, n in enumerate(j['nodes']) for c in n.get('children', [])}
    def world(i):
        M = mat(j['nodes'][i]); p = parent.get(i)
        return world(p) @ M if p is not None else M
    mesh_world = {}
    for i, n in enumerate(j['nodes']):
        if 'mesh' in n: mesh_world.setdefault(n['mesh'], (world(i), n.get('name', '')))
    colour_cache = {}
    def colour(mi):
        if mi in colour_cache: return colour_cache[mi]
        m = j['materials'][mi]; pb = m.get('pbrMetallicRoughness', {}); f = np.array(pb.get('baseColorFactor', [1, 1, 1, 1])[:3], float)
        c = np.ones(3)
        if 'baseColorTexture' in pb:
            try:
                img = j['images'][j['textures'][pb['baseColorTexture']['index']]['source']]; bv = j['bufferViews'][img['bufferView']]
                data = mm[off + bv.get('byteOffset', 0): off + bv.get('byteOffset', 0) + bv['byteLength']]
                c = np.asarray(Image.open(io.BytesIO(data)).convert('RGB').resize((32, 32)), float).mean((0, 1)) / 255.
                c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
            except Exception: pass
        lin = np.clip(f * c, 0, 1); srgb = np.where(lin <= .0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - .055)
        colour_cache[mi] = (srgb * 255).astype(np.uint8); return colour_cache[mi]
    Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    tris, cls, col = [], [], []; names = {}
    for mi, mesh in enumerate(j['meshes']):
        name = mesh.get('name', '')
        if name.split('__')[0] not in LAYERS: continue
        M, _ = mesh_world.get(mi, (np.eye(4), ''))
        c_name = mesh_class(name); cid = names.setdefault(c_name, len(names))
        for prim in mesh['primitives']:
            pos = acc(prim['attributes']['POSITION']).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
            idx = acc(prim['indices']).astype(np.int64) if 'indices' in prim else np.arange(len(pos))
            T = (pos @ Y2Z.T)[idx.reshape(-1, 3)].astype(np.float32)
            tris.append(T); cls.append(np.full(len(T), cid, np.int16)); col.append(np.tile(colour(prim['material']) if 'material' in prim else np.array([200, 200, 200], np.uint8), (len(T), 1)))
    T = np.concatenate(tris); C = np.concatenate(cls); K = np.concatenate(col)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    opening = np.array([n in OPENING for n in sorted(names, key=names.get)])
    return {'T': T, 'cls': C, 'col': K, 'nrm': nrm.astype(np.float32), 'opening': opening, 'names': names, 'kd': cKDTree(T.mean(1)[:, :2])}


def render_wall(build, wall, z0, H, L):
    """The build's geometry in front of and on the wall plane, drawn square-on: colour image and opening mask."""
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
    W = max(8, int(L * PPM)); Hp = max(8, int(H * PPM))
    idx = build['kd'].query_ball_point(m, L / 2 + 6)
    img = Image.new('RGB', (W, Hp), (236, 238, 240)); msk = Image.new('L', (W, Hp), 0); edge = Image.new('L', (W, Hp), 0)
    if not idx: return np.asarray(img), np.zeros((Hp, W), bool), np.zeros((Hp, W), bool)
    idx = np.asarray(idx); T = build['T'][idx].astype(np.float64); nr = build['nrm'][idx]
    rel = T - np.r_[m, z0]
    u = rel[..., :2] @ t + L / 2; d = rel[..., :2] @ n; zz = rel[..., 2]
    facing = nr[:, :2] @ n
    keep = (facing > .05) & (d.mean(1) > -2.5) & (d.mean(1) < 4.) & (u.max(1) > -.5) & (u.min(1) < L + .5) & (zz.max(1) > -.5) & (zz.min(1) < H + .5)
    order = np.flatnonzero(keep)[np.argsort(d.mean(1)[keep])]          # far to near; the viewer culls back faces
    dr = ImageDraw.Draw(img); dm = ImageDraw.Draw(msk)
    px = u * PPM; py = (H - zz) * PPM
    for i in order:
        shade = .55 + .45 * min(1., facing[i]); c = tuple(int(v * shade) for v in build['col'][idx[i]])
        poly = [(px[i, k], py[i, k]) for k in range(3)]
        dr.polygon(poly, fill=c); dm.polygon(poly, fill=255 if build['opening'][build['cls'][idx[i]]] else 1)
    a = np.asarray(img); mk = np.asarray(msk)
    return a, mk == 255, mk > 0


def structure(gray, sigma=1.):
    g = ndi.gaussian_filter(gray, sigma); gx = ndi.sobel(g, 1); gy = ndi.sobel(g, 0)
    s = np.percentile(np.hypot(gx, gy), 95) + 1e-6
    return np.clip(np.abs(gx) / s, 0, 1), np.clip(np.abs(gy) / s, 0, 1)


def best_corr(a, b, shift):
    a = ndi.gaussian_filter1d(a, 5.); b = ndi.gaussian_filter1d(b, 5.); best = -1.
    for s in range(-shift, shift + 1):
        x = a[max(0, s): len(a) + min(0, s)]; y = b[max(0, -s): len(b) + min(0, -s)]
        if len(x) < 10 or x.std() < 1e-6 or y.std() < 1e-6: continue
        best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def judge(photo, render, opening, covered, H):
    """Structure agreement of a rectified photograph and a square-on render of the same wall (both 12 px/m)."""
    Hp, W = opening.shape; street = int(STREET_M * PPM)
    upper = slice(0, max(1, Hp - street))
    pgx, pgy = structure(photo.mean(2) * 255 if photo.max() <= 1.01 else photo.mean(2))
    rgx, rgy = structure(render.mean(2).astype(float))
    cov = covered.astype(float)
    rows = best_corr(pgy[upper].mean(1), rgy[upper].mean(1), 6)
    cols = best_corr(pgx[upper].mean(0), rgx[upper].mean(0), 6)
    k = int(PPM)                                  # 1 m cells
    hc, wc = Hp // k, W // k
    if hc < 2 or wc < 2: return None
    def cells(a): return a[:hc * k, :wc * k].reshape(hc, k, wc, k).mean((1, 3))
    dp = cells(np.maximum(pgx, pgy)); dm = cells(np.maximum(rgx, rgy)); op = cells(opening.astype(float)); cv = cells(cov)
    up = np.zeros((hc, wc), bool); up[: max(1, hc - int(STREET_M))] = True
    use = up & (cv > .5)
    cellcorr = float(np.corrcoef(dp[use], dm[use])[0, 1]) if use.sum() >= 6 and dp[use].std() > 1e-6 and dm[use].std() > 1e-6 else 0.
    blank = use & (dp < .09); detailed = use & (dp > .22)
    false_open = float((op * blank).sum() / max(op[use].sum(), 1e-6)) if op[use].sum() > .5 else 0.
    missing = float((detailed & (op < .05) & (dm < .06)).sum() / max(detailed.sum(), 1)) if detailed.sum() >= 3 else 0.
    score = .3 * rows + .3 * cols + .4 * cellcorr - .5 * false_open - .3 * missing
    return {'score': round(score, 3), 'rows': round(rows, 3), 'cols': round(cols, 3), 'cells': round(cellcorr, 3),
            'falseOpenings': round(false_open, 3), 'missingDetail': round(missing, 3), 'modelOpeningShare': round(float(op[use].mean()) if use.any() else 0., 3),
            'coverage': round(float(cv[up].mean()), 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--walls', type=pathlib.Path, required=True); ap.add_argument('--build', action='append', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True); ap.add_argument('--max-walls', type=int, default=200); a = ap.parse_args()
    out = a.out; (out / 'sheets').mkdir(parents=True, exist_ok=True)
    world = json.load(open(a.walls / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    ev = json.load(open(a.walls / 'public/facade-evidence.json'))
    walls = [dict(w, source='envelope' if w.get('status') == 'envelope-graph' else 'wall') for w in ev['walls'] if w.get('buildingId') in buildings]
    walls = [w for w in walls if w.get('lengthM', 0) >= 6 and w.get('heightM', 0) >= 3]
    occ = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    best = {}
    for w in walls:
        cands = fc.wall_candidates(w, cams, occ, trees, fc.MAX_DISTANCE_M)
        if not cands: continue
        c = cands[0]; key = (re.sub(r'_p\d+$', '', w['buildingId']), int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])) / 10)))
        s = c['score'] * w['lengthM']
        if key not in best or s > best[key][0]: best[key] = (s, w, c)
    chosen = sorted(best.values(), key=lambda x: -x[0])[:a.max_walls]
    print('walls judged', len(chosen), 'of', len(walls), flush=True)
    # Photographs first (one panorama cache), then each build in turn (one model in memory at a time).
    photos = []; pano_cache = {}
    for i, (_, w, c) in enumerate(chosen):
        cam = cams[c['image']]; bid = w['buildingId']; z0 = buildings[bid]['center'][2]
        if c['image'] not in pano_cache:
            if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
            pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
            road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
            gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
            pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
        img, H, L = rectify(w, cam, z0, pano_cache[c['image']])
        key = f"{i:03d}_{re.sub(r'[^A-Za-z0-9_]', '', bid)}"
        photos.append({'key': key, 'wall': w, 'camera': {'image': c['image'], 'distanceM': round(c['distanceM'], 1), 'incidence': round(c['incidence'], 2)}, 'z0': z0, 'H': H, 'L': L, 'photo': (img * 255).astype(np.uint8)})
    labels = [b.split('=', 1)[0] for b in a.build]; renders = {p['key']: {} for p in photos}; results = {p['key']: {} for p in photos}
    for spec in a.build:
        label, path = spec.split('=', 1); build = load_build(path); print('build', label, len(build['T']), 'triangles', flush=True)
        for p in photos:
            r, op, cov = render_wall(build, p['wall'], p['z0'], p['H'], p['L'])
            renders[p['key']][label] = r; results[p['key']][label] = judge(p['photo'].astype(float), r, op, cov, p['H'])
        del build
    # Sheets: photograph, then each build, scores under each.
    try: font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except Exception: font = ImageFont.load_default()
    rows = []
    for p in photos:
        tiles = [('photo', p['photo'])] + [(l, renders[p['key']][l]) for l in labels]
        h = p['photo'].shape[0]; scale = min(1., 300 / max(h, 1)); tw = max(60, int(p['photo'].shape[1] * scale)); th = max(40, int(h * scale))
        sheet = Image.new('RGB', (len(tiles) * (tw + 8) + 8, th + 64), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
        for k, (l, im) in enumerate(tiles):
            x = 8 + k * (tw + 8); sheet.paste(Image.fromarray(im).resize((tw, th)), (x, 26))
            r = results[p['key']].get(l); txt = l if r is None or l == 'photo' else f"{l}  S {r['score']:+.2f}  false {r['falseOpenings']:.2f}  miss {r['missingDetail']:.2f}"
            dr.text((x, 6), txt, fill=(20, 20, 20), font=font)
        dr.text((8, th + 32), f"{p['key']}  {p['wall']['buildingId']}  {p['L']:.0f} x {p['H']:.0f} m  camera {p['camera']['distanceM']} m  source {p['wall']['source']}", fill=(90, 90, 90), font=font)
        sheet.save(out / 'sheets' / f"{p['key']}.jpg", quality=85)
        rows.append({'key': p['key'], 'buildingId': p['wall']['buildingId'], 'source': p['wall']['source'], 'lengthM': round(p['L'], 1), 'heightM': round(p['H'], 1), 'midpoint': p['wall']['midpoint'], 'normal': p['wall']['normal'], 'camera': p['camera'], 'results': results[p['key']]})
    summary = {}
    for l in labels:
        s = [r['results'][l]['score'] for r in rows if r['results'].get(l)]
        summary[l] = {'walls': len(s), 'meanScore': round(float(np.mean(s)), 3) if s else None,
                      'meanFalseOpenings': round(float(np.mean([r['results'][l]['falseOpenings'] for r in rows if r['results'].get(l)])), 3) if s else None,
                      'meanMissingDetail': round(float(np.mean([r['results'][l]['missingDetail'] for r in rows if r['results'].get(l)])), 3) if s else None}
    last = labels[-1]; regressions = []
    for r in rows:
        cur = r['results'].get(last)
        if not cur: continue
        prev = [(l, r['results'][l]['score']) for l in labels[:-1] if r['results'].get(l)]
        if not prev: continue
        bl, bs = max(prev, key=lambda x: x[1])
        if cur['score'] < bs - .1: regressions.append({'key': r['key'], 'buildingId': r['buildingId'], 'bestBuild': bl, 'bestScore': bs, 'current': cur['score'], 'drop': round(bs - cur['score'], 3)})
    regressions.sort(key=lambda x: -x['drop'])
    json.dump({'builds': labels, 'summary': summary, 'regressions': regressions, 'walls': rows}, open(out / 'judge.json', 'w'), indent=1)
    print(json.dumps({'summary': summary, 'regressionsVsBestEarlier': len(regressions)}, indent=1))


if __name__ == '__main__': main()
