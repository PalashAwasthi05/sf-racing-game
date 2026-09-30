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
from panorama_balance import road_gain
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
from auto_read_frontages_v24 import rectify, PPM, read as periodic_read
from scipy.ndimage import map_coordinates


def rectify_at(wall, cam, z0, pano, ppm):
    """The frontage module's square-on rectification at a chosen resolution (the reader's is fixed at 12 px/m)."""
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
    L = float(wall['lengthM']); H = float(wall['heightM']); W = max(8, int(L * ppm)); Hp = max(8, int(H * ppm))
    xs = (np.arange(W) + .5) / ppm - L / 2; zs = H - (np.arange(Hp) + .5) / ppm
    P = np.empty((Hp, W, 3)); P[..., 0] = (m[0] + t[0] * xs)[None, :] + n[0] * .05; P[..., 1] = (m[1] + t[1] * xs)[None, :] + n[1] * .05; P[..., 2] = (z0 + zs)[:, None]
    C = np.array([cam['xy'][0], cam['xy'][1], z0 + fc.EYE_HEIGHT_M]); u, v = fc.world_dirs_to_pano_uv(P - C, cam['R']); ph, pw = pano.shape[:2]
    return np.stack([map_coordinates(pano[..., k], [v * (ph - 1), u * (pw - 1)], order=1, mode='wrap', output=np.float32) for k in range(3)], -1)

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


def to_lab(img):
    c = img / 255.; c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    X = c @ np.array([[.4124, .2126, .0193], [.3576, .7152, .1192], [.1805, .0722, .9505]]) / np.array([.9505, 1., 1.089])
    f = np.where(X > .008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def photo_openings(photo):
    """Openings in a rectified photograph: roughly rectangular blobs of window size whose colour departs from the
    wall's (in Lab, luminance down-weighted so shadows, which keep the wall's hue, do not count; glass reflecting sky
    or showing a dark interior does). Foliage and sky are excluded. The wall colour is the median of the upper wall's
    least-detailed pixels. Returns (opening mask, usable mask, wall colour RGB)."""
    img = photo if photo.max() > 1.01 else photo * 255.
    r, g, b = img[..., 0], img[..., 1], img[..., 2]; lum = .299 * r + .587 * g + .114 * b
    mx = img.max(2); mn = img.min(2); sat = (mx - mn) / np.maximum(mx, 1)
    foliage = (g > r * 1.04) & (g > b * 1.02) & (sat > .12)
    Hp = img.shape[0]; top = max(1, Hp // 8); sky = np.zeros(lum.shape, bool); sky[:top] = (b[:top] > r[:top] + 8) & (lum[:top] > 150)
    gx, gy = structure(lum); calm = ndi.uniform_filter(np.maximum(gx, gy), 9) < .12
    wall_px = img[calm & ~foliage & ~sky]
    wall = np.median(wall_px, 0) if len(wall_px) > 200 else np.median(img.reshape(-1, 3), 0)
    lab = to_lab(ndi.uniform_filter(img, (3, 3, 1))); wl = to_lab(wall[None, None, :].astype(float))[0, 0]
    d = np.sqrt(((lab[..., 0] - wl[0]) * .45) ** 2 + (lab[..., 1] - wl[1]) ** 2 + (lab[..., 2] - wl[2]) ** 2)
    cand = ndi.binary_opening((d > 16) & ~foliage & ~sky, iterations=1)
    # Framed openings: panes enclosed by frame and reveal edges, whatever their colour against the wall.
    edges = ndi.binary_closing(np.maximum(gx, gy) > .35, iterations=1)
    panes = ndi.binary_fill_holes(edges) & ~edges
    cand = (cand | ndi.binary_opening(panes, iterations=1)) & ~foliage & ~sky
    lab_i, n = ndi.label(cand); keep = np.zeros(n + 1, bool)
    for k, sl in enumerate(ndi.find_objects(lab_i), 1):
        h = (sl[0].stop - sl[0].start) / PPM; w = (sl[1].stop - sl[1].start) / PPM; area = (lab_i[sl] == k).sum() / PPM ** 2
        keep[k] = .3 <= area <= 30 and h <= 6 and w <= 12 and area / max(h * w, 1e-6) >= .5      # a window, not a tree or a neighbour
    return keep[lab_i], ~foliage & ~sky, wall


def labelled_openings(codes):
    """Opening, usable and sign masks from segmentation group codes (facade_segment_v27.CODES). A sign board or a
    pylon hides the wall behind it, so it is not usable for comparing openings."""
    return codes == 1, (codes != 3) & (codes != 4) & (codes != 2), codes == 2


def judge(photo, render, opening, covered, H, codes=None, grid_share=None):
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
    popen, usable, wall = photo_openings(photo)
    if codes is not None:popen, usable, _ = labelled_openings(codes)
    # Register the photograph to the model on the opening maps (camera position and heading carry a metre or so of
    # error), within +-1.5 m along and +-1 m up; then compare at 1 m with 0.6 m of blur.
    region = np.zeros_like(opening); region[:Hp - street] = True
    Pm = ndi.gaussian_filter((popen & usable & region).astype(float), 7.); Mm = ndi.gaussian_filter((opening & covered & region).astype(float), 7.)
    best = (-2., 0, 0)
    for dy in range(-12, 13, 3):
        for dx in range(-18, 19, 3):
            A = np.roll(np.roll(Pm, dy, 0), dx, 1)[18:-18 or None, 12:-12 or None] if Hp > 40 and W > 60 else Pm
            B = Mm[18:-18 or None, 12:-12 or None] if Hp > 40 and W > 60 else Mm
            if A.std() > 1e-6 and B.std() > 1e-6:
                c = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
                if c > best[0]: best = (c, dy, dx)
    shift = best[1:]; Pr = np.roll(np.roll(popen, shift[0], 0), shift[1], 1); Ur = np.roll(np.roll(usable, shift[0], 0), shift[1], 1)
    pp = cells(ndi.gaussian_filter(Pr.astype(float), 7.)); opb = cells(ndi.gaussian_filter(opening.astype(float), 7.))
    ok = use & (cells(Ur.astype(float)) > .6)
    # Openings against openings, metre by metre: model openings on wall the photograph shows solid, photographed
    # openings the model leaves solid, and the overall window share.
    solid = ok & (pp < .05)
    false_open = float((opb * solid).sum() / max((opb * ok).sum(), 1e-6)) if (opb * ok).sum() > .5 else 0.
    missing = float((pp * ok * (opb < .04)).sum() / max((pp * ok).sum(), 1e-6)) if (pp * ok).sum() > .5 else 0.
    # The detector and the facade reader share a model; the periodic-edge grid is an independent opinion of how much
    # of the wall is window. Where it is confident, the photographed share is at least what that grid implies.
    photo_share = float((cells(Pr.astype(float)) * ok).sum() / max(ok.sum(), 1))
    if grid_share is not None: photo_share = max(photo_share, grid_share)
    share_err = abs(float((op * ok).sum() / max(ok.sum(), 1)) - photo_share)
    opencorr = float(np.corrcoef(pp[ok], opb[ok])[0, 1]) if ok.sum() >= 6 and pp[ok].std() > 1e-6 and opb[ok].std() > 1e-6 else 0.
    mw = render[:Hp - street][~opening[:Hp - street]] if (~opening[:Hp - street]).any() else render.reshape(-1, 3)
    colour = float(np.linalg.norm(np.median(mw.reshape(-1, 3), 0).astype(float) - wall) / 441.7)
    # Windows where the photograph shows wall and wall where it shows windows weigh the same: blanking a facade is
    # not rewarded over drawing it wrongly.
    score = .15 * rows + .15 * cols + .35 * opencorr - .3 * false_open - .3 * missing - .4 * share_err - .2 * colour
    reliable = float((cells(Ur.astype(float))[use] > .6).mean()) if use.any() else 0.
    return {'score': round(score, 3), 'rows': round(rows, 3), 'cols': round(cols, 3), 'cells': round(cellcorr, 3),
            'falseOpenings': round(false_open, 3), 'missingDetail': round(missing, 3), 'openingCorr': round(opencorr, 3), 'openingShareError': round(share_err, 3),
            'colourError': round(colour, 3), 'modelOpeningShare': round(float(op[ok].mean()) if ok.any() else 0., 3), 'photoOpeningShare': round(float(cells(Pr.astype(float))[ok].mean()) if ok.any() else 0., 3),
            'registrationM': [round(shift[1] / PPM, 2), round(-shift[0] / PPM, 2)], 'visibleShare': round(reliable, 3), 'reliable': reliable >= .4,
            'coverage': round(float(cv[up].mean()), 3)}


def held_out_sources(build_specs):
    """For --holdout: a function giving, for a judged wall, every photograph any build's facade evidence lists as a
    source of that wall (same building, facing within 25 degrees, overlapping along the frontage). A reading is not
    judged against the photograph it was read from."""
    import collections
    by = collections.defaultdict(list)
    for spec in build_specs:
        d = pathlib.Path(spec.split('=', 1)[1]).resolve().parent
        while d != d.parent and not (d / 'public/facade-evidence.json').exists(): d = d.parent
        if not (d / 'public/facade-evidence.json').exists(): raise SystemExit(f'--holdout: no facade evidence above {spec}')
        for w in json.load(open(d / 'public/facade-evidence.json'))['walls']:
            ids = [s_ for s_ in w.get('sourceIds', []) if str(s_).startswith('mapillary:')]
            if ids: by[re.sub(r'_p\d+$', '', w['buildingId'])].append((np.asarray(w['normal'], float), np.asarray(w['midpoint'], float), float(w['lengthM']), ids))
    def used(wall):
        n = np.asarray(wall['normal'], float); m = np.asarray(wall['midpoint'], float); out = set()
        for en, em, el, ids in by.get(re.sub(r'_p\d+$', '', wall['buildingId']), []):
            if np.dot(n, en) / (np.linalg.norm(n) * np.linalg.norm(en)) > .9 and np.linalg.norm(m - em) < (el + float(wall['lengthM'])) / 2 + 3: out.update(ids)
        return out
    return used


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--walls', type=pathlib.Path, required=True); ap.add_argument('--build', action='append', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True); ap.add_argument('--max-walls', type=int, default=200)
    ap.add_argument('--all-cameras', action='store_true', help='also judge from panoramas fetched later (changes the photographs, so results stop being comparable with earlier judgements)')
    ap.add_argument('--holdout', action='store_true', help='judge each wall from the best photograph that no build read it from (all panoramas; implies --all-cameras)'); a = ap.parse_args()
    if a.holdout: a.all_cameras = True
    out = a.out; (out / 'sheets').mkdir(parents=True, exist_ok=True)
    world = json.load(open(a.walls / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    ev = json.load(open(a.walls / 'public/facade-evidence.json'))
    walls = [dict(w, source='envelope' if w.get('status') == 'envelope-graph' else 'wall') for w in ev['walls'] if w.get('buildingId') in buildings]
    walls = [w for w in walls if w.get('lengthM', 0) >= 6 and w.get('heightM', 0) >= 3]
    # A long frontage is judged in pieces of about 30 m, each from its own best camera (one panorama cannot see a
    # 150 m wall square-on).
    pieces = []
    for w in walls:
        L = float(w['lengthM']); k = max(1, int(round(L / 30.)))
        if k == 1: pieces.append(w); continue
        n = np.asarray(w['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]]); m = np.asarray(w['midpoint'], float)
        for i in range(k):
            c = m + t * ((i + .5) / k - .5) * L
            pieces.append(dict(w, midpoint=c.tolist(), lengthM=L / k, piece=f'{i + 1}/{k}'))
    walls = pieces
    occ = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    # Every release is judged against the same photographs: panoramas fetched later (which the readers may draw from)
    # would otherwise replace a wall's judging photograph and its cached reading, and the numbers would stop being
    # comparable from one judgement to the next.
    if not a.all_cameras: cams = {k: c for k, c in cams.items() if not c.get('batch')}
    used = held_out_sources(a.build) if a.holdout else None
    best = {}
    for w in walls:
        cands = fc.wall_candidates(w, cams, occ, trees, fc.MAX_DISTANCE_M)
        if used is not None: skip = used(w); cands = [c for c in cands if 'mapillary:' + c['image'] not in skip]
        if not cands: continue
        c = cands[0]; key = (re.sub(r'_p\d+$', '', w['buildingId']), int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])) / 10)), w.get('piece', ''))
        s = c['score'] * w['lengthM']
        if key not in best or s > best[key][0]: best[key] = (s, w, c)
    chosen = sorted(best.values(), key=lambda x: -x[0])[:a.max_walls]
    print('walls judged', len(chosen), 'of', len(walls), flush=True)
    # Photographs first (one panorama cache), then each build in turn (one model in memory at a time).
    photos = []; pano_cache = {}
    try:
        from facade_segment_v27 import Segmenter, Detector
        seg = Segmenter(); det = Detector(); print('photo reading: SegFormer-B2 ADE20K (occluders, sky) + Grounding DINO (openings, signs) on', seg.device, flush=True)
    except Exception as e:
        seg = det = None; print('models unavailable, colour/frame detector only:', str(e)[:120], flush=True)
    cache = out / 'readings'; cache.mkdir(exist_ok=True)
    for i, (_, w, c) in enumerate(chosen):
        cam = cams[c['image']]; bid = w['buildingId']; z0 = buildings[bid]['center'][2]
        if c['image'] not in pano_cache:
            if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
            pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
            gain = road_gain(pano)   # the road below the horizon, not the capture vehicle (panorama_balance)
            pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
        img, H, L = rectify(w, cam, z0, pano_cache[c['image']])
        key = f"{i:03d}_{re.sub(r'[^A-Za-z0-9_]', '', bid)}"
        codes = None; cfile = cache / f"{key}_{c['image']}.npz"; gfile = cache / f"{key}_{c['image']}_glass.json"
        if seg is not None and cfile.exists(): codes = np.load(cfile)['codes'].copy()
        elif seg is not None:
            # Group codes at 12 px/m: occluders and sky from the segmentation; openings are detector window and door
            # boxes plus the colour/frame blobs (each catches what the other misses); signs and awnings from boxes.
            u8 = (img * 255).astype(np.uint8); codes = seg.labels(u8).copy()
            codes[codes == 1] = 0; codes[codes == 2] = 0
            po, _, _ = photo_openings(u8.astype(float))
            for x0, y0, x1, y1, kind, sc in det.boxes(u8):
                sl = (slice(int(max(0, y0)), int(min(img.shape[0], y1))), slice(int(max(0, x0)), int(min(img.shape[1], x1))))
                if kind in ('window', 'door') and (x1 - x0) * (y1 - y0) / PPM ** 2 <= 40: codes[sl] = np.where(codes[sl] == 3, 3, 1)
                elif kind in ('sign', 'awning'): codes[sl] = np.where(codes[sl] == 0, 2, codes[sl])
            codes[po & (codes == 0)] = 1
            np.savez_compressed(cfile, codes=codes)
        if seg is not None:
            # Curtain-wall glazing (one large object, found at whole-wall scale) is opening wherever it is not hidden.
            from facade_segment_v27 import glazed_boxes
            if gfile.exists(): glass = json.loads(gfile.read_text())
            else: glass = [list(map(float, g)) for g in glazed_boxes(det, (img * 255).astype(np.uint8))]; gfile.write_text(json.dumps(glass))
            for x0, y0, x1, y1, sc in glass:
                # Glazing detected at whole-wall scale is not trusted as openings (it fires on dark and bushy walls too):
                # the judge abstains there instead of scoring for or against the model.
                if sc < .4: continue
                sl = (slice(int(max(0, y0)), int(min(img.shape[0], y1))), slice(int(max(0, x0)), int(min(img.shape[1], x1))))
                codes[sl] = np.where(codes[sl] == 0, 3, codes[sl])
        pr = periodic_read(img, H, L)
        grid_share = float(pr['openingRatio'] * pr['openingHeightRatio']) if pr.get('floorPitchM') and pr.get('gridConfidence', 0) >= .2 else None
        photos.append({'gridShare': grid_share, 'codes': codes, 'key': key, 'wall': w, 'camera': {'image': c['image'], 'distanceM': round(c['distanceM'], 1), 'incidence': round(c['incidence'], 2)}, 'z0': z0, 'H': H, 'L': L, 'photo': (img * 255).astype(np.uint8)})
    labels = [b.split('=', 1)[0] for b in a.build]; renders = {p['key']: {} for p in photos}; results = {p['key']: {} for p in photos}
    for spec in a.build:
        label, path = spec.split('=', 1); build = load_build(path); print('build', label, len(build['T']), 'triangles', flush=True)
        for p in photos:
            r, op, cov = render_wall(build, p['wall'], p['z0'], p['H'], p['L'])
            renders[p['key']][label] = r; results[p['key']][label] = judge(p['photo'].astype(float), r, op, cov, p['H'], p['codes'], p.get('gridShare'))
        del build
    # Sheets: photograph, then each build, scores under each.
    try: font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except Exception: font = ImageFont.load_default()
    rows = []
    for p in photos:
        if p['codes'] is not None: po, pu, ps = labelled_openings(p['codes'])
        else: po, pu, _ = photo_openings(p['photo'].astype(float)); ps = np.zeros_like(po)
        ov = p['photo'].copy(); ov[po] = (ov[po] * .35 + np.array([255, 40, 40]) * .65).astype(np.uint8); ov[ps] = (ov[ps] * .35 + np.array([255, 210, 0]) * .65).astype(np.uint8); ov[~pu] = (ov[~pu] * .4).astype(np.uint8)
        tiles = [('photo', p['photo']), ('openings read', ov)] + [(l, renders[p['key']][l]) for l in labels]
        h = p['photo'].shape[0]; scale = min(1., 300 / max(h, 1)); tw = max(60, int(p['photo'].shape[1] * scale)); th = max(40, int(h * scale))
        sheet = Image.new('RGB', (len(tiles) * (tw + 8) + 8, th + 64), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
        for k, (l, im) in enumerate(tiles):
            x = 8 + k * (tw + 8); sheet.paste(Image.fromarray(im).resize((tw, th)), (x, 26))
            r = results[p['key']].get(l); txt = l if r is None or l in ('photo', 'openings read') else (f"{l}  S {r['score']:+.2f}  false {r['falseOpenings']:.2f}  miss {r['missingDetail']:.2f}  share {r['modelOpeningShare']:.2f}/{r['photoOpeningShare']:.2f}" + ('' if r['reliable'] else '  (photo mostly hidden: not scored)'))
            dr.text((x, 6), txt, fill=(20, 20, 20), font=font)
        dr.text((8, th + 32), f"{p['key']}  {p['wall']['buildingId']}  {p['L']:.0f} x {p['H']:.0f} m  camera {p['camera']['distanceM']} m  source {p['wall']['source']}", fill=(90, 90, 90), font=font)
        sheet.save(out / 'sheets' / f"{p['key']}.jpg", quality=85)
        rows.append({'key': p['key'], 'buildingId': p['wall']['buildingId'], 'piece': p['wall'].get('piece'), 'source': p['wall']['source'], 'lengthM': round(p['L'], 1), 'heightM': round(p['H'], 1), 'midpoint': p['wall']['midpoint'], 'normal': p['wall']['normal'], 'camera': p['camera'], 'results': results[p['key']]})
    summary = {}
    for l in labels:
        use_r = [r for r in rows if r['results'].get(l) and r['results'][l]['reliable'] and r['lengthM'] * r['heightM'] >= 60]
        s = [r['results'][l]['score'] for r in use_r]; wts = [r['lengthM'] * r['heightM'] for r in use_r]
        summary[l] = {'walls': len(s), 'meanScore': round(float(np.average(s, weights=wts)), 3) if s else None,
                      'meanFalseOpenings': round(float(np.mean([r['results'][l]['falseOpenings'] for r in rows if r['results'].get(l) and r['results'][l]['reliable']])), 3) if s else None,
                      'meanMissingDetail': round(float(np.mean([r['results'][l]['missingDetail'] for r in rows if r['results'].get(l) and r['results'][l]['reliable']])), 3) if s else None}
    last = labels[-1]; regressions = []
    for r in rows:
        cur = r['results'].get(last)
        if not cur or not cur['reliable'] or r['lengthM'] * r['heightM'] < 60: continue
        prev = [(l, r['results'][l]['score']) for l in labels[:-1] if r['results'].get(l)]
        if not prev: continue
        bl, bs = max(prev, key=lambda x: x[1])
        if cur['score'] < bs - .1: regressions.append({'key': r['key'], 'buildingId': r['buildingId'], 'bestBuild': bl, 'bestScore': bs, 'current': cur['score'], 'drop': round(bs - cur['score'], 3)})
    regressions.sort(key=lambda x: -x['drop'])
    # The gate: the last build against the one before it, wall by wall (reliable walls of at least 60 m2). A drop of
    # more than 0.05 is a regression to explain or fix before release.
    gate = {'improved': [], 'regressed': [], 'unchanged': 0}
    if len(labels) >= 2:
        prev = labels[-2]
        for r in rows:
            a_, b_ = r['results'].get(prev), r['results'].get(last)
            if not a_ or not b_ or not (a_['reliable'] and b_['reliable']) or r['lengthM'] * r['heightM'] < 60: continue
            d = round(b_['score'] - a_['score'], 3)
            if d > .05: gate['improved'].append({'key': r['key'], 'delta': d})
            elif d < -.05: gate['regressed'].append({'key': r['key'], 'delta': d})
            else: gate['unchanged'] += 1
        gate['improved'].sort(key=lambda x: -x['delta']); gate['regressed'].sort(key=lambda x: x['delta'])
    json.dump({'builds': labels, 'summary': summary, 'regressions': regressions, 'gate': gate, 'walls': rows}, open(out / 'judge.json', 'w'), indent=1)
    print(json.dumps({'summary': summary, 'regressionsVsBestEarlier': len(regressions), 'gate': {k: (len(v) if isinstance(v, list) else v) for k, v in gate.items()}}, indent=1))


if __name__ == '__main__': main()
