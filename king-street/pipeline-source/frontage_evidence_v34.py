"""Evidence rules for frontage reading v34.

Occlusion by buildings. A photograph of a wall often shows another building (or a lower or taller part of the same
one) in front of part of it; the reader then took that building's facade for the wall (a tall block read as the pink
brick of the low building before it, from a photograph in which the low building hid most of it). The LiDAR says
where solid mass stands: the highest return per 0.5 m cell inside a building footprint (this survey labels buildings
and trees alike as unclassified; the footprints separate them, and street trees stay with the segmenter's tree mask),
kept where six of the nine cells around it hold returns (posts, masts and branches are single cells).
A wall point is hidden when the camera's ray to it passes under that surface somewhere between one metre from the
camera and 1.5 m before the wall (the wall's own face and anything fixed to it are not in front of it), over at
least a metre of the wall's width. The camera stands 2.4 m above the LiDAR ground under it.
`hidden_mask` gives that mask on the rectified wall grid of auto_read_frontages_v24.rectify.

Walls the building does not have. Where the LiDAR mass directly behind an evidence wall stays well below the wall's
top, the wall is not this building's face (`unbacked_mask`): an outline carried to a corner where only a lower building
stands had a 15 m "wall" backed by a 9.5 m roof, and every photograph of it showed the buildings beyond; its reading
(their brick) was attached to the tall building. Those columns count as hidden.

Regular layouts. Detected windows are boxes of every width at every position (palm trunks and sign frames among
them); drawn as they came they were the scatter of odd windows. `regular_layout` drops slivers (under 0.45 m wide,
under 0.35 m2, or eight times higher than wide; a wide low ribbon of transom lights stays), groups the rest into storeys (common sill and head), fits each
storey's bay pitch and phase where three or more windows support one (85 % within 15 % of the pitch: a regular
facade read with noise, not windows in pairs and groups), shares one grid between storeys whose pitches agree within
8 % and whose phases agree, snaps windows to it, gives windows one width where their widths already agree (spread
under 25 %) and each storey one height, and merges windows that then overlap. Nothing is invented: a window is only moved to the bay the
detections imply, and storeys without a pitch keep their positions (their widths are unified).
"""
import json, math, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
SURFACE_PATH = ROOT / 'lidar-building-surface-v34.npz'
STEP, NEAR_M, FAR_M, MARGIN_M, PPM = .5, 1., 1.5, .3, 12.


def building_surface(world, frame, rebuild=False):
    """(top, (x0, y1, step)): highest unclassified LiDAR return per cell inside a building footprint, scene frame
    (x east, y north, z above the world's vertical origin); NaN where no building stands. Cached."""
    if SURFACE_PATH.exists() and not rebuild:
        d = np.load(SURFACE_PATH); return d['top'], tuple(float(v) for v in d['g'])
    import shapely
    from shapely.ops import unary_union
    from pyproj import Transformer
    import frontage_crops_v15 as fc
    pts = np.load(ROOT / 'data/points.npz')['points']; P = pts[pts[:, 3] == 1]; del pts
    px, py = Transformer.from_crs(3857, 32610, always_xy=True).transform(P[:, 0], P[:, 1])
    x = px - frame.cx; y = py - frame.cy; z = P[:, 2] - float(world['verticalOrigin'])
    fp = unary_union([g.buffer(-.3) for g in fc.load_footprints(frame).values() if not g.is_empty])
    inside = shapely.contains_xy(fp, x, y); x, y, z = x[inside], y[inside], z[inside]
    x0, y1 = math.floor(x.min()) - 2., math.ceil(y.max()) + 2.
    nx, ny = int((x.max() + 2 - x0) / STEP) + 1, int((y1 - y.min() + 2) / STEP) + 1
    flat = ((y1 - y) / STEP).astype(int) * nx + ((x - x0) / STEP).astype(int)
    top = np.full(nx * ny, -np.inf); np.maximum.at(top, flat, z); top = top.reshape(ny, nx)
    # Solid mass only: a cell stands when at least 6 of its 3 x 3 neighbourhood hold returns, at the neighbourhood's
    # median height. Posts, masts, antennas and branches (single cells) were thin stripes of false occlusion.
    from scipy import ndimage as ndi
    have = np.isfinite(top); vals = np.where(have, top, np.nan)
    count = ndi.uniform_filter(have.astype(float), 3) * 9
    med = ndi.generic_filter(vals, np.nanmedian, size=3, mode='constant', cval=np.nan)
    top = np.where(have & (count >= 6), med, np.nan)
    np.savez_compressed(SURFACE_PATH, top=top.astype(np.float32), g=np.array([x0, y1, STEP]))
    return top.astype(np.float32), (x0, y1, STEP)


GROUND_PATH = ROOT / 'lidar-ground-v34.npz'


def ground_grid(world, frame, step=1., rebuild=False):
    """(z, (x0, y1, step)): median ground return per 1 m cell (scene z), gaps filled from the nearest cell. Cameras
    stand on it: a wall's base can be metres above the street it is photographed from."""
    if GROUND_PATH.exists() and not rebuild:
        d = np.load(GROUND_PATH); return d['z'], tuple(float(v) for v in d['g'])
    from scipy import ndimage as ndi
    from pyproj import Transformer
    pts = np.load(ROOT / 'data/points.npz')['points']; P = pts[pts[:, 3] == 2]; del pts
    px, py = Transformer.from_crs(3857, 32610, always_xy=True).transform(P[:, 0], P[:, 1])
    x = px - frame.cx; y = py - frame.cy; z = P[:, 2] - float(world['verticalOrigin'])
    x0, y1 = math.floor(x.min()), math.ceil(y.max()); nx, ny = int((x.max() - x0) / step) + 1, int((y1 - y.min()) / step) + 1
    flat = ((y1 - y) / step).astype(int) * nx + ((x - x0) / step).astype(int)
    o = np.lexsort((z, flat)); f, zz = flat[o], z[o]; first = np.r_[0, np.flatnonzero(np.diff(f)) + 1]; cnt = np.diff(np.r_[first, len(f)])
    g = np.full(nx * ny, np.nan); g[f[first]] = zz[first + cnt // 2]; g = g.reshape(ny, nx)
    miss = ~np.isfinite(g); g = g[tuple(ndi.distance_transform_edt(miss, return_distances=False, return_indices=True))]
    np.savez_compressed(GROUND_PATH, z=g.astype(np.float32), g=np.array([x0, y1, step]))
    return g.astype(np.float32), (x0, y1, step)


def ground_at(ground, xy):
    z, (x0, y1, st) = ground
    ix = int(np.clip((xy[0] - x0) / st, 0, z.shape[1] - 1)); iy = int(np.clip((y1 - xy[1]) / st, 0, z.shape[0] - 1))
    return float(z[iy, ix])


def own_footprints(wall, occluders):
    """Every footprint that holds the wall's building (both map and city outlines of it): those containing points
    0.5 m and 1.5 m behind the wall, as the camera visibility test takes them."""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n)
    idx = occluders.own_indices(Point(*(m - n * .5))) | occluders.own_indices(Point(*(m - n * 1.5)))
    return unary_union([occluders.polys[i] for i in idx]) if idx else None


def hidden_below(wall, cam_xy, cam_z, surf, L, own_poly=None, wall_top=None):
    """Per rectified column, the height (scene z) below which the wall is hidden behind LiDAR building mass. Mass
    inside the wall's own building (every footprint holding it: map and city outlines disagree by metres, and one ran
    13 m in front of the other's wall) is not counted: another part of the same building in front of the wall is not
    detected this way, a known limit."""
    top, (x0, y1, st) = surf
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
    W = max(8, int(L * PPM)); xs = (np.arange(W) + .5) / PPM - L / 2
    P = m[None, :] + t[None, :] * xs[:, None] + n[None, :] * .05; C = np.asarray(cam_xy, float)
    D = P - C[None, :]; dist = np.linalg.norm(D, axis=1)
    k = np.arange(NEAR_M, max(NEAR_M + st, dist.max()), st / 2)
    T = k[None, :] / dist[:, None]; ok = k[None, :] < dist[:, None] - FAR_M
    X = C[0] + D[:, :1] * T; Y = C[1] + D[:, 1:2] * T
    ix = np.floor((X - x0) / st).astype(int); iy = np.floor((y1 - Y) / st).astype(int)
    ok &= (ix >= 0) & (ix < top.shape[1]) & (iy >= 0) & (iy < top.shape[0])
    d = np.full(X.shape, np.nan, np.float32); d[ok] = top[iy[ok], ix[ok]]
    if own_poly is not None:
        import shapely
        fin = np.isfinite(d)
        if fin.any():
            own = np.zeros(X.shape, bool); own[fin] = shapely.contains_xy(own_poly, X[fin], Y[fin]); d[own] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        h = cam_z + (d - MARGIN_M - cam_z) / T
    h[~np.isfinite(h)] = -np.inf
    hb = h.max(1)
    # a hidden stretch narrower than a metre of wall is noise in the surface or the pose, not a building in front
    from scipy import ndimage as ndi
    return ndi.grey_opening(hb, size=max(1, int(1. * PPM)))


def hidden_mask(wall, cam, z0, H, L, surf, own_poly=None, ground=None, eye_m=2.4):
    """(Hp, W) bool on the rectify grid (row 0 at the wall top): True where another building hides the wall. The
    camera stands eye_m above the LiDAR ground under it (the wall's base when no ground grid is given)."""
    cz = (ground_at(ground, cam['xy']) if ground is not None else z0) + eye_m
    hb = hidden_below(wall, cam['xy'], cz, surf, L, own_poly, z0 + H)
    Hp = max(8, int(H * PPM)); zs = z0 + H - (np.arange(Hp) + .5) / PPM
    return zs[:, None] < hb[None, :]


BACK_NEAR_M, BACK_FAR_M, BACK_DROP_M, BACK_MIN_WALL_M = .5, 3., 3., 6.


def unbacked_mask(wall, z0, H, L, surf, own_poly=None, setback=None):
    """(Hp, W) bool on the rectify grid: True where the wall is not this building's face. Per column, the LiDAR mass
    0.5-3 m behind the wall must reach within 3 m of the wall's top. Where it does not, the wall is not this building's
    face at that place: an outline carried to a corner where only a lower building stands (a 15 m wall backed by a
    9.5 m roof), or a wall drawn to a height the mass there does not reach. If this building's own mass 3-25 m further
    back reaches the top, the rows above the near mass show its set-back face (good for its colour, not for a layout at
    the wall's plane) and only the rows below are masked; otherwise the whole column is. Parts are split by roof tier,
    so mass well below a part's wall top is another part's. Walls up to 6 m (garden and service walls, free-standing)
    are not tested; envelope designs model their own setbacks and are not tested either. `setback`, a dict, receives
    the share of columns that show a set-back face ('share') and the share of the unmasked wall that is set-back face
    ('visible')."""
    top, (x0, y1, st) = surf
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
    W = max(8, int(L * PPM)); Hp = max(8, int(H * PPM)); xs = (np.arange(W) + .5) / PPM - L / 2
    if setback is not None: setback['share'] = setback['visible'] = 0.
    if H <= BACK_MIN_WALL_M: return np.zeros((Hp, W), bool)     # a garden or service wall has no roof behind it
    def band(d0, d1, own_only):
        ds = np.arange(d0, d1 + 1e-6, st / 2)
        X = m[0] + t[0] * xs[:, None] - n[0] * ds[None, :]; Y = m[1] + t[1] * xs[:, None] - n[1] * ds[None, :]
        ix = np.floor((X - x0) / st).astype(int); iy = np.floor((y1 - Y) / st).astype(int)
        ok = (ix >= 0) & (ix < top.shape[1]) & (iy >= 0) & (iy < top.shape[0])
        d = np.full(X.shape, np.nan, np.float32); d[ok] = top[iy[ok], ix[ok]]
        if own_only:
            if own_poly is None: return np.full(W, -np.inf)
            import shapely
            fin = np.isfinite(d); own = np.zeros(X.shape, bool); own[fin] = shapely.contains_xy(own_poly, X[fin], Y[fin]); d[~own] = np.nan
        return np.nanmax(np.where(np.isfinite(d), d, -np.inf), 1)
    wall_top = z0 + H; near = band(BACK_NEAR_M, BACK_FAR_M, False)
    from scipy import ndimage as ndi
    run = max(1, int(1. * PPM))
    bad = ndi.binary_opening(near < wall_top - BACK_DROP_M, structure=np.ones(run))  # a metre at least: an edge cell is not a gap
    if not bad.any(): return np.zeros((Hp, W), bool)
    far = band(BACK_FAR_M, 25., True); sb = bad & (far >= wall_top - BACK_DROP_M)
    zs = z0 + H - (np.arange(Hp) + .5) / PPM
    mask = np.zeros((Hp, W), bool); mask[:, bad & ~sb] = True
    lim = np.where(sb, near + 1., -np.inf); mask |= (zs[:, None] < lim[None, :]) & sb[None, :]
    if setback is not None:
        setback['cols'] = sb; setback['near'] = near
        setback['share'] = float(sb.mean()); free = ~mask
        setback['visible'] = float((free & sb[None, :]).sum() / max(1, free.sum()))   # of what is left to read, the set-back face
    return mask


def setback_other(wall, cam, z0, H, L, surf, own_poly, cols, ground=None, eye_m=2.4, reach=25.):
    """(Hp, W) bool: in set-back columns, the pixels whose camera ray, followed beyond the wall plane, meets another
    building (or nothing within 25 m) before this building's own mass. The set-back face is only this building where
    the ray reaches it first; beyond a phantom corner wall the rays found a neighbour's brick as often as the glass
    block they were meant to read."""
    top, (x0, y1, st) = surf
    W = max(8, int(L * PPM)); Hp = max(8, int(H * PPM)); out = np.zeros((Hp, W), bool)
    idx = np.flatnonzero(cols) if cols is not None else []
    if not len(idx) or own_poly is None: return out
    import shapely
    m = np.asarray(wall['midpoint'], float); n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
    xs = (idx + .5) / PPM - L / 2; P = m[None, :] + t[None, :] * xs[:, None]; C = np.asarray(cam['xy'], float)
    cz = (ground_at(ground, cam['xy']) if ground is not None else z0) + eye_m
    D = P - C[None, :]; dist = np.linalg.norm(D, axis=1); u = D / dist[:, None]
    ks = np.arange(.25, reach, st / 2)
    X = P[:, :1] + u[:, :1] * ks[None, :]; Y = P[:, 1:] + u[:, 1:] * ks[None, :]
    ix = np.floor((X - x0) / st).astype(int); iy = np.floor((y1 - Y) / st).astype(int)
    ok = (ix >= 0) & (ix < top.shape[1]) & (iy >= 0) & (iy < top.shape[0])
    d = np.full(X.shape, np.nan, np.float32); d[ok] = top[iy[ok], ix[ok]]
    fin = np.isfinite(d); own = np.zeros(X.shape, bool); own[fin] = shapely.contains_xy(own_poly, X[fin], Y[fin])
    T = (dist[:, None] + ks[None, :]) / dist[:, None]
    zs = z0 + H - (np.arange(Hp) + .5) / PPM
    ray = cz + (zs[:, None, None] - cz) * T[None, :, :]                 # (Hp, cols, K) ray height beyond the plane
    hit = fin[None, :, :] & (d[None, :, :] >= ray)
    first = hit.argmax(2); anyhit = hit.any(2)
    first_own = np.take_along_axis(np.broadcast_to(own[None], hit.shape), first[..., None], 2)[..., 0]
    out[:, idx] = ~(anyhit & first_own)
    return out


def box_hidden(box, mask, H):
    """Share of a box (x0, z0, x1, z1 in wall metres) that is hidden."""
    Hp, W = mask.shape; x0, z0, x1, z1 = box[:4]
    r0, r1 = int(max(0, (H - z1) * PPM)), int(min(Hp, (H - z0) * PPM)); c0, c1 = int(max(0, x0 * PPM)), int(min(W, x1 * PPM))
    return float(mask[r0:r1, c0:c1].mean()) if r1 > r0 and c1 > c0 else 0.


def mask_found(found, hidden, H):
    """Detections with the hidden area added to the occluder mask and the boxes that stand in it dropped (their
    windows and signs are the building in front)."""
    ops, signs, occ, visible, u8, glass = found
    if hidden is None or not hidden.any(): return found
    occ2 = (occ.astype(bool) | hidden) if occ is not None and occ.shape == hidden.shape else hidden
    keep = lambda bs: [b for b in bs if box_hidden(b, hidden, H) < .5]
    street = max(1, occ2.shape[0] - int(2.5 * PPM))
    return keep(ops), keep(signs), occ2, float(1 - occ2[:street].mean()), u8, keep(glass)


# ----------------------------------------------------------------------------------------------- regular layouts

SNAP_TOL, FIT_SHARE, SHARE_PITCH, SHARE_PHASE, WIDTH_CV = .15, .85, .08, .15, .25


def fit_pitch(xs, min_pitch=1.):
    """(pitch, phase, inliers) for sorted window centres xs, or None: the pitch and phase that put 85 % of the centres
    within 15 % of a pitch of a grid position (a regular facade read with noise, not windows in pairs and groups)."""
    xs = np.asarray(sorted(xs), float)
    if len(xs) < 3: return None
    d = np.diff(xs); d = d[d > .3]
    if not len(d): return None
    best = None
    for p0 in {float(np.median(d)), float(np.min(d)), float(np.median(d[d < 1.5 * np.min(d)])) if (d < 1.5 * np.min(d)).any() else float(np.min(d))}:
        if p0 < min_pitch: continue
        k = np.round((xs - xs[0]) / p0)
        if len(set(k)) < 3: continue
        A = np.c_[np.ones(len(k)), k]; (ph, p), *_ = np.linalg.lstsq(A, xs, rcond=None)
        if p < min_pitch: continue
        res = np.abs(xs - (ph + np.round((xs - ph) / p) * p)); inl = res < SNAP_TOL * p
        if inl.mean() >= FIT_SHARE and (best is None or inl.sum() > best[2].sum() or (inl.sum() == best[2].sum() and res[inl].mean() < best[3])):
            best = (float(p), float(ph), inl, float(res[inl].mean()))
    return None if best is None else best[:3]


def regular_layout(p):
    """Rewrite a photo-read profile's upper-storey windows as a regular layout (module docstring). Returns a summary."""
    if p.get('materialOnly') or not p.get('openings') or not p.get('frameU') or not p.get('heightRefM'): return None
    L = float(p['frameU'][1] - p['frameU'][0]); H = float(p['heightRefM'])
    if L <= 0 or H <= 0: return None
    gh = float((p.get('reading') or {}).get('groundStoreyM') or min(4.5, H * .35))
    wins, rest = [], []
    for o in p['openings']:
        x0, z0, x1, z1 = o['rect']; b = [x0 * L, z0 * H, x1 * L, z1 * H]
        (wins if o['style'].get('kind') == 'window' and (b[1] + b[3]) / 2 > gh else rest).append((b, o))
    if not wins: return None
    def sliver(b):
        w, h = b[2] - b[0], b[3] - b[1]; return w < .45 or w * h < .35 or h > 8 * w      # thin and tall, or tiny; a wide low ribbon (transom lights) stays
    kept = [(b, o) for b, o in wins if not sliver(b)]; dropped = len(wins) - len(kept)
    kept.sort(key=lambda t: (t[0][1] + t[0][3]) / 2); rows = []
    for b, o in kept:
        cz = (b[1] + b[3]) / 2
        if rows and abs(cz - rows[-1]['cz']) < .9:
            rows[-1]['w'].append([list(b), o]); rows[-1]['cz'] = float(np.mean([(q[0][1] + q[0][3]) / 2 for q in rows[-1]['w']]))
        else: rows.append({'cz': cz, 'w': [[list(b), o]]})
    for r in rows:
        r['z0'] = float(np.median([b[1] for b, _ in r['w']])); r['z1'] = float(np.median([b[3] for b, _ in r['w']]))
        r['fit'] = fit_pitch([(b[0] + b[2]) / 2 for b, _ in r['w']])
    fitted = [r for r in rows if r['fit']]
    common = None
    if len(fitted) >= 2:
        pm = float(np.median([r['fit'][0] for r in fitted])); ref = max(fitted, key=lambda r: len(r['w'])); ph0 = ref['fit'][1] % pm
        def phase_ok(r):
            dph = abs(((r['fit'][1] - ph0) + pm / 2) % pm - pm / 2); return dph <= SHARE_PHASE * pm
        if all(abs(r['fit'][0] - pm) <= SHARE_PITCH * pm and phase_ok(r) for r in fitted): common = (pm, ph0)
    snapped = 0
    def on_grid(r, pch, ph):
        out = []
        for q in r['w']:
            b = q[0]; cx = (b[0] + b[2]) / 2; g = ph + round((cx - ph) / pch) * pch
            if abs(cx - g) <= .3 * pch: out.append((q, g))
        return out
    def similar(ws): return len(ws) >= 2 and float(np.std(ws) / max(np.mean(ws), 1e-6)) <= WIDTH_CV
    common_w = None
    if common:                                               # storeys sharing a grid share one window width
        allw = [q[0][2] - q[0][0] for r in rows for q, _ in on_grid(r, *common)]
        common_w = min(float(np.median(allw)), .85 * common[0]) if similar(allw) else None
    for r in rows:
        grid = common or ((r['fit'][0], r['fit'][1] % r['fit'][0]) if r['fit'] else None)
        if grid:
            pch, ph = grid; on = on_grid(r, pch, ph)
            if on:
                ow = [q[0][2] - q[0][0] for q, _ in on]
                wm = common_w if common_w else (min(float(np.median(ow)), .85 * pch) if similar(ow) else None)
                for q, g in on:
                    w = q[0][2] - q[0][0]
                    if wm and .5 * wm <= w <= 1.6 * wm: q[0][0], q[0][2] = g - wm / 2, g + wm / 2
                    else: q[0][0], q[0][2] = g - w / 2, g + w / 2
                    snapped += 1
        else:
            ws = [q[0][2] - q[0][0] for q in r['w']]; wm = float(np.median(ws))
            if similar(ws):
                for q in r['w']:
                    w = q[0][2] - q[0][0]; cx = (q[0][0] + q[0][2]) / 2
                    if .6 * wm <= w <= 1.4 * wm: q[0][0], q[0][2] = cx - wm / 2, cx + wm / 2
    heights = [r['z1'] - r['z0'] for r in rows]; hm = float(np.median(heights)) if heights else 0.
    for r in rows:
        h = r['z1'] - r['z0']
        if hm and abs(h - hm) <= .25 * hm: r['z1'] = r['z0'] + hm
        r['w'].sort(key=lambda q: q[0][0]); merged = []
        for q in r['w']:
            b = q[0]; b[1], b[3] = r['z0'], r['z1']; b[0], b[2] = max(0., b[0]), min(L, b[2])
            if b[2] - b[0] < .3: continue
            if merged and b[0] < merged[-1][0][2] + .15: merged[-1][0][2] = max(merged[-1][0][2], b[2])
            else: merged.append(q)
        r['w'] = merged
    out = [o for _, o in rest]
    for r in rows:
        for b, o in r['w']:
            o = dict(o); o['rect'] = [round(b[0] / L, 4), round(b[1] / H, 4), round(b[2] / L, 4), round(b[3] / H, 4)]
            o['evidence'] = 'image-detected, regular layout'; out.append(o)
    p['openings'] = out
    summary = {'storeys': len(rows), 'bayPitchM': round(common[0], 2) if common else None, 'rowsWithPitch': len(fitted), 'windowsSnapped': snapped, 'sliversDropped': dropped}
    p.setdefault('reading', {})['layout'] = summary
    return summary


# --------------------------------------------------------------------------------------- components that agree

def agree_positions(values, tol, min_photos=2):
    """values: [(photo index, position, item)]. Clusters of positions within tol held by min_photos photographs:
    [(median position, [items])]."""
    values = sorted(values, key=lambda v: v[1]); out = []; cur = []
    for v in values:
        if cur and v[1] - cur[-1][1] > tol:
            if len({c[0] for c in cur}) >= min_photos: out.append((float(np.median([c[1] for c in cur])), [c[2] for c in cur]))
            cur = []
        cur.append(v)
    if cur and len({c[0] for c in cur}) >= min_photos: out.append((float(np.median([c[1] for c in cur])), [c[2] for c in cur]))
    return out


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0])); iz = max(0., min(a[3], b[3]) - max(a[1], b[1])); inter = ix * iz
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.


# ------------------------------------------------------------------------------------ windows that photographs agree on

def same_opening(a, b):
    """Two boxes (x0, z0, x1, z1 in wall metres) from different photographs are one opening: centres within 0.8 m (or
    half the width) and sizes within a factor two. Photographs of one wall disagree by up to a metre (pose), so an
    overlap test would split them."""
    wa, wb = a[2] - a[0], b[2] - b[0]; ha, hb = a[3] - a[1], b[3] - b[1]
    if min(wa, wb, ha, hb) <= 0: return False
    tol = max(.8, .5 * max(wa, wb))
    return (abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) <= tol and abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) <= max(.8, .5 * max(ha, hb))
            and max(wa, wb) / min(wa, wb) <= 2. and max(ha, hb) / min(ha, hb) <= 2.)


def corroborated(box, others, H, min_clear=.6):
    """(confirmed, contradicted): whether any other photograph detected this opening, and how many others that see its
    place clearly (60 % unoccluded) did not. `others`: [(ops, occ)] of the other photographs of the wall."""
    confirmed = False; contradicted = 0
    for ops, occ in others:
        if any(same_opening(box, o) for o in ops): confirmed = True; continue
        if 1. - box_hidden(box, occ, H) >= min_clear: contradicted += 1
    return confirmed, contradicted


def agreed_openings(ops, others, H, min_contradicting=2):
    """The detected openings of one photograph that the others do not contradict: an opening goes when no other
    photograph detected it and two or more that see its place clearly show nothing there (a palm trunk, a sign frame, a
    reflection is in one photograph; a window is in all that see it)."""
    keep, dropped = [], 0
    for o in ops:
        ok, against = corroborated(o, others, H)
        if ok or against < min_contradicting: keep.append(o)
        else: dropped += 1
    return keep, dropped
