"""Ballpark landmark builder.

Measured:
- The stands are the LiDAR surface itself: the highest return per cell, with raised structures narrower than 6.5 m
  (masts, rigs, the board, rails) replaced by what they stand on and a 2.5 m median against noise, triangulated inside
  the footprint so the outline stays exact. Each face is typed by slope and
  by whether it falls toward the field: seating (rows parallel to the field edge), near-vertical deck fronts (glazed
  at suite height when they face the field, concrete elsewhere), flat concourses and roofs.
- The playing field is the footprint's opening: warning-track clay, the mapped turf, and a regulation infield placed
  by fitting its dirt/grass pattern to the return intensity (dirt reads darker than grass).
- Light rigs and the outfield board are the LiDAR clusters standing above the filtered surface, rebuilt as lamp banks
  on posts and a board with a screen facing the plate.
Authored: the shapes of the props named in the landmark entry (a contour bottle, a four-finger glove); their position,
axis and size come from their own LiDAR clusters.
"""
import math
import numpy as np
import trimesh
import shapely
import shapely.geometry.polygon
from scipy import ndimage as ndi
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely import segmentize, make_valid
from . import diamond

ROW = .85                     # seat rows: tread depth, measured away from the field edge
SEAT_TILE = (14., 6.8)        # the seats recipe tile: 14 m along the row by 8 rows
TURF_TILE = 10.               # the turf recipe tile (a 2 x 2 mowing check)


def _polys(g):
    if g.is_empty: return []
    if g.geom_type == 'Polygon': return [g]
    return [p for a in getattr(g, 'geoms', []) for p in _polys(a)]


def surface_grid(P, bounds, step):
    """The solid surface: the highest return per cell (holes filled from the nearest cell), then a 2.5 m median, which
    drops masts, posts, rails and thin walls while steps between decks survive. Thin structures standing above the
    stands must already be out of P (see `standing_clusters`)."""
    x0, y0, x1, y1 = bounds; x0 -= 4; y0 -= 4; x1 += 4; y1 += 4
    nx = int((x1 - x0) / step) + 1; ny = int((y1 - y0) / step) + 1
    ix = ((P[:, 0] - x0) / step).astype(int); iy = ((y1 - P[:, 1]) / step).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy[ok] * nx + ix[ok]; z = P[ok, 2]; o = np.argsort(z)
    top = np.full(nx * ny, np.nan); top[flat[o]] = z[o]; top = top.reshape(ny, nx)
    miss = ~np.isfinite(top)
    if miss.all(): raise ValueError('no LiDAR returns over the landmark')
    top = top[tuple(ndi.distance_transform_edt(miss, return_distances=False, return_indices=True))]
    med = ndi.median_filter(top, size=max(3, int(round(2.5 / step)) | 1))
    level, flat = plateaus(med, step)
    # A light blur off the plateaus: a deck front crossing the grid diagonally would otherwise alternate between the
    # two levels cell by cell (a sawtooth); it becomes a steep front two cells wide. Plateaus stay exactly level.
    return np.where(flat, level, ndi.gaussian_filter(med, 1.)), (x0, y1, step)


def plateaus(H, step, slope_max=.15, dz=.3, min_m2=16.):
    """Level regions: 4-neighbour cells that are both near-flat and within `dz` of each other, grouped; each group of at
    least `min_m2` is set to its median height (concourses and roofs come out exactly level, with the clutter on them
    gone and sharp edges where they meet). Returns (levelled heights, mask of levelled cells)."""
    gy, gx = np.gradient(H, step); flat = np.hypot(gx, gy) < slope_max
    ny, nx = H.shape; idx = np.arange(ny * nx).reshape(ny, nx); Hr = H.ravel(); fr = flat.ravel(); edges = []
    for a, b in ((idx[:, :-1].ravel(), idx[:, 1:].ravel()), (idx[:-1, :].ravel(), idx[1:, :].ravel())):
        m = fr[a] & fr[b] & (np.abs(Hr[a] - Hr[b]) < dz); edges.append(np.c_[a[m], b[m]])
    e = np.concatenate(edges)
    n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(ny * nx, ny * nx)), directed=False)
    count = np.bincount(lab, minlength=n); keep = fr & (count[lab] * step * step >= min_m2)
    order = np.lexsort((Hr, lab)); first = np.searchsorted(lab[order], np.arange(n))
    median = Hr[order][np.minimum(first + count // 2, len(order) - 1)]
    level = np.where(keep, median[lab], Hr).reshape(ny, nx); keep = keep.reshape(ny, nx)
    # Clutter on a plateau (umbrellas, kiosks, plant): a small patch whose whole rim is levelled takes the rim's level.
    blobs, nb = ndi.label(~keep)
    if nb:
        size = np.bincount(blobs.ravel(), minlength=nb + 1); rim = ndi.grey_dilation(blobs, size=3) * keep
        rim_n = np.bincount(rim.ravel(), minlength=nb + 1); rim_z = np.bincount(rim.ravel(), weights=level.ravel(), minlength=nb + 1)
        edge_n = np.bincount(ndi.grey_dilation(blobs, size=3).ravel(), minlength=nb + 1) - size        # rim cells of any kind
        enclosed = (size * step * step < min_m2) & (rim_n >= edge_n) & (rim_n > 0); enclosed[0] = False
        fill = enclosed[blobs]; level = np.where(fill, (rim_z / np.maximum(rim_n, 1))[blobs], level); keep = keep | fill
    return level, keep


def standing_clusters(P, site, z_min, link=2.5, min_points=60):
    """Returns higher than z_min over the site, grouped by 3D proximity: the light rigs, mast heads and board tops that
    stand above a ballpark's roofs (which stay below z_min). Returns (mask of P, list of member-index arrays)."""
    hi = np.flatnonzero((P[:, 2] > z_min) & shapely.contains_xy(site.buffer(2), P[:, 0], P[:, 1]))
    if len(hi) < min_points: return np.zeros(len(P), bool), []
    pairs = cKDTree(P[hi, :3]).query_pairs(link, output_type='ndarray')
    n, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(hi), len(hi))), directed=False)
    groups = [hi[lab == k] for k in range(n) if (lab == k).sum() >= min_points]
    mask = np.zeros(len(P), bool); mask[hi] = True
    return mask, groups


def sample(grid, g, x, y):
    x0, y1, st = g; fx = (np.asarray(x, float) - x0) / st - .5; fy = (y1 - np.asarray(y, float)) / st - .5
    ix = np.clip(np.floor(fx).astype(int), 0, grid.shape[1] - 2); iy = np.clip(np.floor(fy).astype(int), 0, grid.shape[0] - 2)
    fx = np.clip(fx - ix, 0, 1); fy = np.clip(fy - iy, 0, 1)
    return (grid[iy, ix] * (1 - fx) * (1 - fy) + grid[iy, ix + 1] * fx * (1 - fy)
            + grid[iy + 1, ix] * (1 - fx) * fy + grid[iy + 1, ix + 1] * fx * fy)


def triangulate(poly, step, interior_step=None):
    """Vertices on the outline every `step` metres plus a grid inside; Delaunay triangles kept only inside, CCW."""
    interior_step = interior_step or step
    rings = [np.asarray(r.coords) for q in _polys(poly) for r in [q.exterior, *q.interiors]]
    edge = np.concatenate([np.asarray(segmentize(LineString(r), step).coords)[:-1] for r in rings])
    xmin, ymin, xmax, ymax = poly.bounds
    xx, yy = np.meshgrid(np.arange(xmin, xmax, interior_step) + interior_step / 2, np.arange(ymin, ymax, interior_step) + interior_step / 2)
    inner = np.c_[xx.ravel(), yy.ravel()]
    inner = inner[shapely.contains_xy(poly.buffer(-interior_step * .35), inner[:, 0], inner[:, 1])]
    V = np.unique(np.round(np.r_[edge, inner], 3), axis=0)
    F = Delaunay(V).simplices
    F = F[shapely.covered_by(shapely.polygons(V[F]), poly.buffer(.01))]
    a = V[F[:, 1]] - V[F[:, 0]]; b = V[F[:, 2]] - V[F[:, 0]]
    cw = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0; F[cw] = F[cw][:, [0, 2, 1]]
    return V, F


def boundary_edges(F):
    """Directed edges used by one triangle only (interior on their left)."""
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.ravel()] == 1]


def wall(a, b, za, zb, bottom_a, bottom_b):
    """Quad from ground to top along a->b, facing right of a->b (outward for an edge with the surface on its left)."""
    v = np.array([[*a, bottom_a], [*b, bottom_b], [*b, zb], [*a, za]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]])


def flat_surface(ctx, poly, offset, key, mat, uv_of):
    """A ground-hugging surface (terrain height + offset, densified on the terrain grid) with its own UV mapping."""
    for a in _polys(make_valid(poly)):
        if a.area < .05: continue
        V, F = triangulate(a, 1.5, ctx['STEP'])
        z = ctx['elev'](V[:, 0], V[:, 1]) + offset
        ctx['put'](trimesh.Trimesh(np.c_[V, z], F, process=False), key, mat, uv=uv_of(V))


def lathe(axis_a, axis_b, profile, radius, sections=28):
    """Surface of revolution along a->b: profile is [(s in 0..1, r fraction)], capped at both ends."""
    a = np.asarray(axis_a, float); b = np.asarray(axis_b, float); d = b - a; L = np.linalg.norm(d); d /= L
    t = np.cross(d, [0, 0, 1.]) if abs(d[2]) < .95 else np.cross(d, [1., 0, 0]); t /= np.linalg.norm(t); w = np.cross(d, t)
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    rings = [a + d * s * L + radius * r * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * w) for s, r in profile]
    V = np.concatenate(rings + [a[None], (a + d * L)[None]]); F = []
    for i in range(len(rings) - 1):
        for j in range(sections):
            p, q = i * sections + j, i * sections + (j + 1) % sections
            F += [[p, q, q + sections], [p, q + sections, p + sections]]
    base, tip = len(V) - 2, len(V) - 1; last = (len(rings) - 1) * sections
    for j in range(sections):
        F += [[base, (j + 1) % sections, j], [tip, last + j, last + (j + 1) % sections]]
    return trimesh.Trimesh(V, F, process=False)


# A contour bottle, base to lip: body, waist, shoulder, neck.
BOTTLE = [(0, .80), (.03, .95), (.12, 1.), (.24, .9), (.33, .8), (.43, .9), (.52, 1.), (.62, .9), (.72, .62), (.8, .44),
          (.9, .36), (.97, .37), (1., .4)]


def build(ctx, c, z0, resolved, lm):
    put, cube, cylinder, elev = ctx['put'], ctx['cube'], ctx['cylinder'], ctx['elev']
    import time; clock = [time.perf_counter()]; timing = {}
    def lap(name): timing[name] = round(time.perf_counter() - clock[0], 1); clock[0] = time.perf_counter()
    fid = c['id']; g = c['g']; step = lm.get('surfaceStepM', .5)
    main = max(_polys(g), key=lambda p: p.area)
    field = max((Polygon(h) for h in main.interiors), key=lambda h: h.area)
    site = Polygon(main.exterior)
    # Observed frontages keep a strip as deep as their frame (their facade, trim and frame stand in it, lidded at the
    # facade's own top); the stands begin behind it and close against it with a dark wall the frame reads against.
    view = ctx['landmarks_view']
    strip = make_valid(ctx['envelopes'].replacement_mask(view)).intersection(main) if view else Polygon()
    stands = make_valid(main.difference(strip.buffer(.01))) if not strip.is_empty else main
    stands = max(_polys(stands), key=lambda q: q.area) if stands.geom_type != 'Polygon' else stands
    pts = ctx['pts']; b = site.bounds
    near = (pts[:, 0] > b[0] - 6) & (pts[:, 0] < b[2] + 6) & (pts[:, 1] > b[1] - 6) & (pts[:, 1] < b[3] + 6) & (pts[:, 3] != 7) & (pts[:, 3] != 18)
    P = pts[near]
    standing, groups = standing_clusters(P, site, z0 + 40.)
    smooth, gref = surface_grid(P[~standing], b, step)
    record = {'landmark': lm.get('label'), 'builder': 'ballpark', 'frontageStripM2': round(strip.area)}
    lap('surface')
    # Authored props own their LiDAR clusters: each return goes to the nearest seed within reach, the prop's platform
    # is the common height of the returns round it, and the stands surface is flattened to that platform under it.
    props_in = []
    for pr in lm.get('props', []):
        seed = np.array(pr['seed'], float); dist = np.hypot(P[:, 0] - seed[0], P[:, 1] - seed[1])
        mine = dist < pr['radiusM']
        for other in lm.get('props', []):                 # shared returns go to the prop they are deepest inside
            if other is not pr: mine &= dist / pr['radiusM'] <= np.hypot(P[:, 0] - other['seed'][0], P[:, 1] - other['seed'][1]) / other['radiusM']
        ring_z = P[(dist >= pr['radiusM']) & (dist < pr['radiusM'] + 4), 2]
        platform = float(np.percentile(ring_z, 40)) if len(ring_z) > 20 else float(np.percentile(P[mine, 2], 20))
        props_in.append((pr, seed, P[mine & (P[:, 2] > platform + 2)], platform))
        yy_, xx_ = np.mgrid[0:smooth.shape[0], 0:smooth.shape[1]]
        cx_ = gref[0] + (xx_ + .5) * gref[2]; cy_ = gref[1] - (yy_ + .5) * gref[2]
        under = (np.hypot(cx_ - seed[0], cy_ - seed[1]) < pr['radiusM']) & (smooth > platform + 1.5)
        smooth[under] = platform

    # Field level: the median of ground returns on the field; the field surfaces follow the terrain above it.
    on_field = shapely.contains_xy(field.buffer(-2), P[:, 0], P[:, 1])
    zf = float(np.median(P[on_field & (P[:, 3] == 2), 2])) if (on_field & (P[:, 3] == 2)).sum() > 100 else float(z0)
    record['fieldLevelM'] = round(zf, 2)

    # --- Stands: the filtered LiDAR surface inside the footprint.
    V, F = triangulate(stands, step)
    edge_grid = ndi.maximum_filter(smooth, size=3)
    on_edge = ~shapely.contains_xy(stands.buffer(-step * .6), V[:, 0], V[:, 1])
    Z = np.where(on_edge, sample(edge_grid, gref, V[:, 0], V[:, 1]), sample(smooth, gref, V[:, 0], V[:, 1]))
    Z = np.maximum(Z, elev(V[:, 0], V[:, 1]) + .4)
    V3 = np.c_[V, Z]
    # Direction to the field and position along its edge, for every vertex: the edge runs from centre field round.
    ring = np.asarray(segmentize(field.exterior, .5).coords)[:-1]
    fit_plate = None
    kd = cKDTree(ring)
    T = V3[F]; n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    nz = n[:, 2]; slope = np.sqrt(np.maximum(0, 1 - nz ** 2)) / np.maximum(nz, 1e-6); cen = T.mean(1)
    d_c, i_c = kd.query(cen[:, :2]); to_field = (ring[i_c] - cen[:, :2]) / np.maximum(d_c, 1e-6)[:, None]
    downhill = n[:, :2] / np.maximum(np.linalg.norm(n[:, :2], axis=1, keepdims=True), 1e-9)
    facing = (downhill * to_field).sum(1) > .35
    seats = (slope > .1) & (slope < 1.35) & facing
    steep = (nz < .42) & ~seats
    zmid = cen[:, 2] - z0
    suites = steep & facing & (zmid > 6) & (zmid < 24)
    kinds = {'seats': seats, 'glass_dark': suites, 'brick_modular': steep & ~facing,
             'venue_concrete': (steep & facing & ~suites) | (~seats & ~steep & (zmid <= 30)), 'roofing': ~seats & ~steep & (zmid > 30)}

    # --- The playing field: plate and bearing fitted to the return intensity.
    x0, y0, x1, y1 = field.bounds; st = .5; nx = int((x1 - x0) / st) + 1; ny = int((y1 - y0) / st) + 1
    low = on_field & (P[:, 2] < zf + 1.5)
    ix = ((P[low, 0] - x0) / st).astype(int); iy = ((P[low, 1] - y0) / st).astype(int); ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    acc = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    np.add.at(acc, (iy[ok], ix[ok]), P[low, 4][ok]); np.add.at(cnt, (iy[ok], ix[ok]), 1)
    R = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    xs, ys = np.meshgrid(x0 + (np.arange(nx) + .5) * st, y0 + (np.arange(ny) + .5) * st)
    R[~shapely.contains_xy(field.buffer(-1), xs, ys)] = np.nan
    if np.isfinite(R).sum() > 2000:
        # Starting guess: the plate's dirt circle (a dark disc ringed by grass), bearing toward the field's centroid.
        Rn = np.where(np.isfinite(R), (R - np.nanmean(R)) / np.nanstd(R), 0.)
        yy, xx = np.mgrid[-16:17, -16:17] * st; rr = np.hypot(xx, yy)
        k = np.where(rr <= 3.6, -1., np.where((rr >= 4.6) & (rr <= 7.5), 1., 0.)); k[k < 0] /= abs(k[k < 0].sum()); k[k > 0] /= k[k > 0].sum()
        resp = ndi.convolve(Rn, k[::-1, ::-1]); resp[~np.isfinite(R)] = -9
        jy, jx = np.unravel_index(np.argmax(resp), resp.shape); plate0 = np.array([xs[jy, jx], ys[jy, jx]])
        cxy = np.array(field.centroid.coords[0]); b0 = math.atan2(cxy[1] - plate0[1], cxy[0] - plate0[0])
        plate, bearing, score = diamond.fit(R, xs, ys, plate0, b0)
        paths, foul_cut = diamond.variant(R, xs, ys, plate, bearing)
        fit_plate = (plate, bearing)
        record['infield'] = {'plate': [round(float(v), 2) for v in plate], 'bearingDeg': round(math.degrees(bearing), 2),
                             'separation': round(score, 3), 'basePaths': paths, 'foulCutouts': foul_cut}
    # Seat rows start at the field edge opposite the plate so the texture seam falls in the outfield.
    if fit_plate is not None:
        far = int(np.argmax(np.linalg.norm(ring - fit_plate[0], axis=1))); ring = np.roll(ring, -far, axis=0); kd = cKDTree(ring)
    s_along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
    d_v, i_v = kd.query(V[:, :2]); seat_uv = np.c_[s_along[i_v] / SEAT_TILE[0], d_v / SEAT_TILE[1]]
    def along_uv(mat, s_m, z_m):
        # Walls and fronts: u runs along the stands (continuous round every bend), v up the wall, in the material's tiles.
        su, sv = ctx['material_bank'].get(mat)[1]; return np.c_[s_m / su, z_m / sv]
    lap('field fit')
    for mat, mask in kinds.items():
        if not mask.any(): continue
        if mat == 'seats': put(trimesh.Trimesh(V3, F[mask], process=False), 'roofs__' + fid, mat, uv=seat_uv); continue
        up_f = mask & ~steep; st_f = mask & steep
        if up_f.any(): put(trimesh.Trimesh(V3, F[up_f], process=False), 'roofs__' + fid, mat)
        if st_f.any(): put(trimesh.Trimesh(V3, F[st_f], process=False), 'roofs__' + fid, mat, uv=along_uv(mat, s_along[i_v], Z))
    record['standsFaces'] = {k: int(v.sum()) for k, v in kinds.items()}
    lap('stands mesh')

    # --- Walls: outline to the ground in brick, the field wall in padding, a dark backdrop against the frontage strip,
    # concrete round courts. u runs along the ring each wall stands on, so bends never break the texture.
    outer = site.exterior; fring = field.exterior; strip_edge = strip.boundary if not strip.is_empty else None
    walls = {'brick_modular': [], 'padding': [], 'recess': [], 'venue_concrete': []}
    for a_i, b_i in boundary_edges(F):
        a, bb = V[a_i], V[b_i]; mid = Point((a + bb) / 2)
        if fring.distance(mid) < .05: mat, ring_l, bot = 'padding', fring, (zf - .1, zf - .1)
        elif strip_edge is not None and strip_edge.distance(mid) < .05: mat, ring_l, bot = 'recess', outer, (float(elev(*a)) - .2, float(elev(*bb)) - .2)
        elif outer.distance(mid) < .05: mat, ring_l, bot = 'brick_modular', outer, (float(elev(*a)) - .2, float(elev(*bb)) - .2)
        else: mat, ring_l, bot = 'venue_concrete', outer, (float(elev(*a)) - .2, float(elev(*bb)) - .2)
        wv, wf = wall(a, bb, Z[a_i], Z[b_i], *bot)
        sa, sb = ring_l.project(Point(a)), ring_l.project(Point(bb))
        if abs(sb - sa) > ring_l.length / 2: sb = sa + math.copysign(np.linalg.norm(bb - a), sb - sa)   # across the ring's start
        walls[mat].append((wv, wf, np.array([sa, sb, sb, sa])))
    for mat, quads in walls.items():
        if quads:
            WV = np.concatenate([q[0] for q in quads]); WF = np.concatenate([q[1] + 4 * i for i, q in enumerate(quads)])
            put(trimesh.Trimesh(WV, WF, process=False), 'buildings__' + fid, mat, uv=along_uv(mat, np.concatenate([q[2] for q in quads]), WV[:, 2]))
    # Arcade: the brick outline wall is pierced by arches at a steady pitch (dark recesses a hand's width proud of the
    # wall, so they win its depth), wherever the wall is tall enough and runs straight across the arch.
    ring_ccw = shapely.geometry.polygon.orient(site, 1.0).exterior; arch = 0
    for sm in np.arange(2.6, ring_ccw.length - 2.6, lm.get('arcadePitchM', 5.2)):
        pa, pm, pb = (np.array(ring_ccw.interpolate(v).coords[0]) for v in (sm - 1.5, sm, sm + 1.5))
        if not strip.is_empty and strip.buffer(1.).contains(Point(*pm)): continue
        t1, t2 = pm - pa, pb - pm
        if np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-9) < .94: continue          # a corner inside the arch
        t = (pb - pa) / np.linalg.norm(pb - pa); out = np.array([t[1], -t[0]])
        base = float(elev(*pm)); top = float(sample(edge_grid, gref, *(pm - out * .8)))
        if top - base < 6: continue
        h = min(4.5, .6 * (top - base) - 1.5); ang = np.linspace(0, np.pi, 9)
        prof = np.r_[[[-1.5, 0.]], [[1.5, 0.]], np.c_[1.5 * np.cos(ang), h + 1.5 * np.sin(ang)]]   # sill, jambs, round head
        c0 = (pa + pb) / 2 + out * .06
        AV = np.c_[c0[0] + prof[:, 0] * t[0], c0[1] + prof[:, 0] * t[1], base + .4 + prof[:, 1]]
        AF = [[0, 1, 2]] + [[0, 2 + k, 3 + k] for k in range(len(ang) - 1)]
        m = trimesh.Trimesh(AV, AF, process=False)
        if np.dot(m.face_normals[:, :2].mean(0), out) < 0: m.faces = m.faces[:, ::-1]
        put(m, 'facades__' + fid + '__landmark', 'recess'); arch += 1
    record['arcadeArches'] = arch
    if view:
        for slot, z1 in ctx['envelopes'].lids(view, 0.):
            lid = make_valid(slot).intersection(main)
            if not lid.is_empty: ctx['extrude'](lid, z0 + z1 - .10, .10, 'roofs__' + fid, 'roof')
    lap('walls')
    # --- Field surfaces, stacked by real height so they never share a depth.
    turf_uv = (lambda V: V / TURF_TILE)
    if fit_plate is not None:
        p0, f0, r0 = diamond.frame(*fit_plate); e1 = (f0 + r0) / math.sqrt(2); e3 = (f0 - r0) / math.sqrt(2)
        turf_uv = (lambda V: np.c_[(V - p0) @ e1, (V - p0) @ e3] / TURF_TILE)     # mowing check square to the base lines
    flat_surface(ctx, field, .04, 'landscape', 'clay', lambda V: V / 4.)
    mapped = unary_union([w['g'] for w in ctx['ways'] if w['tags'].get('leisure') == 'pitch' and w['g'].geom_type == 'Polygon'
                          and w['g'].intersection(field).area > .5 * w['g'].area]).intersection(field)
    turf = mapped
    if fit_plate is not None:
        # Grass is as bright as the fitted infield grass (at least its dimmest 5 %): the warning track is a third, darker
        # surface, not grass. Cells without returns take their neighbours' value; the result is cleaned at 2 m.
        t_in = diamond.template(fit_plate[0], fit_plate[1], xs, ys, record['infield']['basePaths'], record['infield']['foulCutouts'])
        thr = float(np.nanpercentile(R[np.isfinite(t_in) & (t_in < .5)], 5))
        inside = shapely.contains_xy(field.buffer(-1), xs, ys)
        Rf = R[tuple(ndi.distance_transform_edt(~np.isfinite(R), return_distances=False, return_indices=True))]
        grass = inside & (Rf >= thr)
        grass = ndi.binary_closing(ndi.binary_opening(grass, iterations=4), iterations=4) & inside
        record['grassThreshold'] = round(thr)
        from rasterio.features import shapes as _shapes
        from rasterio.transform import from_origin
        from shapely.geometry import shape as _shape
        polys_g = [_shape(gm) for gm, v in _shapes(grass.astype(np.uint8)[::-1], mask=grass[::-1], transform=from_origin(x0, y0 + ny * st, st, st)) if v == 1]
        measured = unary_union(polys_g).buffer(0).simplify(.3).intersection(field) if polys_g else Polygon()
        if measured.area > .6 * max(mapped.area, 1): turf = measured
        record['turfSource'] = 'return intensity' if turf is measured else 'mapped pitch outline'
    if turf.is_empty: turf = field.buffer(-4.5)
    flat_surface(ctx, turf, .08, 'landscape', 'turf', turf_uv)
    record['turfM2'] = round(turf.area)
    if fit_plate is not None:
        L = diamond.layout(fit_plate[0], fit_plate[1], record['infield']['basePaths'], record['infield']['foulCutouts'])
        flat_surface(ctx, L['dirt'].intersection(field), .12, 'landscape', 'clay', lambda V: V / 4.)
        flat_surface(ctx, L['grass'], .16, 'landscape', 'turf', turf_uv)
        flat_surface(ctx, L['chalk'].intersection(field.buffer(-.3)), .19, 'landscape', 'white', lambda V: V)
        flat_surface(ctx, L['home'], .21, 'landscape', 'white', lambda V: V)
        for bag in _polys(L['bases']):
            cxy = np.array(bag.centroid.coords[0]); cube([cxy[0], cxy[1], float(elev(*cxy)) + .22], [.38, .38, .08], 'landscape', 'white', fit_plate[1] + math.pi / 4)
        # The mound: a low clay dome with the rubber on top.
        mc = diamond.local_to_world(*fit_plate, [0, diamond.MOUND_C])[0]; base_z = float(elev(*mc)) + .12
        rs = np.linspace(0, 1, 7); ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring_v = [np.c_[mc[0] + r * diamond.MOUND_R * np.cos(ang), mc[1] + r * diamond.MOUND_R * np.sin(ang), np.full(32, base_z + diamond.MOUND_H * (1 - r * r))] for r in rs[1:]]
        MV = np.r_[[[mc[0], mc[1], base_z + diamond.MOUND_H]], np.concatenate(ring_v)]; MF = [[0, 1 + j, 1 + (j + 1) % 32] for j in range(32)]
        for i in range(len(ring_v) - 1):
            for j in range(32):
                p, q = 1 + i * 32 + j, 1 + i * 32 + (j + 1) % 32; MF += [[p, p + 32, q + 32], [p, q + 32, q]]
        put(trimesh.Trimesh(MV, MF, process=False), 'landscape', 'clay')
        rub = diamond.local_to_world(*fit_plate, [0, diamond.RUBBER + .076])[0]
        cube([rub[0], rub[1], base_z + diamond.MOUND_H + .03], [.61, .152, .06], 'landscape', 'white', fit_plate[1] + math.pi / 2)

    # --- Light rigs: the standing clusters above the roofs, rebuilt as lamp banks on the posts the scan saw.
    rigs = []
    if groups:
        n_c = len(groups)
        plate_xy = fit_plate[0] if fit_plate is not None else np.array(field.centroid.coords[0])
        for k in range(n_c):
            q = P[groups[k]]
            if any(np.hypot(*(q[:, :2].mean(0) - seed)) < pr['radiusM'] for pr, seed, _, _ in props_in): continue
            xy = q[:, :2] - q[:, :2].mean(0); ev, vec = np.linalg.eigh(xy.T @ xy); ax = vec[:, -1]; wd = vec[:, 0]
            Lk = float(np.ptp(xy @ ax)); Wk = float(np.ptp(xy @ wd)); ctr = q[:, :2].mean(0)
            topz = float(np.percentile(q[:, 2], 97)); lowz = float(np.percentile(q[:, 2], 3)); zs = float(sample(smooth, gref, *ctr))
            facing_v = plate_xy - ctr; facing_v /= max(np.linalg.norm(facing_v), 1e-6)
            if np.dot(wd, facing_v) < 0: wd = -wd
            ang = math.atan2(ax[1], ax[0]); kind = None
            # Carried: returns between the surface and the bank's underside (posts, a mast). A cluster with nothing
            # under it (a crane jib, a passing object) is not built.
            foot = np.hypot(P[:, 0] - ctr[0], P[:, 1] - ctr[1]) < max(3., Lk / 2)
            carried = int(((P[:, 2] < lowz - 1) & foot & (P[:, 2] > sample(smooth, gref, P[:, 0], P[:, 1]) + 3)).sum())
            if Wk <= 6 and Lk >= 4 and carried >= 20: kind = 'lights'
            rigs.append({'kind': kind, 'centre': [round(float(v), 1) for v in ctr], 'lengthM': round(Lk, 1), 'widthM': round(Wk, 1), 'topM': round(topz - z0, 1), 'standsOnM': round(zs - z0, 1), 'carryingReturns': carried})
            key = 'facades__' + fid + '__landmark'
            if kind != 'lights': continue
            bank = min(6., max(2.5, topz - lowz)); bz = topz - bank / 2
            cube([ctr[0], ctr[1], bz], [Lk, .7, bank], key, 'verdigris', ang)
            for u in np.arange(-Lk / 2 + .7, Lk / 2 - .6, 1.25):
                for vz in np.arange(bz - bank / 2 + .6, bz + bank / 2 - .4, 1.1):
                    lp = ctr + ax * u + wd * .4; cube([lp[0], lp[1], vz], [.8, .12, .7], key, 'lamp', ang)
            posts = [0.] if Lk < 12 else [-.38 * Lk, 0., .38 * Lk]
            for u in posts:
                pp = ctr + ax * u; foot = float(sample(smooth, gref, *pp))
                if topz - bank - foot > .5: cube([pp[0], pp[1], (foot + topz - bank) / 2], [.7, .7, topz - bank - foot], key, 'verdigris', ang)
    record['standingStructures'] = rigs

    # --- Authored props, placed and sized by their own LiDAR clusters.
    props = []
    for pr, seed, q, base_z in props_in:
        q = q[:, :3]
        if len(q) < 30: props.append({'type': pr['type'], 'status': 'no LiDAR cluster'}); continue
        key = 'facades__' + fid + '__landmark'
        if pr['type'] == 'bottle':
            # Axis by trimmed principal components: returns from the deck under it fall off the axis and are dropped.
            keep = np.ones(len(q), bool)
            for _ in range(4):
                ctr = q[keep].mean(0); ev, vec = np.linalg.eigh((q[keep] - ctr).T @ (q[keep] - ctr)); ax = vec[:, -1]
                off = np.linalg.norm((q - ctr) - ((q - ctr) @ ax)[:, None] * ax, axis=1); keep = off <= np.percentile(off[keep], 75)
            if ax[2] < 0: ax = -ax
            s_all = (q - ctr) @ ax; s = s_all[keep]
            radial = np.linalg.norm((q - ctr) - s_all[:, None] * ax, axis=1)
            base_pt = ctr + ax * s.min(); tip = ctr + ax * s.max()
            # Returns lie on the skin, plus some from the deck beside it: the skin radius is the median in each 3 m slice
            # along the axis; the body is the wider slices.
            slices = [np.median(radial[(s_all >= b0) & (s_all < b0 + 3)]) for b0 in np.arange(s.min(), s.max(), 3.) if ((s_all >= b0) & (s_all < b0 + 3)).sum() >= 30]
            r = float(np.percentile(slices, 70)) if slices else float(np.median(radial))
            put(lathe(base_pt, tip, BOTTLE, r), key, 'bottle')
            cap = lathe(tip - ax * .02 * (s.max() - s.min()), tip + ax * .02 * (s.max() - s.min()), [(0, .42), (1, .42)], r)
            put(cap, key, 'redpaint')
            props.append({'type': 'bottle', 'lengthM': round(float(s.max() - s.min()), 1), 'radiusM': round(r, 2), 'tiltDeg': round(math.degrees(math.asin(min(1, ax[2]))), 1), 'topM': round(float(tip[2] - z0), 1)})
        elif pr['type'] == 'board':
            # The slab its cluster describes, with a screen on the face toward the plate.
            xy = q[:, :2] - q[:, :2].mean(0); ev, vec = np.linalg.eigh(xy.T @ xy); ax = vec[:, -1]; wd = vec[:, 0]
            Lb = float(np.ptp(xy @ ax)); ctr = q[:, :2].mean(0); topz = float(np.percentile(q[:, 2], 98)); H = topz - base_z
            face = (fit_plate[0] if fit_plate is not None else np.array(field.centroid.coords[0])) - ctr
            if np.dot(wd, face) < 0: wd = -wd
            ang = math.atan2(ax[1], ax[0]); depth = 3.
            cube([ctr[0], ctr[1], base_z + H / 2], [Lb, depth, H], key, 'steel', ang)
            sc = ctr + wd * (depth / 2 + .06); cube([sc[0], sc[1], base_z + H * .64], [Lb - 1.6, .1, H * .6], key, 'screen', ang)
            props.append({'type': 'board', 'lengthM': round(Lb, 1), 'heightM': round(H, 1), 'topM': round(topz - z0, 1)})
        elif pr['type'] == 'glove':
            ctr = q[:, :2].mean(0); lo_z = float(np.percentile(q[:, 2], 3)); hi_z = float(np.percentile(q[:, 2], 97)); H = max(3., hi_z - lo_z)
            xy = q[:, :2] - ctr; ev, vec = np.linalg.eigh(xy.T @ xy); ax = vec[:, -1]; Wd = float(np.ptp(xy @ ax)); Dp = float(np.ptp(xy @ vec[:, 0]))
            face = (fit_plate[0] if fit_plate is not None else np.array(field.centroid.coords[0])) - ctr; face /= np.linalg.norm(face)
            palm = trimesh.creation.icosphere(3); palm.apply_scale([Wd * .42, max(1.2, Dp * .35), H * .38])
            T4 = np.eye(4); ang = math.atan2(face[1], face[0]) + math.pi / 2; T4[:2, :2] = [[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]; T4[:3, 3] = [ctr[0], ctr[1], lo_z + H * .38]
            palm.apply_transform(T4); put(palm, key, 'leather')
            side = np.array([math.cos(ang), math.sin(ang)])
            for i, u in enumerate(np.linspace(-.3, .3, 4)):
                fb = np.r_[ctr + side * u * Wd, lo_z + H * .62]; ft = fb + np.r_[side * u * Wd * .25, H * (.34 if i in (1, 2) else .28)]
                cylinder(fb, ft, Wd * .075, key, 'leather', 12)
            thumb_b = np.r_[ctr + side * Wd * .42, lo_z + H * .35]; cylinder(thumb_b, thumb_b + np.r_[side * Wd * .12 + face * .5, H * .22], Wd * .08, key, 'leather', 12)
            props.append({'type': 'glove', 'widthM': round(Wd, 1), 'heightM': round(H, 1), 'standsOnM': round(lo_z - z0, 1)})
    record['props'] = props
    lap('props')

    # --- Ground beyond the model boundary under and around the landmark.
    apron = site.buffer(lm.get('apronM', 12)).difference(ctx['boundary'].buffer(-1.))
    if not apron.is_empty: ctx['surface'](apron, 'terrain', 'sidewalk', -.01)
    record['extendedBeyondBoundaryM2'] = round(site.difference(ctx['boundary']).area)
    lap('apron'); record['timingS'] = timing
    return {'height': float(Z.max() - z0), 'roofParts': int(sum(v.any() for v in kinds.values())), 'site': site, 'field': field, 'record': record}
