"""Ballpark landmark builder.

Measured:
- The stands start from the LiDAR surface: the highest return per cell, with raised structures narrower than 6.5 m
  (masts, rigs, the board, rails) replaced by what they stand on and a 2.5 m median against noise. Seating (cells
  raked toward the field edge) stays that surface, triangulated inside the footprint, rows parallel to the field edge,
  with skirts at its open edges. Everything else is built as floors: the surface's plateau heights give the floor
  levels, every other cell takes the nearest one, and the levels become nested outlines with vertical walls (brick on
  the site outline, pierced by an arcade; toward the field concrete, glazed at suite height; back walls above the outline
  wall dark metal cladding with a braced steel frame, as the photographs show the structure behind the decks) and flat roofs. A draped surface over the whole structure read as a brick mountain with drips wherever it was steep.
- The playing field is the footprint's opening: warning-track clay, the mapped turf, and a regulation infield placed
  by fitting its dirt/grass pattern to the return intensity (dirt reads darker than grass).
- Light rigs and the outfield board are the LiDAR clusters standing above the filtered surface, rebuilt as lamp banks
  on posts and a board with a screen facing the plate.
Authored: the shapes of the props named in the landmark entry (a contour bottle, a four-finger glove); their position,
axis and size come from their own LiDAR clusters.
"""
import collections
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


def seat_cells(H, gref, inside, ring, min_m2=60.):
    """Raked seating on the surface grid: cells sloping 0.1-1.35 down toward the field edge, cleaned at 1 m and kept in
    sections of at least `min_m2`."""
    x0, y1, st = gref; ny, nx = H.shape
    XX, YY = np.meshgrid(x0 + (np.arange(nx) + .5) * st, y1 - (np.arange(ny) + .5) * st)
    gy, gx = np.gradient(H, st); gy = -gy                                 # grid rows run north to south
    slope = np.hypot(gx, gy); xy = np.c_[XX.ravel(), YY.ravel()]
    d, i = cKDTree(ring).query(xy); to_f = (ring[i] - xy) / np.maximum(d, 1e-6)[:, None]
    down = -np.c_[gx.ravel(), gy.ravel()] / np.maximum(slope.ravel(), 1e-6)[:, None]
    seat = inside & (slope > .1) & (slope < 1.35) & ((down * to_f).sum(1) > .35).reshape(ny, nx)
    seat = ndi.binary_closing(ndi.binary_opening(seat, iterations=2), iterations=2) & inside
    lab, _ = ndi.label(seat); keep = np.bincount(lab.ravel()) * st * st >= min_m2; keep[0] = False
    return keep[lab]


def levels_of(h, st, min_m2=60., merge=1.5, min_gap=3.):
    """The floor levels of a structure: area-weighted peaks of its plateau heights (a 1.25 m window), each level the
    median of the heights round its peak; levels closer than `min_gap` (less than a storey) become one, weighted by the
    area each holds."""
    if len(h) == 0: return []
    b = np.arange(np.floor(h.min()) - 1, np.ceil(h.max()) + 1.25, .25); hist, e = np.histogram(h, bins=b); c = (e[:-1] + e[1:]) / 2
    sm = np.convolve(hist, np.ones(5), 'same') * st * st
    peaks = sorted((k for k in range(1, len(sm) - 1) if sm[k] >= sm[k - 1] and sm[k] > sm[k + 1] and sm[k] >= min_m2), key=lambda k: -sm[k]); out = []
    for k in peaks:
        if all(abs(c[k] - q) >= merge for q in out): out.append(c[k])
    lv = sorted(float(np.median(h[np.abs(h - q) < merge / 2])) for q in out)
    while len(lv) > 1:
        gaps = np.diff(lv); k = int(np.argmin(gaps))
        if gaps[k] >= min_gap: break
        wa, wb = (np.abs(h - lv[k]) < merge / 2).sum(), (np.abs(h - lv[k + 1]) < merge / 2).sum()
        lv[k:k + 2] = [lv[k] if wa >= wb else lv[k + 1]]                      # the floor that holds more area stays
    return lv


def level_cells(H, flat, st, inside, seat):
    """Every structure cell that is not seating takes the nearest floor level; a 7 x 7 majority and the clutter rule
    clean the map, and specks or narrow strips held by seating become part of the seat surface (a step in it, not a
    block). Returns (levels, level index per cell or -1, seat mask)."""
    ns = inside & ~seat; lv = levels_of(H[ns & flat], st)
    if not lv: return lv, np.full(H.shape, -1), seat
    idx = np.argmin(np.abs(H[..., None] - np.array(lv)[None, None, :]), axis=2); idx[~ns] = -1
    K = len(lv)
    for _ in range(2):
        pad = np.pad(idx, 3, constant_values=-1)
        counts = np.zeros((K,) + H.shape, np.int16)
        for dy in range(7):
            for dx in range(7):
                win = pad[dy:dy + H.shape[0], dx:dx + H.shape[1]]
                for k in range(K): counts[k] += win == k
        idx = np.where(ns, counts.argmax(0), -1)
    # Pieces of a floor under 60 m2, and pieces under 150 m2 held by one other floor (four fifths of their rim), are
    # clutter on that floor (kiosks, planters, terrace furniture): they take the floor round them, unless they stand more
    # than 6 m above it (a rig base or a stair tower is a structure, not clutter).
    for _ in range(3):
        changed = False
        for k in range(K):
            lab, n = ndi.label(idx == k); sz = np.bincount(lab.ravel()) * st * st
            for j in np.flatnonzero((sz < 150) & (np.arange(len(sz)) > 0)):
                m = lab == j; rim = ndi.binary_dilation(m, iterations=2) & ~m & ns & (idx >= 0) & (idx != k)
                if not rim.any(): continue
                cnt = np.bincount(idx[rim], minlength=K); best = int(cnt.argmax())
                if lv[k] - lv[best] > 6. and sz[j] >= 20: continue           # a structure standing on the floor (a rig base, a stair tower) stays
                if sz[j] < 60 or cnt[best] >= .8 * (ndi.binary_dilation(m, iterations=2) & ~m).sum(): idx[m] = best; changed = True
        if not changed: break
    for k in range(K):
        m = idx == k; core = ndi.binary_opening(m, iterations=2); lab, n = ndi.label(m)
        for j in range(1, n + 1):
            c = lab == j; rim = ndi.binary_dilation(c, iterations=2) & ~c
            if (seat & rim).sum() >= .6 * max(rim.sum(), 1) and (c.sum() * st * st < 40 or not (core & c).any()):
                seat = seat | c; idx[c] = -1
    return lv, idx, seat


def polygonize(mask, gref, sigma_m=.5):
    """Outline of a cell mask as the 0.5 contour of the mask blurred by `sigma_m` (marching squares, saddles resolved by
    the cell centre): an edge at any angle to the grid comes out straight and where the mask's edge is, not as a
    staircase of cell sides, and anything narrower than about 2 sigma drops out."""
    from shapely.ops import polygonize as _faces
    x0, y1, st = gref
    F = np.pad(ndi.gaussian_filter(mask.astype(np.float32), sigma_m / st), 2, constant_values=0.)
    a, b, c, d = F[:-1, :-1], F[:-1, 1:], F[1:, 1:], F[1:, :-1]                 # corners: top-left, top-right, bottom-right, bottom-left
    code = (a > .5) * 8 + (b > .5) * 4 + (c > .5) * 2 + (d > .5) * 1
    ii, jj = np.nonzero((code > 0) & (code < 15))
    if not len(ii): return Polygon()
    A, B, C, D = a[ii, jj], b[ii, jj], c[ii, jj], d[ii, jj]; k = code[ii, jj]; centre = (A + B + C + D) / 4 > .5
    t = lambda p, q: np.clip((.5 - p) / np.where(np.abs(q - p) < 1e-9, 1e-9, q - p), 0, 1)
    pts = {'T': np.c_[ii, jj + t(A, B)], 'R': np.c_[ii + t(B, C), jj + 1], 'B': np.c_[ii + 1, jj + t(D, C)], 'L': np.c_[ii + t(A, D), jj]}
    table = {1: ['LB'], 2: ['BR'], 3: ['LR'], 4: ['TR'], 6: ['TB'], 7: ['LT'], 8: ['LT'], 9: ['TB'], 11: ['TR'], 12: ['LR'], 13: ['BR'], 14: ['LB']}
    segs = []
    for code_v in range(1, 15):
        sel = k == code_v
        if not sel.any(): continue
        if code_v in (5, 10):
            inside_joined = centre[sel]
            one = ['LT', 'BR'] if code_v == 5 else ['TR', 'LB']; other = ['TR', 'LB'] if code_v == 5 else ['LT', 'BR']
            for pair_set, m in ((one, inside_joined), (other, ~inside_joined)):
                for e in pair_set: segs.append(np.stack([pts[e[0]][sel][m], pts[e[1]][sel][m]], 1))
        else:
            for e in table[code_v]: segs.append(np.stack([pts[e[0]][sel], pts[e[1]][sel]], 1))
    S = np.concatenate(segs)                                                    # (n, 2 ends, row/col)
    S = np.stack([x0 + (S[..., 1] - 2 + .5) * st, y1 - (S[..., 0] - 2 + .5) * st], -1)
    S = S[np.hypot(*(S[:, 1] - S[:, 0]).T) > 1e-6]
    faces = list(_faces([LineString(q) for q in np.round(S, 6)]))
    keep = []
    for f in faces:
        rp = f.representative_point(); j = int((rp.x - x0) / st + 2 - .5 + .5); i = int((y1 - rp.y) / st + 2 - .5 + .5)
        # a face is solid when the blurred mask is above half at its interior point (bilinear on the padded samples)
        fx = (rp.x - x0) / st - .5 + 2; fy = (y1 - rp.y) / st - .5 + 2; j0, i0 = int(np.floor(fx)), int(np.floor(fy)); u, v = fx - j0, fy - i0
        if 0 <= i0 < F.shape[0] - 1 and 0 <= j0 < F.shape[1] - 1:
            val = F[i0, j0] * (1 - u) * (1 - v) + F[i0, j0 + 1] * u * (1 - v) + F[i0 + 1, j0] * (1 - u) * v + F[i0 + 1, j0 + 1] * u * v
            if val > .5: keep.append(f)
    return sharpen(unary_union(keep).buffer(0).simplify(.15, preserve_topology=True)) if keep else Polygon()


def sharpen(poly, short=2., run=3., keep_angle=(30., 150.)):
    """Square up corners a smoothed outline rounded: a run of short segments (each under `short`, together under `run`)
    between two longer edges that meet at 30-150 degrees is replaced by the point where those edges' lines cross."""
    def ring(cs):
        P = [np.asarray(c, float) for c in cs[:-1]]
        changed = True
        while changed and len(P) > 4:
            changed = False; n = len(P); L = [np.linalg.norm(P[(i + 1) % n] - P[i]) for i in range(n)]
            for i in range(n):
                if L[i] < short: continue
                j = (i + 1) % n; tot = 0.; m = 0
                while L[j] < short and tot + L[j] < run and m < n - 2: tot += L[j]; j = (j + 1) % n; m += 1
                if m == 0 or L[j] < short: continue
                a0, a1 = P[i], P[(i + 1) % n]; b0, b1 = P[j], P[(j + 1) % n]
                da, db = a1 - a0, b1 - b0; cr = da[0] * db[1] - da[1] * db[0]
                ang = math.degrees(math.acos(np.clip(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db)), -1, 1)))
                if abs(cr) < 1e-9 or not (keep_angle[0] <= ang <= keep_angle[1]): continue
                tpar = ((b0 - a0)[0] * db[1] - (b0 - a0)[1] * db[0]) / cr; X = a0 + da * tpar
                if np.linalg.norm(X - (a1 + b0) / 2) > run: continue
                drop = [(i + 1 + q) % n for q in range(m + 1)]                   # a1 .. b0 go, the crossing comes in
                keep = [P[q] for q in range(n) if q not in drop]; at = keep.index(next(v for v in keep if v is a0)) + 1
                keep.insert(at, X); P = keep; changed = True; break
        return P + [P[0]]
    out = []
    for q in _polys(poly):
        try:
            g = Polygon(ring(list(q.exterior.coords)), [ring(list(r.coords)) for r in q.interiors])
            g = make_valid(g); out.extend(_polys(g) if g.area > .5 * q.area else [q])
        except Exception:
            out.append(q)
    return unary_union(out) if out else poly


def band_outlines(idx, levels, gref, stands):
    """Nested outlines, one per floor level: band k is the contour of every cell at level k or above, snapped to the band
    below within 0.45 m and cut to it (walls of consecutive bands stand in one plane), slivers under 0.7 m and pieces
    under 8 m2 dropped."""
    out, prev = [], stands
    for k in range(len(levels)):
        m = idx >= k
        pk = polygonize(m, gref).intersection(stands) if m.any() else Polygon()
        if k and not pk.is_empty: pk = shapely.snap(pk, prev, .45).intersection(prev)
        if not pk.is_empty:
            pk = make_valid(pk.buffer(-.35, join_style=2).buffer(.35, join_style=2).intersection(prev))
            pk = unary_union([q for q in _polys(pk) if q.area >= 8.]) if _polys(pk) else Polygon()
        out.append(pk)
        if not pk.is_empty: prev = pk
    return out


def flat_roof(poly, z, step=2.):
    """A flat roof over a polygon at height z."""
    parts = []
    for q in _polys(poly):
        if q.area < 1.: continue
        V, F = triangulate(q, step, step * 1.5); parts.append(trimesh.Trimesh(np.c_[V, np.full(len(V), z)], F, process=False))
    return trimesh.util.concatenate(parts) if parts else None


def outline_stretches(samples, ring, max_gap=20.):
    """Runs of one wall family along the outline from the outline readings (samples carry their position, so any
    parametrisation of the ring works): same-family runs join across unread gaps up to `max_gap` with nothing
    contradicting them. Returns [(s0, s1, family, colour)] along `ring`."""
    pts = sorted((ring.project(Point(*x['xy'])), x['family'], x.get('colour')) for x in samples if x.get('family') and x.get('xy'))
    out = []
    for sv, fam, col in pts:
        if out and out[-1][2] == fam and sv - out[-1][1] <= max_gap: out[-1][1] = sv; out[-1][4].append(col)
        else: out.append([sv, sv, fam, col, [col]])
    res = []
    for s0, s1, fam, _, cols in out:
        if len(cols) < 3: continue
        rgb = np.median([[int(c[q:q + 2], 16) for q in (1, 3, 5)] for c in cols], 0).astype(int)
        res.append((s0 - 2., s1 + 2., fam, '#%02x%02x%02x' % tuple(rgb)))
    return res


def arch_outline(width, sill, spring, rise, n=12):
    """A window with a segmental head in wall coordinates (u across, z up), counter-clockwise from the sill."""
    hw = width / 2; R = (hw * hw + rise * rise) / (2 * rise); cz = spring + rise - R; a0 = math.atan2(spring - cz, hw)
    arc = [(R * math.cos(a), cz + R * math.sin(a)) for a in np.linspace(a0, math.pi - a0, n)]
    return [(-hw, sill), (hw, sill)] + arc, (R, cz)


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

    # --- Stands: the seating stays the raked LiDAR surface; everything else in the stands is built as floors. The
    # surface's plateau heights give the floor levels, every other cell takes the nearest level, and the levels become
    # nested outlines with vertical walls between them and flat roofs: a draped surface read as a brick mountain with
    # drips wherever it was steep.
    ring = np.asarray(segmentize(field.exterior, .5).coords)[:-1]
    fit_plate = None
    kd = cKDTree(ring)
    gx0, gy1, gst = gref; gny, gnx = smooth.shape
    GX, GY = np.meshgrid(gx0 + (np.arange(gnx) + .5) * gst, gy1 - (np.arange(gny) + .5) * gst)
    in_stands = shapely.contains_xy(stands, GX, GY)
    seat_m = seat_cells(smooth, gref, in_stands, ring)
    _, flat_m = plateaus(smooth, gst)
    levels, lidx, seat_m = level_cells(smooth, flat_m, gst, in_stands, seat_m)
    bands = band_outlines(lidx, levels, gref, stands)
    # The top of the stands per cell: a floor's level, the seat surface where there are seats.
    top_grid = np.where(lidx >= 0, np.array(levels + [0.])[np.maximum(lidx, 0)] if levels else smooth, smooth)
    seat_poly = polygonize(seat_m, gref).intersection(stands)
    seat_body = make_valid(seat_poly.buffer(.3, join_style=2).intersection(stands)) if not seat_poly.is_empty else Polygon()
    # Whatever the outlines left uncovered (pieces the smoothing dropped) joins what it borders: the seat surface, or the
    # floors up to the nearest floor cell's level, so the stands have no holes.
    if bands and not bands[0].is_empty:
        near_idx = lidx[tuple(ndi.distance_transform_edt(lidx < 0, return_distances=False, return_indices=True))]
        gap = make_valid(stands.difference(unary_union([bands[0], seat_body])))
        for q in _polys(gap):
            if q.area < .02: continue
            rp = q.representative_point(); i_, j_ = int(np.clip((gy1 - rp.y) / gst, 0, gny - 1)), int(np.clip((rp.x - gx0) / gst, 0, gnx - 1))
            if not seat_body.is_empty and seat_body.distance(q) + .1 < bands[0].distance(q) or (seat_m[i_, j_] and seat_body.distance(q) < .5):
                seat_body = unary_union([seat_body, q]); continue
            for j in range(int(near_idx[i_, j_]) + 1): bands[j] = unary_union([bands[j], q]) if not bands[j].is_empty else bands[j]
    record['standsLevelsM'] = [round(v - z0, 2) for v in levels]
    # The street wall's material by stretch, read from the panoramas (read_landmark_outline_v29.py): stone or brick in
    # the measured colour; stretches no photograph reads keep brick.
    import json as _json, pathlib as _pl
    ring_ccw = shapely.geometry.polygon.orient(site, 1.0).exterior
    _rp = _pl.Path(__file__).with_name('outline-readings.json')
    stretches = outline_stretches(_json.loads(_rp.read_text()).get(fid, {}).get('samples', []), ring_ccw) if _rp.exists() else []
    record['outlineStretches'] = [{'s0': round(a, 1), 's1': round(b, 1), 'family': f, 'colour': col} for a, b, f, col in stretches]
    def stretch_at(pt):
        sv = ring_ccw.project(Point(*pt))
        return next(((f, col) for a, b, f, col in stretches if a <= sv <= b), (None, None))
    STONE_STOREY_M = 7.5                                   # the stone arcade storey; anything taller on the outline is brick

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
    def along_uv(mat, s_m, z_m):
        # Walls and fronts: u runs along the stands (continuous round every bend), v up the wall, in the material's tiles.
        su, sv = ctx['material_bank'].get(mat)[1]; return np.c_[s_m / su, z_m / sv]
    lap('field fit')
    outer = site.exterior; fring = field.exterior; strip_edge = strip.boundary if not strip.is_empty else None
    def cell(x, y):
        j = np.clip(((np.asarray(x, float) - gx0) / gst).astype(int), 0, gnx - 1); i = np.clip(((gy1 - np.asarray(y, float)) / gst).astype(int), 0, gny - 1)
        return i, j
    def top_at(x, y): return top_grid[cell(x, y)]
    def level_at(x, y): return lidx[cell(x, y)]
    walls = collections.defaultdict(list)
    def add_wall(mat, a, b, top_a, top_b, bot_a, bot_b, u0):
        """Quad from bot to top along a->b, facing right of a->b (outward for a ring with the solid on its left)."""
        if max(top_a - bot_a, top_b - bot_b) < .03: return
        L = float(np.linalg.norm(b - a))
        walls[mat].append((np.array([[*a, bot_a], [*b, bot_b], [*b, top_b], [*a, top_a]], float), np.array([u0, u0 + L, u0 + L, u0])))
    def ring_u(ring_l, a, b):
        sa, sb = ring_l.project(Point(a)), ring_l.project(Point(b))
        return sa if abs(sb - sa) <= ring_l.length / 2 else sb - math.copysign(np.linalg.norm(b - a), sb - sa)
    def outline_wall(pa, pb, top_a, top_b, bot_a, bot_b, u0):
        fam, col = stretch_at((pa + pb) / 2)
        if fam == 'stone':
            cut = float(min(max(top_a, top_b), elev(*((pa + pb) / 2)) + STONE_STOREY_M))
            add_wall('cast_stone' + col, pa, pb, min(top_a, cut), min(top_b, cut), bot_a, bot_b, u0)
            if max(top_a, top_b) > cut + .03: add_wall('brick_modular', pa, pb, top_a, top_b, max(bot_a, cut), max(bot_b, cut), u0)
        else: add_wall('brick_modular' + (col if fam == 'brick' else ''), pa, pb, top_a, top_b, bot_a, bot_b, u0)
    SZ = np.zeros(0)
    # Seats: the raked surface, reaching 0.3 m under the floors beside it (never above a lower floor there), steep
    # steps inside it in concrete, and a skirt down to whatever lies beyond each open edge.
    if not seat_body.is_empty:
        SV, SF = triangulate(seat_body, step)
        SZ = sample(smooth, gref, SV[:, 0], SV[:, 1]); lk = level_at(SV[:, 0], SV[:, 1])
        SZ = np.where(lk >= 0, np.minimum(SZ, np.array(levels + [0.])[np.maximum(lk, 0)] - .02), SZ)
        SZ = np.maximum(SZ, elev(SV[:, 0], SV[:, 1]) + .4); S3 = np.c_[SV, SZ]
        T = S3[SF]; n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        steep = n[:, 2] < .42
        d_v, i_v = kd.query(SV); seat_uv = np.c_[s_along[i_v] / SEAT_TILE[0], d_v / SEAT_TILE[1]]
        put(trimesh.Trimesh(S3, SF[~steep], process=False), 'roofs__' + fid, 'seats', uv=seat_uv)
        if steep.any(): put(trimesh.Trimesh(S3, SF[steep], process=False), 'roofs__' + fid, 'venue_concrete', uv=along_uv('venue_concrete', s_along[i_v], SZ))
        for a_i, b_i in boundary_edges(SF):
            pa, pb = SV[a_i], SV[b_i]; mid = (pa + pb) / 2; pm = Point(mid); t = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
            if fring.distance(pm) < .3: mat, ba, bb, rl = 'padding', zf - .1, zf - .1, fring
            elif outer.distance(pm) < .3: mat, ba, bb, rl = 'outline', float(elev(*pa)) - .2, float(elev(*pb)) - .2, outer
            elif strip_edge is not None and strip_edge.distance(pm) < .3: mat, ba, bb, rl = 'recess', float(elev(*pa)) - .2, float(elev(*pb)) - .2, outer
            else:
                k = int(level_at(*(mid + np.array([t[1], -t[0]]) * .8)))
                ba = bb = levels[k] if k >= 0 else float(elev(*mid)) - .2; mat, rl = 'venue_concrete', outer
            if mat == 'outline': outline_wall(pa, pb, SZ[a_i], SZ[b_i], min(ba, SZ[a_i]), min(bb, SZ[b_i]), ring_u(rl, pa, pb))
            else: add_wall(mat, pa, pb, SZ[a_i], SZ[b_i], min(ba, SZ[a_i]), min(bb, SZ[b_i]), ring_u(rl, pa, pb))
        record['seatsM2'] = round(float(seat_poly.area))
    # Floors: each band's outline walls rise from the band below (the ground for the first) to its level. On the site
    # outline they are the brick outer wall; against the frontage strip a dark recess; toward the field concrete,
    # glazed at suite height (6-24 m); back walls reaching above the street storey (9.5 m) are the dark steel structure
    # with a braced frame, low ones brick. Each band's exposed top is a flat roof.
    caps = collections.Counter(); bands_rec = []; frames = []
    for k, pk in enumerate(bands):
        if pk.is_empty: continue
        ztop = levels[k]
        for q in _polys(pk):
            q = shapely.geometry.polygon.orient(q, 1.0)
            for r in [q.exterior, *q.interiors]:
                cs = np.asarray(r.coords); u = 0.
                for pa, pb in zip(cs[:-1], cs[1:]):
                    L = float(np.linalg.norm(pb - pa))
                    if L < .05: continue
                    mid = (pa + pb) / 2; pm = Point(mid)
                    ba, bb = (levels[k - 1], levels[k - 1]) if k else (float(elev(*pa)) - .2, float(elev(*pb)) - .2)
                    if outer.distance(pm) < .3: outline_wall(pa, pb, ztop, ztop, ba, bb, ring_u(outer, pa, pb))
                    elif strip_edge is not None and strip_edge.distance(pm) < .3: add_wall('recess', pa, pb, ztop, ztop, ba, bb, ring_u(outer, pa, pb))
                    elif fring.distance(pm) < .3: add_wall('padding', pa, pb, ztop, ztop, min(ba, zf - .1), min(bb, zf - .1), ring_u(fring, pa, pb))
                    else:
                        t = (pb - pa) / L; out = np.array([t[1], -t[0]]); d_, i_ = kd.query(mid); tf = (ring[i_] - mid) / max(d_, 1e-6)
                        if out @ tf > .35:
                            cuts = [z0 + 6., z0 + 24.]
                            for lo, hi, mat in ((-1e9, cuts[0], 'venue_concrete'), (cuts[0], cuts[1], 'glass_dark'), (cuts[1], 1e9, 'venue_concrete')):
                                t_top = min(ztop, hi)
                                if t_top > max(ba, bb, lo) + .03: add_wall(mat, pa, pb, t_top, t_top, max(ba, lo), max(bb, lo), u)
                        else:
                            # A back wall reaching above the street storey is the steel structure behind the decks, however
                            # the floors stack it: dark metal cladding with the braced frame in front of it.
                            street = float(elev(*mid)); tall = k and ztop - street > 9.5
                            add_wall('panel_dark' if tall else 'brick_modular', pa, pb, ztop, ztop, ba, bb, u)
                            if tall and ztop - max(ba, bb) > 2.5: frames.append((k, pa, pb, max(ba, bb), ztop))
                    u += L
        nxt = next((bands[j] for j in range(k + 1, len(bands)) if not bands[j].is_empty), None)
        exposed = make_valid(pk.difference(nxt)) if nxt is not None else pk
        mat = 'venue_concrete' if ztop - z0 < 30 else 'roofing'; m = flat_roof(exposed, ztop)
        if m is not None: put(m, 'roofs__' + fid, mat); caps[mat] += exposed.area
        bands_rec.append({'levelM': round(ztop - z0, 2), 'areaM2': round(pk.area), 'pieces': len(_polys(pk))})
    wall_m2 = {}
    for mat, quads in walls.items():
        WV = np.concatenate([q[0] for q in quads]); WF = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + 4 * i for i in range(len(quads))])
        put(trimesh.Trimesh(WV, WF, process=False), 'buildings__' + fid, mat, uv=along_uv(mat, np.concatenate([q[1] for q in quads]), WV[:, 2]))
        wall_m2[mat] = round(float(sum(np.linalg.norm(q[0][1, :2] - q[0][0, :2]) * ((q[0][2, 2] - q[0][1, 2]) + (q[0][3, 2] - q[0][0, 2])) / 2 for q in quads)))
    # Braced frames on the back walls: posts every ~7 m along each run of back wall, X braces between them and three
    # rails, 0.45 m proud of the wall, in the painted steel of the rigs.
    key_f = 'facades__' + fid + '__landmark'; members = 0; runs = []
    for k, pa, pb, zlo, zhi in frames:
        if runs and runs[-1][0] == k and np.allclose(runs[-1][1][-1], pa, atol=1e-6) and abs(runs[-1][2] - zlo) < .05: runs[-1][1].append(pb)
        else: runs.append([k, [pa, pb], zlo, zhi])
    for k, pts_r, zlo, zhi in runs:
        P2 = np.asarray(pts_r); seg = np.diff(P2, axis=0); Ls = np.linalg.norm(seg, axis=1); total = Ls.sum()
        if total < 3.: continue
        cum = np.r_[0, np.cumsum(Ls)]; nposts = max(2, int(round(total / 7.)) + 1)
        posts = []
        for sv in np.linspace(0, total, nposts):
            i = min(int(np.searchsorted(cum, sv, side='right')) - 1, len(seg) - 1); f = (sv - cum[i]) / max(Ls[i], 1e-9)
            t = seg[i] / max(Ls[i], 1e-9); out = np.array([t[1], -t[0]]); posts.append(P2[i] + seg[i] * f + out * .45)
        for q in posts: cylinder([q[0], q[1], zlo], [q[0], q[1], zhi], .16, key_f, 'verdigris', 6); members += 1
        for q0, q1 in zip(posts, posts[1:]):
            cylinder([q0[0], q0[1], zlo], [q1[0], q1[1], zhi], .1, key_f, 'verdigris', 6); cylinder([q1[0], q1[1], zlo], [q0[0], q0[1], zhi], .1, key_f, 'verdigris', 6)
            for zr in (zlo + .15, zhi - .9, zhi - .15): cylinder([q0[0], q0[1], zr], [q1[0], q1[1], zr], .16, key_f, 'verdigris', 6)
            members += 5
    record['standsBands'] = bands_rec; record['wallsM2'] = wall_m2; record['roofsM2'] = {k: round(v) for k, v in caps.items()}; record['bracedMembers'] = members
    lap('stands mesh')
    # Arcade: where the photographs read the street wall as stone, it is the glazed arcade they show: segmental arches
    # (3.0 m wide, sill 0.2 m, springing 4.1 m, rise 0.9 m, 4.5 m apart, measured on rectified panoramas) with a stone
    # surround, a mullion grid under a transom, louvres in the head, and the terrace railing along the wall top.
    # Elsewhere the outline wall keeps the dark arches at a steady pitch, wherever it is tall enough and straight.
    key_l = 'facades__' + fid + '__landmark'; arch = 0; glazed = 0; rail_m = 0.
    def frame_at(sm, half):
        pa, pm, pb = (np.array(ring_ccw.interpolate(v).coords[0]) for v in (sm - half, sm, sm + half))
        t1, t2 = pm - pa, pb - pm
        if np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-9) < .94: return None                  # a corner inside the arch
        t = (pb - pa) / np.linalg.norm(pb - pa); return pa, pm, pb, t, np.array([t[1], -t[0]])
    def flat(poly_uv, c0, t, out, off, base, mat):
        V = np.array([[c0[0] + u * t[0] + out[0] * off, c0[1] + u * t[1] + out[1] * off, base + z] for u, z in poly_uv])
        cu = V.mean(0); V = np.r_[V, [cu]]; n = len(poly_uv); F = [[n, i, (i + 1) % n] for i in range(n)]
        m = trimesh.Trimesh(V, F, process=False)
        if np.dot(m.face_normals[:, :2].mean(0), out) < 0: m.faces = m.faces[:, ::-1]
        put(m, key_l, mat)
    W_A, SILL, SPRING, RISE, PITCH = 3.0, .2, 4.1, .9, 4.5
    for s0, s1, fam, col in stretches:
        if fam != 'stone': continue
        for sm in np.arange(s0 + PITCH / 2, s1 - PITCH / 2 + 1e-6, PITCH):
            fr = frame_at(sm, W_A / 2 + .4)
            if fr is None: continue
            pa, pm, pb, t, out = fr
            if not strip.is_empty and strip.buffer(1.).contains(Point(*pm)): continue
            base = float(elev(*pm)); top = float(top_at(*(pm - out * .8)))
            if top - base < SPRING + RISE + .6: continue
            c0 = pm
            ring_o, (r_arc, cz) = arch_outline(W_A + .5, SILL, SPRING, RISE + .12)            # the stone surround
            flat(ring_o, c0, t, out, .05, base, 'cast_stone' + col)
            win, (r_arc, cz) = arch_outline(W_A, SILL, SPRING, RISE)
            flat(win, c0, t, out, .09, base, 'glass_dark')
            ang_t = math.atan2(t[1], t[0])
            def bar(u, z, w, h, mat='charcoal'): cube([c0[0] + t[0] * u + out[0] * .12, c0[1] + t[1] * u + out[1] * .12, base + z], [w, .05, h], key_l, mat, ang_t)
            tr = SPRING - .7                                                                  # transom; louvres above it
            for u in (-W_A / 2, -W_A / 4, 0., W_A / 4, W_A / 2): bar(u, (SILL + tr) / 2, .07, tr - SILL)
            for z in (SILL + .03, 1.45, tr): bar(0., z, W_A, .07)
            for z in np.arange(tr + .16, SPRING + RISE - .05, .17):
                hw = W_A / 2 if z <= SPRING else math.sqrt(max(r_arc * r_arc - (z - cz) ** 2, 0.))
                if hw > .15: bar(0., z, 2 * hw - .1, .05)
            for (u1, z1), (u2, z2) in zip(win[1:], win[2:]):                                 # the frame round the head
                cylinder([c0[0] + t[0] * u1 + out[0] * .12, c0[1] + t[1] * u1 + out[1] * .12, base + z1], [c0[0] + t[0] * u2 + out[0] * .12, c0[1] + t[1] * u2 + out[1] * .12, base + z2], .04, key_l, 'charcoal', 5)
            arch += 1; glazed += 1
        # the terrace railing along the stone storey's top
        for sv in np.arange(s0, s1, 1.8):
            p0, p1 = np.array(ring_ccw.interpolate(sv).coords[0]), np.array(ring_ccw.interpolate(min(s1, sv + 1.8)).coords[0])
            d = p1 - p0; L = float(np.linalg.norm(d))
            if L < .2: continue
            out = np.array([d[1], -d[0]]) / L; q0, q1 = p0 - out * .15, p1 - out * .15
            base = float(elev(*p0)); top = float(top_at(*(p0 - out * .8)))
            if not (4.5 < top - base < 9.): continue
            cylinder([q0[0], q0[1], top], [q0[0], q0[1], top + 1.05], .03, key_l, 'steel', 5)
            for zr in (top + 1.05, top + .55): cylinder([q0[0], q0[1], zr], [q1[0], q1[1], zr], .025, key_l, 'steel', 5)
            rail_m += L
    in_stone = lambda sm: any(a <= sm <= b for a, b, f, _ in stretches if f == 'stone')
    for sm in np.arange(2.6, ring_ccw.length - 2.6, lm.get('arcadePitchM', 5.2)):
        if in_stone(sm): continue
        fr = frame_at(sm, 1.5)
        if fr is None: continue
        pa, pm, pb, t, out = fr
        if not strip.is_empty and strip.buffer(1.).contains(Point(*pm)): continue
        base = float(elev(*pm)); top = float(top_at(*(pm - out * .8)))
        if top - base < 6: continue
        h = min(4.5, .6 * (top - base) - 1.5); ang = np.linspace(0, np.pi, 9)
        prof = np.r_[[[-1.5, 0.]], [[1.5, 0.]], np.c_[1.5 * np.cos(ang), h + 1.5 * np.sin(ang)]]   # sill, jambs, round head
        c0 = (pa + pb) / 2 + out * .06
        AV = np.c_[c0[0] + prof[:, 0] * t[0], c0[1] + prof[:, 0] * t[1], base + .4 + prof[:, 1]]
        AF = [[0, 1, 2]] + [[0, 2 + k, 3 + k] for k in range(len(ang) - 1)]
        m = trimesh.Trimesh(AV, AF, process=False)
        if np.dot(m.face_normals[:, :2].mean(0), out) < 0: m.faces = m.faces[:, ::-1]
        put(m, key_l, 'recess'); arch += 1
    record['glazedArches'] = glazed; record['terraceRailingM'] = round(rail_m)
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
            topz = float(np.percentile(q[:, 2], 97)); lowz = float(np.percentile(q[:, 2], 3)); zs = float(top_at(*ctr))
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
                pp = ctr + ax * u; foot = float(top_at(*pp))
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
    top = max([*levels, float(SZ.max()) if len(SZ) else z0])
    return {'height': float(top - z0), 'roofParts': len([b for b in bands if not b.is_empty]) + (1 if len(SZ) else 0), 'site': site, 'field': field, 'record': record}
