"""Split one footprint into the buildings it actually holds, from the measured roof alone.

A mapped outline often covers several buildings: a low block and a tall one sharing a party wall. Treated as one,
the footprint takes a single base roof (its low part), so the tall building's walls and facades stop at the low
roof and everything above becomes stepped, windowless tiers. Here the roof raster is grown into regions whose
neighbouring cells differ by little (a pitched or sloped roof stays one region; a storey-high step separates
regions). Regions large enough to be buildings, reaching the outline, and standing at clearly different heights
become separate parts; every other cell joins the nearest part. No location or identity is used.
"""
import numpy as np
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio.features import shapes, rasterize
from rasterio.transform import from_origin

SMOOTH_STEP_M = 1.5      # neighbouring roof cells closer than this belong to one surface (slopes stay whole)
MIN_SEPARATION_M = 4.0   # parts must stand at least this far apart in median height: more than a storey
MIN_PART_M2 = 150.       # a part is a building, not a penthouse or a bay
MIN_PART_SHARE = .10     # ... and holds at least this share of the footprint
MIN_OUTLINE_M = 8.       # ... and reaches the outline for at least this length (it has a street wall)


def _grow(h, valid):
    """Connected regions where 4-neighbours differ by less than SMOOTH_STEP_M (region growing by union-find)."""
    H, W = h.shape; parent = np.arange(H * W)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    idx = np.arange(H * W).reshape(H, W)
    for dy, dx in ((0, 1), (1, 0)):
        a = valid[:H - dy, :W - dx] & valid[dy:, dx:]
        a &= np.abs(h[:H - dy, :W - dx] - h[dy:, dx:]) < SMOOTH_STEP_M
        for i, j in zip(idx[:H - dy, :W - dx][a], idx[dy:, dx:][a]):
            ri, rj = find(i), find(j)
            if ri != rj: parent[ri] = rj
    lab = np.array([find(i) for i in range(H * W)]).reshape(H, W)
    lab[~valid] = -1
    return lab


def split(footprint, points, z0, cell=2.0, audit=None):
    """Return a list of part polygons (the footprint itself when it holds one building). The raster runs along the
    outline's own axes: on a north-up grid a boundary parallel to a 45-degree frontage became a staircase, and the
    cells along a gable's eaves, cut diagonally, broke off the roof and went to the neighbouring part."""
    from shapely import affinity
    import math
    rect = footprint.minimum_rotated_rectangle; xy = np.asarray(rect.exterior.coords)[:4]
    e = xy[1] - xy[0] if np.linalg.norm(xy[1] - xy[0]) >= np.linalg.norm(xy[2] - xy[1]) else xy[2] - xy[1]
    ang = math.atan2(e[1], e[0]); o = footprint.centroid
    if abs(math.sin(2 * ang)) < 1e-3: return _split(footprint, points, z0, cell, audit)
    c, s_ = math.cos(-ang), math.sin(-ang); rel = points[:, :2] - [o.x, o.y]
    rp = points.copy(); rp[:, 0] = o.x + rel[:, 0] * c - rel[:, 1] * s_; rp[:, 1] = o.y + rel[:, 0] * s_ + rel[:, 1] * c
    parts = _split(affinity.rotate(footprint, -ang, origin=o, use_radians=True), rp, z0, cell, audit)
    out = [affinity.rotate(p, ang, origin=o, use_radians=True).intersection(footprint) for p in parts]
    if len(out) < 2: return [footprint]
    rest = footprint.difference(unary_union(out))                  # rotation round-off: slivers to the part they touch most
    for piece in getattr(rest, 'geoms', [rest]):
        if piece.is_empty or piece.area < 1e-6: continue
        j = int(np.argmax([o_.buffer(.05).intersection(piece.buffer(.05)).area for o_ in out])); out[j] = unary_union([out[j], piece])
    return out


def _split(footprint, points, z0, cell=2.0, audit=None):
    if len(points) < 200 or footprint.area < 2 * MIN_PART_M2: return [footprint]
    xmin, ymin, xmax, ymax = footprint.bounds
    W = max(2, int(np.ceil((xmax - xmin) / cell))); H = max(2, int(np.ceil((ymax - ymin) / cell)))
    tr = from_origin(xmin, ymax, cell, cell)
    inside = rasterize([(footprint, 1)], out_shape=(H, W), transform=tr, fill=0).astype(bool)
    col = np.clip(((points[:, 0] - xmin) / cell).astype(int), 0, W - 1); row = np.clip(((ymax - points[:, 1]) / cell).astype(int), 0, H - 1)
    h = np.full((H, W), -np.inf); np.maximum.at(h, (row, col), points[:, 2] - z0)
    # a roof surface, not its antennas: the median filter removes single-cell spikes
    known = np.isfinite(h) & inside & (h > 3)
    hf = np.where(known, h, 0.); hf = ndimage.median_filter(hf, size=3); hf[~known] = np.nan
    valid = known & np.isfinite(hf)
    lab = _grow(np.nan_to_num(hf), valid)
    edge = inside & ~ndimage.binary_erosion(inside, iterations=2)
    parts = []
    for k in np.unique(lab[lab >= 0]):
        m = lab == k; area = m.sum() * cell * cell
        if area < max(MIN_PART_M2, MIN_PART_SHARE * footprint.area): continue
        if (m & edge).sum() * cell < MIN_OUTLINE_M: continue
        parts.append((area, float(np.median(hf[m])), m))
    parts.sort(key=lambda p: -p[0])
    kept = []
    for p in parts:
        if all(abs(p[1] - q[1]) >= MIN_SEPARATION_M for q in kept): kept.append(p)
    if audit is not None: audit.update(regions=len(parts), parts=len(kept), heights=[round(p[1], 1) for p in kept])
    if len(kept) < 2: return [footprint]
    seed = np.full((H, W), -1)
    for i, (_, _, m) in enumerate(kept): seed[m] = i
    near = ndimage.distance_transform_edt(seed < 0, return_distances=False, return_indices=True)
    owner = seed[near[0], near[1]]; owner[~inside] = -1
    # A measured cell outside every part's core joins the part standing at its height among those within three cells
    # (a gable's lower slope beside its ridge core, not the taller block that happens to be as close); cells with no
    # roof return keep the nearest part.
    medians = np.array([p[1] for p in kept]); dist_to = [ndimage.distance_transform_edt(seed != i) for i in range(len(kept))]
    loose = (seed < 0) & inside & valid
    for r_, c_ in zip(*np.nonzero(loose)):
        cand = [i for i in range(len(kept)) if dist_to[i][r_, c_] <= 3]
        if len(cand) > 1: owner[r_, c_] = min(cand, key=lambda i: abs(hf[r_, c_] - medians[i]))
    out = []
    for i in range(len(kept)):
        mask = (owner == i).astype('uint8')
        g = unary_union([shape(s) for s, v in shapes(mask, mask=mask.astype(bool), transform=tr) if v])
        g = g.buffer(0).intersection(footprint).buffer(-.3, join_style=2).buffer(.3, join_style=2).intersection(footprint)
        g = g.simplify(cell * .5, preserve_topology=True).intersection(footprint)
        if g.area >= MIN_PART_M2: out.append(g)
    if len(out) < 2: return [footprint]
    # cells lost to cleaning go to the nearest part, so the parts tile the footprint exactly
    rest = footprint.difference(unary_union(out))
    for piece in getattr(rest, 'geoms', [rest]):
        if piece.is_empty or piece.area < 1e-6: continue
        # to the part it shares most boundary with (a tie on distance went to the largest part: the tall block took
        # the strips beside a low building and stood its walls there)
        j = int(np.argmax([o.buffer(.05).intersection(piece.buffer(.05)).area for o in out])); out[j] = unary_union([out[j], piece])
    return out
