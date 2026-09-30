"""Fences and other thin low structures between the street and a wall, from the LiDAR (v36).

The photograph readers mask what stands in front of a wall: other buildings (frontage_evidence_v34.building_surface,
solid mass inside footprints) and what the segmenter labels (trees, cars, people, fences). A chain-link fence is see-
through: the segmenter does not label it, and the detector read a 12.8 m low wall seen through one as three shopfronts
from the ground to the wall's top. The LiDAR sees such a fence as a thin line of returns 2.0-3.6 m above the ground
outside every footprint, with nothing higher around it (a tree has a crown, a building a roof). A run is a fence when it
is at least MIN_RUN_M long and at most MAX_WIDTH_M wide on average (a parked car is 5 m by 2 m). The surface holds each
fence cell's top (scene z) on building_surface's grid, NaN elsewhere; it occludes like building mass but is not wall
backing. Cached."""
import math, pathlib
import numpy as np
import frontage_evidence_v34 as E

PATH = E.ROOT / 'lidar-fence-surface-v36.npz'
LOW, HIGH = 2.0, 3.6            # metres above the ground: over a parked car's roof, under a tree's crown
MIN_RUN_M, MAX_WIDTH_M = 8., 1.5


def fence_surface(world, frame, surf, ground, rebuild=False):
    """(top, grid) with surf's grid: the top of fence-like low runs outside footprints."""
    top_b, (x0, y1, st) = surf
    if PATH.exists() and not rebuild:
        d = np.load(PATH)
        if tuple(float(v) for v in d['g']) == (x0, y1, st): return d['top'], (x0, y1, st)
    import shapely
    from shapely.ops import unary_union
    from pyproj import Transformer
    from scipy import ndimage as ndi
    import frontage_crops_v15 as fc
    pts = np.load(E.ROOT / 'data/points.npz')['points']; P = pts[(pts[:, 3] != 7) & (pts[:, 3] != 18)]; del pts
    px, py = Transformer.from_crs(3857, 32610, always_xy=True).transform(P[:, 0], P[:, 1])
    x = px - frame.cx; y = py - frame.cy; z = P[:, 2] - float(world['verticalOrigin'])
    ny, nx = top_b.shape
    ix = np.floor((x - x0) / st).astype(int); iy = np.floor((y1 - y) / st).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny); x, y, z, ix, iy = x[ok], y[ok], z[ok], ix[ok], iy[ok]
    gz, (gx0, gy1, gst) = ground
    gi = np.clip(((x - gx0) / gst).astype(int), 0, gz.shape[1] - 1); gj = np.clip(((gy1 - y) / gst).astype(int), 0, gz.shape[0] - 1)
    h = z - gz[gj, gi]
    fp = unary_union([g.buffer(.3) for g in fc.load_footprints(frame).values() if not g.is_empty])
    out = ~shapely.contains_xy(fp, x, y)
    flat = iy * nx + ix
    band = np.bincount(flat[out & (h >= LOW) & (h <= HIGH)], minlength=nx * ny).reshape(ny, nx)
    above = np.bincount(flat[h > HIGH], minlength=nx * ny).reshape(ny, nx)
    allc = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
    tops = np.full(nx * ny, -np.inf); sel = out & (h >= LOW) & (h <= HIGH); np.maximum.at(tops, flat[sel], z[sel]); tops = tops.reshape(ny, nx)
    # a fence cell: returns in the band, a real share of the cell's returns, and nothing much higher around it
    above3 = ndi.uniform_filter(above.astype(float), 3) * 9; all3 = ndi.uniform_filter(allc.astype(float), 3) * 9
    cand = (band >= 2) & (band >= .3 * np.maximum(allc, 1)) & (above3 < .2 * np.maximum(all3, 1))
    lab, n = ndi.label(cand, structure=np.ones((3, 3), bool))
    keep = np.zeros(n + 1, bool)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        cells = np.argwhere(lab[sl] == k) * st
        if len(cells) < 2: continue
        c = cells - cells.mean(0); ev = np.linalg.eigvalsh(c.T @ c / len(cells))
        length = 4. * math.sqrt(max(ev[1], 0.)) + st                 # the extent of the run along its main axis
        if length >= MIN_RUN_M and len(cells) * st * st / length <= MAX_WIDTH_M: keep[k] = True
    top = np.where(keep[lab] & (lab > 0), tops, np.nan).astype(np.float32)
    np.savez_compressed(PATH, top=top, g=np.array([x0, y1, st]))
    return top, (x0, y1, st)
