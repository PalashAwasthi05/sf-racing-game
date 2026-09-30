"""Register street panoramas to a build by their skylines.

A panorama's provider pose is coarse (metres in position, degrees in heading), so a crop of it and a render of the
model from the same nominal camera show different things: a colour check under a wall mask then measures the wrong
building. This refines each panorama's position (x, y, z) and rotation against the model:

- photograph: in every column of the equirectangular image, the lowest row of sky connected to the top of the frame
  (the skyline), as a direction in the panorama's own frame;
- model: from a candidate position, the highest elevation of any buildings/facades/roofs geometry in every azimuth
  (the upper envelope, which needs no visibility test);
- the pose minimises a robust (Cauchy) misfit between the two, in degrees of elevation. Trees, poles and buildings past
  the model's edge put the photograph's skyline in the wrong place in some azimuths; the robust loss leaves them out.

Both skylines move with the pose, so a far-off position does not score better by seeing less (the degenerate optimum
of an edge-to-edge score). Carries no location: it reads the panoramas named on the command line and the build given.
Writes review/panorama-registration.json (merged by panorama id) and an overlay per panorama for looking at.
"""
import argparse
import json
import pathlib
import time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import frontage_crops_v15 as fc
import judge_facades_v27 as J

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'review/panorama-registration.json'
BINS = 3600                  # model skyline azimuth bins (0.1 degree)
RADIUS_M = 400.              # geometry considered around the panorama
CAMERA_HEIGHT_M = 2.6        # a roof-mounted panorama camera above the road, the starting guess
SCALE_DEG = .6               # Cauchy scale of the elevation misfit


# ---------------------------------------------------------------------------------------------------- photograph

def photo_skyline(pano):
    """(u, v) in [0, 1) of the skyline in each usable column, from a downsampled equirectangular panorama."""
    h, w = pano.shape[:2]
    p = pano.astype(np.float32); r, g, b = p[..., 0], p[..., 1], p[..., 2]
    lum = p.mean(2); sat = p.max(2) - p.min(2)
    # sky: blue over red, or bright and colourless (cloud, haze); smooth
    grad = np.hypot(ndi.sobel(lum, 0), ndi.sobel(lum, 1))
    # local texture: facades have windows and joints, sky and cloud are smooth
    std = np.sqrt(np.maximum(ndi.uniform_filter(lum ** 2, 7) - ndi.uniform_filter(lum, 7) ** 2, 0))
    sky = (((b > r + 18) & (b > 110) & (grad < 60)) | ((lum > 185) & (sat < 55) & (b >= r - 4) & (grad < 90))
           | ((lum > 140) & (sat < 45) & (b >= r) & (std < 9)))
    pad = lambda m, op, k: op(np.pad(m, k, mode='edge'), iterations=k)[k:-k, k:-k]   # no border erosion at the zenith rows
    sky = pad(sky, ndi.binary_opening, 1)
    # overhead wires and cables: thin dark lines across the sky, bridged by a vertical closing
    sky = ndi.binary_closing(np.pad(sky, 6, mode='edge'), structure=np.ones((11, 1), bool))[6:-6, 6:-6]
    lab, _ = ndi.label(sky)
    top = np.unique(lab[: max(2, h // 40)]); top = top[top > 0]
    sky = pad(np.isin(lab, top), ndi.binary_closing, 2)
    us, vs = [], []
    for x in range(w):
        col = sky[:, x]
        if not col[0]: continue
        # the first run of at least 6 non-sky rows: thin seams at cloud edges are not the skyline
        solid = np.convolve((~col).astype(np.int32), np.ones(6, np.int32), 'valid') == 6
        rows = np.flatnonzero(solid)
        if not len(rows): continue
        y = rows[0]
        if y >= h // 2 - 2: continue          # sky down to the horizon: nothing modelled there
        us.append((x + .5) / w); vs.append(y / h)
    return np.array(us), np.array(vs)


def pano_dirs(u, v):
    """Unit directions in the panorama's camera frame (x right, y down, z forward) for equirectangular (u, v)."""
    lon = (u - .5) * 2 * np.pi; lat = (.5 - v) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


# ---------------------------------------------------------------------------------------------------- model

def edge_points(build, centre, spacing_rad=.0015):
    """Points along every triangle edge near `centre`, spaced finely enough for the envelope seen from near there."""
    T = build['T']; near = np.linalg.norm(T.mean(1)[:, :2] - centre[:2], axis=1) < RADIUS_M
    T = T[near].astype(np.float64)
    pts = [T.reshape(-1, 3)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        A, B = T[:, a], T[:, b]
        dist = np.maximum(np.linalg.norm((A + B) / 2 - centre, axis=1), 2.)
        n = np.clip(np.ceil(np.linalg.norm(B - A, axis=1) / (dist * spacing_rad)), 1, 400).astype(int)
        for k in np.unique(n):
            if k < 2: continue
            s = n == k; t = (np.arange(1, k) / k)[None, :, None]
            pts.append((A[s][:, None] + (B[s] - A[s])[:, None] * t).reshape(-1, 3))
    return np.concatenate(pts).astype(np.float32)


def column_tops(points, centre):
    """From any eye at the same horizontal distance, a point's elevation grows with its height, so only the highest
    point over each small patch of ground can be on the skyline: keep those (patches finer near the centre)."""
    p = points.astype(np.float64); hz = np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1])
    cell = np.maximum(.02, hz * .001); band = np.floor(np.log2(cell / .02)).astype(np.int64)
    size = .02 * 2. ** band
    ix = np.floor(p[:, 0] / size).astype(np.int64); iy = np.floor(p[:, 1] / size).astype(np.int64)
    key = (band << 50) + ((ix + (1 << 24)) << 25) + (iy + (1 << 24))
    order = np.lexsort((-p[:, 2], key)); k = key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    return points[order[first]]


def envelope_points(points, eye, shift_m=12.):
    """The points that can be on the skyline for any eye within shift_m of eye: those within reach of the envelope
    seen from eye (a near point can rise by up to atan(shift / distance) as the eye moves; a far one by little)."""
    d = points.astype(np.float64) - eye; hz = np.maximum(np.hypot(d[:, 0], d[:, 1]), .5)
    el = np.arctan2(d[:, 2], hz); sky = model_skyline(points, eye)
    i = ((np.arctan2(d[:, 0], d[:, 1]) / (2 * np.pi) + .5) * BINS).astype(int) % BINS
    reach = np.arctan(shift_m / np.maximum(hz - shift_m, .5)) * 2 + np.radians(1.)
    keep = el >= np.where(np.isfinite(sky[i]), sky[i], -9.) - reach
    return points[keep]


def model_skyline(points, eye):
    """Highest elevation (radians) of the geometry in each azimuth bin seen from eye; -inf where there is none."""
    d = points.astype(np.float64) - eye
    az = np.arctan2(d[:, 0], d[:, 1])                         # clockwise from north
    el = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    i = ((az / (2 * np.pi) + .5) * BINS).astype(int) % BINS
    sky = np.full(BINS, -np.inf); np.maximum.at(sky, i, el)
    # a thin gap between two samples is not an opening in the skyline
    filled = ndi.maximum_filter1d(np.where(np.isfinite(sky), sky, -9.), 3, mode='wrap')
    return np.where(filled > -9., filled, -np.inf)


def world_az_el(dirs):
    return np.arctan2(dirs[:, 0], dirs[:, 1]), np.arcsin(np.clip(dirs[:, 2], -1, 1))


# ---------------------------------------------------------------------------------------------------- fit

def pose(params, base_xy, base_z, R0):
    """params: dx, dy, dz (m), then a small world-frame rotation (degrees) applied before the provider rotation."""
    eye = np.array([base_xy[0] + params[0], base_xy[1] + params[1], base_z + params[2]])
    R = R0 @ Rotation.from_rotvec(np.radians(params[3:6])).as_matrix()
    return eye, R


def misfit(params, points, cam_dirs, base_xy, base_z, R0, detail=False):
    eye, R = pose(params, base_xy, base_z, R0)
    sky = model_skyline(points, eye)
    az, el = world_az_el(cam_dirs @ R)                         # world = R^T cam, as row vectors
    m = sky[((az / (2 * np.pi) + .5) * BINS).astype(int) % BINS]
    res = np.degrees(el - np.where(np.isfinite(m), m, 0.))
    cost = np.log1p((res / SCALE_DEG) ** 2)
    if detail: return res, sky, eye, R
    # priors: the rig's camera height is known to a few decimetres and the provider's tilt to a degree or two; without
    # them the fit lowers or tilts the camera to make up for a building the model has too tall or too short
    prior = (params[2] / .4) ** 2 + (params[3] / 1.5) ** 2 + (params[4] / 1.5) ** 2
    return float(cost.mean() + .05 * prior)


def ground_near(build, xy, radius_m=40.):
    """Road height near a panorama: the low end of the building geometry around it (walls stand on the ground)."""
    T = build['T']; c = T.mean(1)
    near = np.linalg.norm(c[:, :2] - xy, axis=1) < radius_m
    if near.sum() < 50: return None
    return float(np.percentile(T[near][:, :, 2], 2))


def fit(build, cam, pano, ground_z):
    small = np.asarray(Image.fromarray(pano).resize((1920, 960), Image.BILINEAR))
    u, v = photo_skyline(small)
    cam_dirs = pano_dirs(u, v)
    base_xy = np.asarray(cam['xy'], float); R0 = cam['R']
    if ground_z is None: ground_z = ground_near(build, base_xy)
    if ground_z is None: ground_z = 2.3
    base_z = ground_z + CAMERA_HEIGHT_M
    points = edge_points(build, np.array([base_xy[0], base_xy[1], base_z]))
    points = envelope_points(column_tops(points, base_xy), np.array([base_xy[0], base_xy[1], base_z]))
    args = (points, cam_dirs, base_xy, base_z, R0)
    start = misfit(np.zeros(6), *args)
    # coarse search over heading and position (the provider's worst errors), then a local refinement of all six
    best = (start, np.zeros(6))
    for yaw in np.arange(-6, 6.1, 1.5):
        for dx in np.arange(-8, 8.1, 2):
            for dy in np.arange(-8, 8.1, 2):
                p = np.array([dx, dy, 0, 0, 0, yaw]); c = misfit(p, *args)
                if c < best[0]: best = (c, p)
    res = minimize(misfit, best[1], args=args, method='Powell',
                   options={'xtol': 1e-3, 'ftol': 1e-6, 'maxfev': 4000})
    p = res.x
    r0, sky0, eye0, Rs = misfit(np.zeros(6), *args, detail=True)
    r1, sky1, eye1, R1 = misfit(p, *args, detail=True)
    stats = lambda r: {'medianAbsDeg': round(float(np.median(np.abs(r))), 2), 'inlierShare': round(float(np.mean(np.abs(r) < 1.)), 3)}
    return {'ground': ground_z, 'params': p, 'eye': eye1, 'R': R1, 'before': stats(r0), 'after': stats(r1), 'costBefore': start, 'costAfter': float(res.fun),
            'u': u, 'v': v, 'sky0': sky0, 'sky1': sky1, 'eye0': eye0, 'samples': len(u)}


def overlay(pano, cam_R, fitres, path):
    """The panorama with the model skyline before (red) and after (green) registration drawn over it."""
    im = Image.fromarray(pano).resize((2400, 1200), Image.BILINEAR); dr = ImageDraw.Draw(im); W, H = im.size
    az = (np.arange(BINS) + .5) / BINS * 2 * np.pi - np.pi
    for sky, eye, R, colour in ((fitres['sky0'], fitres['eye0'], cam_R, (255, 40, 40)), (fitres['sky1'], fitres['eye'], fitres['R'], (40, 255, 60))):
        ok = np.isfinite(sky); sky = np.where(ok, sky, 0.)
        d = np.stack([np.sin(az) * np.cos(sky), np.cos(az) * np.cos(sky), np.sin(sky)], -1)[ok]
        u, v = fc.world_dirs_to_pano_uv(d, R)
        pts = sorted(zip(u * W, v * H))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x1 - x0 < 6: dr.line([(x0, y0), (x1, y1)], fill=colour, width=2)
    for x, y in zip(fitres['u'] * W, fitres['v'] * H): dr.point((x, y), fill=(255, 255, 0))
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', type=pathlib.Path, required=True, help='GLB to register against')
    ap.add_argument('--world', type=pathlib.Path, default=ROOT / 'king-street/public/world.json')
    ap.add_argument('--ground-z', type=float, default=None, help='road height near the panoramas, scene metres (default: from the build)')
    ap.add_argument('--out-dir', type=pathlib.Path, default=ROOT / 'review/panorama-registration')
    ap.add_argument('panoramas', nargs='+')
    a = ap.parse_args()
    world = json.load(open(a.world)); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    cams = fc.load_cameras(frame); build = J.load_build(str(a.build)); a.out_dir.mkdir(parents=True, exist_ok=True)
    reg = json.load(open(OUT)) if OUT.exists() else {'schema': 1, 'panoramas': {}}
    reg['rule'] = ('Panorama poses refined against a build by skyline (register_panoramas.py). Review tooling reads them in '
                   'place of the provider pose; the generator never reads this file.')
    for pid in a.panoramas:
        t0 = time.time(); cam = cams[pid]; pano = np.asarray(fc.load_pano(cam['path']))
        f = fit(build, cam, pano, a.ground_z)
        overlay(pano, cam['R'], f, a.out_dir / f'{pid}.jpg')
        dx, dy, dz = (float(x) for x in f['params'][:3])
        reg['panoramas'][pid] = {
            'build': str(a.build), 'xy': [round(float(f['eye'][0]), 3), round(float(f['eye'][1]), 3)], 'z': round(float(f['eye'][2]), 3),
            'rotationAngleAxis': [float(x) for x in Rotation.from_matrix(f['R']).as_rotvec()],
            'shiftM': [round(dx, 2), round(dy, 2), round(dz, 2)], 'rotationDeg': [round(float(x), 2) for x in f['params'][3:]],
            'groundZ': round(f['ground'], 2), 'skylineBefore': f['before'], 'skylineAfter': f['after'], 'samples': f['samples'],
            'overlay': str(a.out_dir / f'{pid}.jpg')}
        print(f"{pid}: shift {reg['panoramas'][pid]['shiftM']} m, rotation {reg['panoramas'][pid]['rotationDeg']} deg, "
              f"skyline median {f['before']['medianAbsDeg']} -> {f['after']['medianAbsDeg']} deg, within 1 deg "
              f"{f['before']['inlierShare']} -> {f['after']['inlierShare']} ({f['samples']} columns, {time.time() - t0:.0f} s)")
    OUT.write_text(json.dumps(reg, indent=1))


if __name__ == '__main__':
    main()
