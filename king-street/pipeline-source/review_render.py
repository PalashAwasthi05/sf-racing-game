"""Same-camera perspective renders of a build for review, without the viewer: every buildings/facades/roofs triangle
of the GLB, back faces culled as in the viewer, painted far to near with the viewer's layer priority (facades over roofs
over massing) as a depth bias, flat-shaded with the material's base colour. Also the photograph from a panorama at the
same camera. Used by check_flagged_views.py; carries no location."""
import gc
import json
import mmap
import struct
import numpy as np
from PIL import Image, ImageDraw
import judge_facades_v27 as J
import frontage_crops_v15 as fc

WIDTH, HEIGHT, HFOV = 960, 720, 80.
PRIORITY = {'facades': 3, 'roofs': 2, 'signs': 1}


def layer_priority(path, n):
    """Per-triangle layer priority, in the triangle order of judge_facades_v27.load_build."""
    f = open(path, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    ln = struct.unpack('<I', mm[12:16])[0]; j = json.loads(mm[20:20 + ln]); out = []
    for mesh in j['meshes']:
        layer = mesh.get('name', '').split('__')[0]
        if layer not in J.LAYERS: continue
        for prim in mesh['primitives']:
            a = j['accessors'][prim['indices']] if 'indices' in prim else j['accessors'][prim['attributes']['POSITION']]
            out.append(np.full(a['count'] // 3, PRIORITY.get(layer, 0), np.float32))
    pr = np.concatenate(out); assert len(pr) == n, (len(pr), n)
    return pr


def mesh_labels(path, n):
    """Per-triangle mesh index (in load_build's triangle order) and the mesh names."""
    f = open(path, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    ln = struct.unpack('<I', mm[12:16])[0]; j = json.loads(mm[20:20 + ln]); out, names = [], []
    for mesh in j['meshes']:
        name = mesh.get('name', '')
        if name.split('__')[0] not in J.LAYERS: continue
        names.append(name)
        for prim in mesh['primitives']:
            a = j['accessors'][prim['indices']] if 'indices' in prim else j['accessors'][prim['attributes']['POSITION']]
            out.append(np.full(a['count'] // 3, len(names) - 1, np.int32))
    idx = np.concatenate(out); assert len(idx) == n
    return idx, names


def basis(eye, at):
    f = np.asarray(at, float) - np.asarray(eye, float); f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.]); right /= np.linalg.norm(right)
    return f, right, np.cross(f, right)


def load(path):
    b = J.load_build(path); b['pri'] = layer_priority(path, len(b['T'])); b['mesh_idx'], b['mesh_names'] = mesh_labels(path, len(b['T'])); return b


def render(build, eye, at, radius=140., mask_ids=None):
    """Pinhole view from eye towards at (scene metres, z up). With mask_ids, also a mask of the pixels where the walls
    of those buildings are what the camera sees (drawn in the same order, so anything in front covers them)."""
    eye = np.asarray(eye, float); f, right, down = basis(eye, at); fx = (WIDTH / 2) / np.tan(np.radians(HFOV / 2))
    near = np.flatnonzero(np.linalg.norm(build['T'].mean(1)[:, :2] - eye[:2], axis=1) < radius)
    T = build['T'][near].astype(np.float64); nrm = build['nrm'][near].astype(np.float64); col = build['col'][near]; pr = build['pri'][near]
    rel = T - eye; zc = rel @ f; xc = rel @ right; yc = rel @ down
    ok = (zc > .5).all(1) & (((eye - T.mean(1)) * nrm).sum(1) > 0)
    px = WIDTH / 2 + fx * xc / np.maximum(zc, 1e-3); py = HEIGHT / 2 + fx * yc / np.maximum(zc, 1e-3)
    ok &= (px.max(1) > -50) & (px.min(1) < WIDTH + 50) & (py.max(1) > -50) & (py.min(1) < HEIGHT + 50)
    idx = np.flatnonzero(ok); idx = idx[np.argsort(-(zc[idx].mean(1) - .35 * pr[idx]))]
    im = Image.new('RGB', (WIDTH, HEIGHT), (196, 214, 232)); dr = ImageDraw.Draw(im)
    horizon = HEIGHT / 2 - fx * f[2] / np.sqrt(max(1e-9, 1 - f[2] ** 2))
    dr.rectangle([0, int(np.clip(horizon, 0, HEIGHT)), WIDTH, HEIGHT], fill=(120, 124, 128))
    light = np.array([.45, -.35, .82]); light /= np.linalg.norm(light)
    mk = dm = None
    if mask_ids:
        names = build['mesh_names']; hit = np.array([any(m in nm for m in mask_ids) for nm in names])
        wall = hit[build['mesh_idx'][near]] & (np.abs(nrm[:, 2]) < .35)
        mk = Image.new('L', (WIDTH, HEIGHT), 0); dm = ImageDraw.Draw(mk)
    for i in idx:
        s = .55 + .45 * abs(float(nrm[i] @ light))
        poly = [(px[i, k], py[i, k]) for k in range(3)]
        dr.polygon(poly, fill=tuple(int(min(255, v * s)) for v in col[i]))
        if dm is not None: dm.polygon(poly, fill=255 if wall[i] else 0)
    return (im, mk) if mask_ids else im


def photo(image_id, eye, at, frame, balanced=False):
    """The panorama's view along the same direction (the panorama's own position; poses are coarse). balanced: white
    balance against the road under the camera, as the frontage readers do."""
    cam = fc.load_cameras(frame)[image_id]; f, _, _ = basis(eye, at); fx = (WIDTH / 2) / np.tan(np.radians(HFOV / 2))
    pano = np.asarray(fc.load_pano(cam['path']))
    if balanced:
        pf = pano.astype(np.float32); road = pf[int(pf.shape[0] * .74):int(pf.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
        gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
        pano = np.clip(pf * gain, 0, 255).astype(np.uint8)
    img, _ = fc.render_perspective(pano, cam['R'], f, HFOV, 2 * np.degrees(np.arctan((HEIGHT / 2) / fx)), width=WIDTH, max_height=HEIGHT)
    return Image.fromarray(img)


def wall_colour(img, mask, erode_px=18):
    """Median Lab of an image's pixels inside an eroded mask (glass, shadow and sky left out), or None."""
    from scipy import ndimage as ndi
    m = ndi.binary_erosion(np.asarray(mask) > 127, iterations=erode_px)
    a = np.asarray(img, dtype=np.float32)[m]
    if len(a) < 1500: return None
    lum = a.max(1) * .5 + a.min(1) * .5; keep = (lum > 60) & ~((a[:, 2] > a[:, 0] + 15) & (lum > 140))
    if keep.sum() < 800: return None
    return np.median(J.to_lab(a[keep][None])[0], 0)


def delta_e(p, q):
    """Colour difference with lightness at half weight (lighting differs between a photograph and a render)."""
    return float(np.sqrt((.5 * (p[0] - q[0])) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2))


def free(build):
    build.clear(); gc.collect()
