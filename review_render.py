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


def basis(eye, at):
    f = np.asarray(at, float) - np.asarray(eye, float); f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.]); right /= np.linalg.norm(right)
    return f, right, np.cross(f, right)


def load(path):
    b = J.load_build(path); b['pri'] = layer_priority(path, len(b['T'])); return b


def render(build, eye, at, radius=140.):
    """Pinhole view from eye towards at (scene metres, z up)."""
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
    for i in idx:
        s = .55 + .45 * abs(float(nrm[i] @ light))
        dr.polygon([(px[i, k], py[i, k]) for k in range(3)], fill=tuple(int(min(255, v * s)) for v in col[i]))
    return im


def photo(image_id, eye, at, frame):
    """The panorama's view along the same direction (the panorama's own position; poses are coarse)."""
    cam = fc.load_cameras(frame)[image_id]; f, _, _ = basis(eye, at); fx = (WIDTH / 2) / np.tan(np.radians(HFOV / 2))
    img, _ = fc.render_perspective(np.asarray(fc.load_pano(cam['path'])), cam['R'], f, HFOV, 2 * np.degrees(np.arctan((HEIGHT / 2) / fx)), width=WIDTH, max_height=HEIGHT)
    return Image.fromarray(img)


def free(build):
    build.clear(); gc.collect()
