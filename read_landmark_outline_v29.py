"""Read a landmark's street wall from the panoramas: along the site outline every 4 m, the closest panorama with a
clear view (sight lines at least 1 m from any other footprint: poses are only good to a few metres) is rectified over
the wall's lower storey (1.5-6 m), white-balanced against the road under the camera, and its dominant wall colour
(glass, shadow, sky and foliage left out) is classified brick or stone. Consecutive samples of one family form
stretches. Writes work/pipeline/landmarks/outline-readings.json for the landmark builder; a debug sheet of every strip
goes to --sheet. Usage: read_landmark_outline_v29.py --candidate work/<dir> --landmark <id> [--sheet out.jpg]"""
import argparse, colorsys, json, math, pathlib, sys
import numpy as np
from panorama_balance import road_gain
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
from auto_read_frontages_v24 import rectify, kmeans

ROOT = pathlib.Path(__file__).resolve().parent
STEP_M, STRIP_M, LO_M, HI_M = 4., 4., 1.5, 6.
MAX_DIST_M, MAX_ANGLE_DEG, CLEAR_M = 40., 60., 1.


def family(rgb):
    h, l, s = colorsys.rgb_to_hls(*rgb); hd = h * 360
    if l >= .52 and s < .32 and (hd < 70 or hd > 330 or s < .08): return 'stone'
    if .16 <= l < .6 and s >= .16 and (hd < 30 or hd > 335): return 'brick'
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True); ap.add_argument('--landmark', required=True)
    ap.add_argument('--sheet', type=pathlib.Path); a = ap.parse_args()
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    z0 = next(b['center'][2] for b in world['buildings'] if b['id'] == a.landmark)
    fps = fc.load_footprints(frame); own = fps[a.landmark]
    site = Polygon(max(getattr(own, 'geoms', [own]), key=lambda p: p.area).exterior)
    from shapely.geometry.polygon import orient
    ring = orient(site, 1.0).exterior
    # other buildings only: any outline mostly inside the site (another mapping of the landmark, its parts) is not an occluder
    others = unary_union([g for k, g in fps.items() if not k.startswith(a.landmark) and g.intersection(site.buffer(1.)).area < .3 * max(g.area, 1e-6)])
    cams = {k: c for k, c in fc.load_cameras(frame).items() if not site.contains(Point(*c['xy']))}
    panos, samples, strips = {}, [], []
    for s in np.arange(STEP_M / 2, ring.length, STEP_M):
        p = np.array(ring.interpolate(s).coords[0]); q = np.array(ring.interpolate(min(ring.length, s + .5)).coords[0]) - np.array(ring.interpolate(max(0, s - .5)).coords[0])
        t = q / max(np.linalg.norm(q), 1e-9); n = np.array([t[1], -t[0]])                   # outward for a CCW ring
        best = None
        for k, c in cams.items():
            v = c['xy'] - p; d = float(np.linalg.norm(v))
            if d < 3 or d > MAX_DIST_M or math.degrees(math.acos(np.clip(v @ n / d, -1, 1))) > MAX_ANGLE_DEG: continue
            ends = [p + t * u for u in (-STRIP_M / 2, 0., STRIP_M / 2)]
            if any(others.distance(LineString([c['xy'], e + n * .3])) < CLEAR_M or site.buffer(-.2).intersects(LineString([c['xy'], e + n * .3])) for e in ends): continue
            score = d * (1 + .5 * (c.get('registrationGrade') == 'unregistered'))
            if best is None or score < best[0]: best = (score, k, d)
        if best is None: samples.append({'s': round(float(s), 1), 'family': None}); continue
        k = best[1]; c = cams[k]
        if k not in panos:
            if len(panos) > 3: panos.pop(next(iter(panos)))
            pano = np.asarray(fc.load_pano(c['path']), dtype=np.float32)
            gain = road_gain(pano)   # the road below the horizon, not the capture vehicle (panorama_balance)
            panos[k] = np.clip(pano * gain, 0, 255).astype(np.uint8)
        wall = {'midpoint': (p + n * .05).tolist(), 'normal': n.tolist(), 'lengthM': STRIP_M, 'heightM': HI_M}
        img, H, L = rectify(wall, c, z0, panos[k]); img = np.clip(img, 0, 1)
        band = img[int((H - HI_M) * img.shape[0] / H):int((H - LO_M) * img.shape[0] / H)].reshape(-1, 3)
        r, g, b = band.T; lum = band.max(1) * .5 + band.min(1) * .5
        keep = (lum > .25) & ~((b > r + .06) & (lum > .5)) & ~((g > r + .04) & (g > b + .02))
        rgb, fam, share = None, None, float(keep.mean())
        if keep.sum() > 200 and share > .15:
            cent, lab = kmeans(band[keep], 3); cnt = np.bincount(lab, minlength=len(cent)); rgb = cent[int(cnt.argmax())]; fam = family(rgb)
        samples.append({'s': round(float(s), 1), 'xy': [round(float(v), 2) for v in p], 'image': k, 'distanceM': round(best[2], 1), 'wallShare': round(share, 2),
                        'colour': '#%02x%02x%02x' % tuple(int(v * 255) for v in rgb) if rgb is not None else None, 'family': fam})
        if a.sheet is not None: strips.append((s, fam, (img * 255).astype(np.uint8)))
    # stretches: runs of one family, single unread or disagreeing samples bridged
    fams = [x['family'] for x in samples]; out = []
    i = 0
    while i < len(samples):
        f = fams[i]
        if f is None: i += 1; continue
        j = i
        while j + 1 < len(samples) and (fams[j + 1] == f or (fams[j + 1] != f and j + 2 < len(samples) and fams[j + 2] == f)): j += 1
        run = [x for x in samples[i:j + 1] if x['family'] == f]
        if len(run) >= 3:
            cols = np.array([[int(x['colour'][q:q + 2], 16) for q in (1, 3, 5)] for x in run]); med = np.median(cols, 0).astype(int)
            out.append({'s0': samples[i]['s'] - STEP_M / 2, 's1': samples[j]['s'] + STEP_M / 2, 'family': f, 'colour': '#%02x%02x%02x' % tuple(med), 'samples': len(run),
                        'images': sorted({x['image'] for x in run})})
        i = j + 1
    res_path = ROOT / 'pipeline/landmarks/outline-readings.json'
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res[a.landmark] = {'method': 'read_landmark_outline_v29.py: rectified lower storey from the closest clear panorama every 4 m; dominant wall colour; brick or stone',
                       'ringLengthM': round(ring.length, 1), 'orientation': 'counter-clockwise from the outline start', 'stretches': out, 'samples': samples}
    res_path.write_text(json.dumps(res, indent=1))
    print(f'{sum(1 for x in samples if x["family"])} of {len(samples)} samples read; stretches:')
    for o in out: print(f"  {o['s0']:6.1f}-{o['s1']:6.1f} m  {o['family']:5s} {o['colour']}  ({o['samples']} samples, {len(o['images'])} images)")
    if a.sheet is not None and strips:
        from PIL import Image, ImageDraw
        h = max(x[2].shape[0] for x in strips); w = max(x[2].shape[1] for x in strips); cols_n = 16
        sheet = Image.new('RGB', (cols_n * (w + 4), ((len(strips) + cols_n - 1) // cols_n) * (h + 18)), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
        for i, (s, f, im) in enumerate(strips):
            x, y = (i % cols_n) * (w + 4), (i // cols_n) * (h + 18); sheet.paste(Image.fromarray(im), (x, y + 16)); dr.text((x, y), f'{s:.0f} {f or "-"}', fill=(0, 0, 0))
        sheet.save(a.sheet, quality=88); print('sheet', a.sheet)


if __name__ == '__main__':
    main()
