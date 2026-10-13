"""More views of a building side for a reader: every panorama that sees the side, and new crops from any of them, so a
part hidden by a tree, a truck or the edge of the frame can be read from another angle, and the top storeys, roof and
gables can be read from a frame that reaches above the LiDAR wall.
Usage (run from work/; the area is Pacific Heights):
  views.py list <side>                                  cameras that see the side, best first: distance, incidence (1 =
                                                        square-on), visible fraction, trees near the sight line, which
                                                        side of square-on it stands (left/right) and the capture date
  views.py rect <side> <camera> <out.jpg> [--top 4]     the side rectified from that camera with metre ticks, the frame
                                                        raised --top metres above the LiDAR wall (same metres as the
                                                        side's own rect images: x from its left edge, z from its ground)
  views.py persp <side> <camera> <out.jpg> [--up 0] [--hfov 0] [--vfov 0]
                                                        an ordinary view from that camera aimed at the side; --up aims it
                                                        that many metres higher, --hfov/--vfov widen it (degrees)"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
WORK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(WORK)); os.environ.setdefault('SFW_AREA', str(WORK / 'areas/pacific-heights'))
import frontage_crops_v15 as fc
import auto_read_frontages_v24 as ar
from panorama_balance import road_gain

AREA = Path(os.environ['SFW_AREA']); FONT = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)


def side(sid):
    p = next((q for f in sorted((AREA / os.environ.get('SIDES_DIR', 'regen/sides')).glob('sides*.json')) for q in json.load(open(f))['pieces'] if q['id'] == sid and q.get('views')), None)
    if p is None: sys.exit(f'unknown side {sid}')
    n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]])
    wall = {'midpoint': (n * p['planeV'] + t * sum(p['frameU']) / 2).tolist(), 'normal': p['normal'], 'lengthM': p['lengthM'], 'heightM': p['heightM']}
    return p, wall


def setup():
    world, frame = fc.load_world(); cams = {k: c for k, c in fc.load_cameras(frame).items() if c['exists']}
    return cams, fc.Occluders(fc.load_footprints(frame)), fc.load_trees(frame)


def pano_of(cam):
    p = np.asarray(fc.load_pano(cam['path']), dtype=np.float32); return np.clip(p * road_gain(p), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['list', 'rect', 'persp']); ap.add_argument('side'); ap.add_argument('camera', nargs='?'); ap.add_argument('out', nargs='?')
    ap.add_argument('--top', type=float, default=4.); ap.add_argument('--up', type=float, default=0.); ap.add_argument('--hfov', type=float, default=0.); ap.add_argument('--vfov', type=float, default=0.)
    a = ap.parse_args(); p, wall = side(a.side); cams, occ, trees = setup(); z0 = float(p['groundZ'])
    if a.mode == 'list':
        used = [v['image'] for v in p['views']]
        cands = fc.wall_candidates(wall, cams, occ, trees, fc.MAX_DISTANCE_M * 1.6)
        n = np.asarray(wall['normal'], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]]); m = np.asarray(wall['midpoint'], float)
        rows = []
        for c in cands[:20]:
            v = np.asarray(c['cameraLocal']) - m; along = float(v @ t)
            rows.append({'camera': c['image'], 'distanceM': c['distanceM'], 'incidence': c['incidence'], 'visibleFraction': c['visibleFraction'],
                         'treesNearSightLine': c['treesNearSightLine'], 'stands': 'right' if along > 2 else 'left' if along < -2 else 'square-on',
                         'captureDate': cams[c['image']]['captureDate'], 'alreadyGiven': (f'v{used.index(c["image"])}' if c['image'] in used else '')})
        print(json.dumps({'side': a.side, 'lengthM': p['lengthM'], 'heightM': p['heightM'], 'cameras': rows}, indent=1)); return
    if not (a.camera and a.out): sys.exit('rect and persp need a camera id and an out path')
    cam = cams.get(a.camera) or cams.get('mapillary:' + a.camera) or next((c for k, c in cams.items() if k.endswith(a.camera)), None)
    if cam is None: sys.exit(f'unknown camera {a.camera}')
    pano = pano_of(cam); gz = fc.ground_at(cam['xy'])
    if a.mode == 'rect':
        H = float(p['heightM']) + max(0., a.top); w2 = {**wall, 'heightM': H}
        ar.PPM = min(45., 1400 / max(p['lengthM'], 1e-3), 1400 / max(H, 1e-3)); ppm = ar.PPM
        img, H, L = ar.rectify(w2, {**cam, 'groundZ': gz}, z0, pano)
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)); W, Hh = im.size; pad = 26
        can = Image.new('RGB', (W + 2 * pad, Hh + 2 * pad), (255, 255, 255)); can.paste(im, (pad, pad)); d = ImageDraw.Draw(can)
        for mtr in range(int(L) + 1):
            x = pad + mtr * ppm; ln = 10 if mtr % 5 == 0 else 4
            d.line([(x, pad + Hh), (x, pad + Hh + ln)], fill=(200, 0, 0)); d.line([(x, pad - ln), (x, pad)], fill=(200, 0, 0))
            if mtr % 5 == 0: d.text((x + 2, pad + Hh + 10), f'{mtr}m', fill=(200, 0, 0), font=FONT)
        for mtr in range(int(H) + 1):
            y = pad + Hh - mtr * ppm; ln = 10 if mtr % 5 == 0 else 4
            d.line([(pad - ln, y), (pad, y)], fill=(0, 0, 200)); d.line([(pad + W, y), (pad + W + ln, y)], fill=(0, 0, 200))
            if mtr % 5 == 0: d.text((1, y - 12), f'{mtr}', fill=(0, 0, 200), font=FONT)
        y = pad + Hh - float(p['heightM']) * ppm
        for x in range(pad, pad + W, 12): d.line([(x, y), (x + 6, y)], fill=(255, 140, 0), width=1)       # the LiDAR wall top
        can.save(a.out, quality=88)
        print(json.dumps({'side': a.side, 'camera': cam['id'], 'image': a.out, 'frameHeightM': round(H, 2), 'lidarWallTopM': p['heightM'], 'note': 'orange dashes: the LiDAR wall top'}))
    else:
        w2 = {**wall, 'heightM': float(p['heightM']) + 2 * max(0., a.up)}
        vp = fc.view_parameters(cam['xy'], w2, z0); f = np.asarray(vp['forward'], float)
        if a.up:
            yaw = math.atan2(f[0], f[1]); pitch = math.asin(max(-1, min(1, f[2] / np.linalg.norm(f)))) + math.atan2(a.up, max(2., float(np.linalg.norm(np.asarray(wall['midpoint']) - cam['xy']))))
            f = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
        hf = a.hfov or min(110, vp['hfovDeg'] * 1.3); vf = a.vfov or min(110, vp['vfovDeg'] * 1.3)
        img, vf = fc.render_perspective(pano, cam['R'], f, hf, vf, width=1200, max_height=1200)
        Image.fromarray(img).save(a.out, quality=88); print(json.dumps({'side': a.side, 'camera': cam['id'], 'image': a.out, 'hfovDeg': round(hf, 1), 'vfovDeg': round(vf, 1)}))


if __name__ == '__main__':
    main()
