"""Viewer camera poses that reproduce the perspective photographs of the buildings round a park, so a render of the
model can stand beside each photograph at the same viewpoint (same camera position, direction and field of view as
regen/sides.py used to cut the photograph's _persp crop from its panorama).
Usage: poses.py <area dir> <park OSM way id> <world.json> <out.json> [--within 32] [--views 2] [--all-sides]
                [--registered <area>/imagery/registered-cameras.json]   (SFW_AREA set to the area; run from work/)
With --registered, a camera that regen/register_cameras.py placed against the LiDAR renders from its registered position
and height, its view turned by the heading and tilt found there, so it looks at what the photograph's crop shows."""
import argparse, glob, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
sys.path.insert(0, '.')
import frontage_crops_v15 as fc

ap = argparse.ArgumentParser(); ap.add_argument('area'); ap.add_argument('park'); ap.add_argument('world'); ap.add_argument('out')
ap.add_argument('--within', type=float, default=32.); ap.add_argument('--views', type=int, default=2); ap.add_argument('--all-sides', action='store_true', help='every photographed side, not only the one facing the park'); ap.add_argument('--registered'); a = ap.parse_args()
reg = {k: r for k, r in json.load(open(a.registered)).items() if r.get('accepted')} if a.registered else {}
# a camera the skyline could not place still carries the provider's shared bias: it moves by the registered cameras' median
bias = np.median([[r['dx'], r['dy']] for r in reg.values()], 0) if len(reg) >= 10 else None
area = Path(a.area); world, frame = fc.load_world(); cams = {k: c for k, c in fc.load_cameras(frame).items() if c['exists']}
root = ET.parse(fc.OSM_PATH).getroot(); nodes = {n.get('id'): frame.local(float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
park = next(Polygon([tuple(nodes[nd.get('ref')]) for nd in w.iter('nd')]) for w in root.iter('way') if w.get('id') == a.park)
centre = np.array(park.centroid.coords[0]); fps = fc.load_footprints(frame)
built = {b['id'] for b in json.load(open(a.world))['buildings']}
ring = sorted({b for b in built if b.split('_p')[0] in fps and fps[b.split('_p')[0]].distance(park) < a.within})
pieces = {}
for f in sorted(glob.glob(str(area / 'regen/sides/sides-*.json'))):
    for p in json.load(open(f))['pieces']:
        if p['views'] and (p['id'] not in pieces or len(p['views']) > len(pieces[p['id']]['views'])): pieces[p['id']] = p
out = []
for bid in ring:
    ps = [p for p in pieces.values() if p['building'] == bid]
    for p in ps:
        n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]]); p['_mid'] = n * p['planeV'] + t * sum(p['frameU']) / 2
        d = centre - p['_mid']; p['_face'] = float(n @ (d / np.linalg.norm(d)))
    # the side that faces the park (else the longest street side), its views in the order regen/sides.py ranked them
    ps.sort(key=lambda p: (-(p['_face'] > .3), -p['streetFacing'], -p['lengthM']))
    for p in (ps if a.all_sides else ps[:1]):
        for k, v in enumerate(p['views'][:a.views]):
            cam = cams[v['image']]; piece = {'midpoint': p['_mid'], 'normal': p['normal'], 'lengthM': p['lengthM'], 'heightM': p['heightM']}
            vp = fc.view_parameters(cam['xy'], piece, p['groundZ'])
            hf, vf = min(100, vp['hfovDeg'] * 1.25), min(100, vp['vfovDeg'] * 1.15)
            fx = 500 / math.tan(math.radians(hf) / 2); h = int(round(2 * fx * math.tan(math.radians(vf) / 2)))
            if h > 1000: h = 1000; vf = math.degrees(2 * math.atan(500 / fx))
            # the eye stands on the camera's own ground: on these hills the street under a camera is often metres above or below
            # the side's ground line, and a render from the side's ground sees the building from the wrong height
            eye = [float(cam['xy'][0]), float(cam['xy'][1]), float(fc.ground_at(cam['xy'])) + fc.EYE_HEIGHT_M]; f = np.asarray(vp['forward'], float)
            r = reg.get(v['image'])
            if r is None and bias is not None:
                xy = np.asarray(cam['xy'], float) + bias
                r = {'xy': xy.tolist(), 'cameraHeightM': fc.EYE_HEIGHT_M, 'dyawDeg': 0., 'tiltDeg': [0., 0.], 'biasOnly': True}
            if r:
                # the registered camera: its own position and height; the crop's direction turned by the heading and tilt the
                # skyline fit found (a direction the nominal rotation calls azimuth t is truly t + dyaw)
                xy = np.asarray(r['xy'], float); eye = [float(xy[0]), float(xy[1]), float(fc.ground_at(xy)) + r['cameraHeightM']]
                t = math.atan2(f[1], f[0]) + math.radians(r['dyawDeg']); e = math.asin(f[2] / np.linalg.norm(f))
                e -= math.radians(r['tiltDeg'][0]) * math.cos(t) + math.radians(r['tiltDeg'][1]) * math.sin(t)
                f = np.array([math.cos(e) * math.cos(t), math.cos(e) * math.sin(t), math.sin(e)])
            out.append({'registered': 'bias' if r and r.get('biasOnly') else bool(r), 'building': bid, 'side': p['id'], 'facesPark': p['_face'] > .3, 'view': k, 'image': v['image'], 'captureDate': v['captureDate'],
                        'persp': str(area / 'regen/sides/img' / v['persp']), 'from': [round(x, 3) for x in eye],
                        'at': [round(e + 20 * x, 3) for e, x in zip(eye, f)], 'vfovDeg': round(vf, 3), 'width': 1000, 'height': h})
json.dump({'park': a.park, 'ring': ring, 'poses': out}, open(a.out, 'w'), indent=1)
print(json.dumps({'ring': len(ring), 'withPhotos': len({o['building'] for o in out}), 'poses': len(out), 'noPhotos': [b for b in ring if b not in {o['building'] for o in out}]}))
