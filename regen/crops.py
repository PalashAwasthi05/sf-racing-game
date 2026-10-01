"""Stage 1 of the regeneration: square-on photo crops of every wall a street camera can see.

Reuses frontage_crops_v15 (camera choice, occlusion, perspective rendering); only the inputs differ: the current
build's wall list, every building, and up to two views per wall. Buildings are addressed only by the ids in the data.
"""
import argparse, json, pathlib, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import frontage_crops_v15 as fc

ap = argparse.ArgumentParser()
ap.add_argument('--evidence', required=True, help='king-street-evidence.json of the build whose walls to crop')
ap.add_argument('--out', required=True)
ap.add_argument('--budget', type=int, default=600)
ap.add_argument('--width', type=int, default=1100)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
world, frame = fc.load_world()
buildings = {b['id']: b for b in world['buildings']}
walls = json.load(open(a.evidence))['facadeWalls']
by_b = {}
for w in walls:
    by_b.setdefault(w['buildingId'], []).append(w)
occ = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
targets = [b for b in by_b if b in buildings]
per_wall, chosen = fc.select_crops(buildings, targets, by_b, cams, occ, trees, a.budget)
entries = []
pano_cache = {}
for w, cand, rank in chosen:
    wall = w['wall']; cam = cams[cand['image']]
    gz = buildings[w['building']]['center'][2]
    vp = fc.view_parameters(cam['xy'], wall, gz)
    if cand['image'] not in pano_cache:
        pano_cache.clear(); pano_cache[cand['image']] = fc.load_pano(cam['path'])
    img, vfov = fc.render_perspective(pano_cache[cand['image']], cam['R'], vp['forward'], vp['hfovDeg'], vp['vfovDeg'], width=a.width, max_height=int(a.width * 1.15))
    name = f"{w['building']}__{w['wallIndex']}__{cand['image']}.jpg"
    Image.fromarray(img).save(out / name, quality=85)
    entries.append({'building': w['building'], 'wallIndex': w['wallIndex'], 'status': wall['status'],
                    'lengthM': wall['lengthM'], 'heightM': wall['heightM'], 'midpoint': wall['midpoint'], 'normal': wall['normal'],
                    'streetFacing': wall.get('streetFacing'), 'image': cand['image'], 'captureDate': cam['captureDate'],
                    'distanceM': cand['distanceM'], 'incidence': cand['incidence'], 'visibleFraction': cand['visibleFraction'],
                    'rank': rank, 'yawDeg': vp['yawDeg'], 'pitchDeg': vp['pitchDeg'], 'hfovDeg': vp['hfovDeg'], 'vfovDeg': round(vfov, 2),
                    'wallSpanDeg': vp['wallSpanDeg'], 'size': [int(img.shape[1]), int(img.shape[0])], 'file': name})
seen = {(e['building'], e['wallIndex']) for e in entries}
summary = {'buildings': len(targets), 'walls': len(per_wall), 'wallsWithCamera': sum(bool(w['candidates']) for w in per_wall),
           'wallsCropped': len(seen), 'crops': len(entries), 'seconds': round(time.time() - t0, 1),
           'buildingsWithNoCrop': sorted(set(targets) - {e['building'] for e in entries})}
json.dump({'summary': summary, 'crops': entries}, open(out / 'manifest.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
