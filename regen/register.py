"""Stage 1b: register every panorama the side views may use to the massing by skyline (register_panoramas.fit).
Writes one file for this regeneration; the shared review registration is left alone."""
import argparse, json, pathlib, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import frontage_crops_v15 as fc, judge_facades_v27 as J, register_panoramas as RP
ap = argparse.ArgumentParser(); ap.add_argument('--build', required=True); ap.add_argument('--sides', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', default='0/1'); a = ap.parse_args()
world, frame = fc.load_world(); cams = fc.load_cameras(frame); build = J.load_build(a.build)
ids = sorted({v['image'] for p in json.load(open(a.sides))['pieces'] if p['lengthM'] * p['heightM'] >= 30 for v in p['views']})
k, n = map(int, a.shard.split('/')); ids = ids[k::n]
if a.limit: ids = ids[:a.limit]
out = pathlib.Path(a.out); reg = json.load(open(out)) if out.exists() else {}
shared = out; out = out.with_name(out.stem + f'.shard{k}.json'); reg = {**reg, **(json.load(open(out)) if out.exists() else {})}
def one(pid):
    t0 = time.time(); cam = cams[pid]
    f = RP.fit(build, cam, np.asarray(fc.load_pano(cam['path'])), None)
    if not np.all(np.isfinite(f['R'])) or not np.all(np.isfinite(f['eye'])): raise ValueError('fit returned a non-finite pose')
    return {'xy': [float(f['eye'][0]), float(f['eye'][1])], 'z': float(f['eye'][2]), 'rotvec': Rotation.from_matrix(f['R']).as_rotvec().tolist(),
            'shiftM': [round(float(x), 2) for x in f['params'][:3]], 'rotationDeg': [round(float(x), 2) for x in f['params'][3:]],
            'groundZ': float(f['ground']), 'before': f['before'], 'after': f['after'], 'seconds': round(time.time() - t0, 1)}
for pid in ids:
    if pid in reg: continue
    try: reg[pid] = one(pid)
    except Exception as e: reg[pid] = {'failed': str(e)[:200]}
    out.write_text(json.dumps(reg))
    r = reg[pid]; print(pid, r.get('failed') or (r['shiftM'], r['before']['medianAbsDeg'], '->', r['after']['medianAbsDeg'], r['seconds']), flush=True)
