"""The facade profiles for a ring build: an area's assembled profiles (regen/assemble.py) with the sides re-read for the
ring (areas/ring/readings, reading v2) in place of theirs. A re-read side keeps the reader's own colours (no photo
re-measurement): the readings judge daylight colour by eye, which the shaded or backlit photographs do not give.
Usage: profiles.py <area dir> <base profiles.json> <readings dir> <out.json> [--keep-low] [--nominal]   (run from work/)
Readings are moved onto the wall where the area's registered cameras put them (regen/remap_readings.py) unless --nominal."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'regen'))
import specs
area, base, rdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
pieces = {p['id']: p for p in json.load(open(area / 'regen/sides/sides.json'))['pieces']}
# readings measured on crops cut from provider camera poses are moved onto the wall where the registered cameras put
# what they measured (regen/remap_readings.py); --nominal keeps them as read
import os
os.environ.setdefault('SFW_AREA', str(area)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
WALLMAP = None
if '--nominal' not in sys.argv and (area / 'imagery/registered-cameras.json').exists():
    import frontage_crops_v15 as fc
    from register_cameras import apply_registration
    from remap_readings import WallMap, remap
    _world, _frame = fc.load_world(); _nom = fc.load_cameras(_frame); _reg = apply_registration(_nom, area / 'imagery/registered-cameras.json', bias_fallback=False)
    # a side whose two registered views disagree by over 1.5 m where the provider's agreed holds one bad registration:
    # its reading stays as read (regen/views_agree.py; the check cannot see a bias two cameras share, so it only vetoes)
    _va = json.load(open(Path(__file__).resolve().parent / 'views-agree.json')) if (Path(__file__).resolve().parent / 'views-agree.json').exists() else None
    def _split(pid):
        if _va is None or pid not in _va['A'] or pid not in _va['B']: return False
        import math as _m
        a, b = _va['A'][pid], _va['B'][pid]
        return min(a['peak'], b['peak']) > 6 and _m.hypot(b['dxM'], b['dyM']) > 1.5 and _m.hypot(a['dxM'], a['dyM']) <= 1.5
    def WALLMAP(r, p):
        if _split(p['id']): return None
        v = (p.get('views') or [None] * 2)[int(r.get('use') or 0)] if p.get('views') else None
        if v is None or v['image'] not in _reg or not _reg[v['image']].get('registered') or v.get('cameraGroundZ') is None: return None
        cn, cr = _nom[v['image']], _reg[v['image']]; gr = fc.ground_at(cr['xy'])
        if gr is None: return None
        return WallMap(p, cn['R'], [*cn['xy'], v['cameraGroundZ'] + fc.EYE_HEIGHT_M], cr['R'], [*cr['xy'], float(gr) + fc.EYE_HEIGHT_M + cr.get('heightOffsetM', 0.)])


def zfix(r, p):
    """A reading rectified squashed (its wall top under the LiDAR wall AND its storeys under 2.6 m apart, both)
    has every height stretched back by the LiDAR wall / its wall top, at most 1.6x (a wall top within 5 % of the LiDAR's is left). One signal alone is not enough: a
    roof structure makes the LiDAR wall taller, and some buildings have low storeys."""
    import numpy as np
    top = r.get('wallTopM'); st = [z for z in (r.get('storeys') or []) if isinstance(z, (int, float))]
    if not isinstance(top, (int, float)) or top <= 0 or len(st) < 4: return r, 1.
    pitch = float(np.median(np.diff(st[1:]))); k = p['heightM'] / top
    if not (top / p['heightM'] < .95 and pitch < 2.6): return r, 1.
    k = float(min(1.6, k)); Z = ('z0', 'z1', 'zM', 'wallTopM', 'openingHeightM', 'thicknessM', 'transomM', 'railingHeightM', 'heightM', 'courseM')
    def walk(o):
        if isinstance(o, dict):
            for key, v in list(o.items()):
                if key in Z and isinstance(v, (int, float)): o[key] = v * k
                elif key in ('storeys', 'levels', 'zCentersM') and isinstance(v, list): o[key] = [z * k if isinstance(z, (int, float)) else z for z in v]
                elif key == 'rows' and isinstance(v, list): o[key] = [[a * k, b * k] for a, b in v if isinstance(a, (int, float))]
                elif key == 'parapetTop' and isinstance(v, list): o[key] = [[x, z * k] for x, z in v]
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    r = json.loads(json.dumps(r)); walk(r); return r, k
prof = json.load(open(base)); byid = {p['id']: i for i, p in enumerate(prof['profiles'])}
done, dropped, moved = [], [], []
# a building whose only readings are low-confidence keeps them (a rough reading beats the generic facade); one with a
# better-read side drops its low ones for the style that side gives the rest of the building
KEEP = set(json.load(open(rdir.parent / 'keep-low.json'))['keep']) if (rdir.parent / 'keep-low.json').exists() else set()   # low readings kept by hand (areas/ring/keep-low.json says why)
# a low reading a reviewer has redone against the model and other cameras (areas/ring/reviews) beats the generic facade
KEEP |= {s['side'] for f in (rdir.parent / 'reviews').glob('*.json') for s in json.load(open(f)).get('sides', []) if s.get('readingChanged')}
good = {pieces[f.stem]['building'] for f in rdir.glob('*.json') if f.stem in pieces
        and (lambda r: r.get('thisBuilding') is not False and r.get('materials') and r.get('confidence') in ('medium', 'high'))(json.load(open(f)))}
REG = rdir.parent / 'readings-reg'      # re-readings made on crops cut from the registered cameras (regen/sides-reg)
for f in sorted(rdir.glob('*.json')):
    rr = REG / f.name; fresh = rr.exists()
    r = json.load(open(rr if fresh else f)); p = pieces.get(r.get('id', f.stem))
    if p is None: print('no side record for', f.stem); continue
    pid = 'regen-' + p['id']
    # a side whose photographs show another wall, or that the reader could barely see (an oblique side wall rectified at
    # the wrong scale, a wall behind a hedge), keeps no reading: the generator gives it the building's measured style
    # from its other sides instead of a guess
    if r.get('thisBuilding') is False or not r.get('materials') or (r.get('confidence') == 'low' and p['building'] in good and p['id'] not in KEEP and '--keep-low' not in sys.argv):
        if pid in byid: prof['profiles'][byid[pid]] = None
        dropped.append(p['id'])
        continue
    if WALLMAP is not None and not fresh and not r.get('registeredCrops'):   # a reading made on registered crops is in place
        wm = WALLMAP(r, p)
        if wm is not None:
            dx = float(wm(p['lengthM'] / 2, 3.)[0]) - p['lengthM'] / 2
            r = remap(r, wm); moved.append((p['id'], round(dx, 2)))
    r, k = zfix(r, p)
    if k != 1.: print(f'{p["id"]}: heights stretched x{k:.2f} (squashed rectification)')
    # a ring reading measures every window, so a bank is separate windows unless it says it is a ribbon (one band of
    # glass a storey): the generator's own guess joined three casements a third of a metre apart into one wide frame
    for o in (r.get('banks') or []) + (r.get('openings') or []): o.setdefault('ribbon', o.get('kind') == 'curtain')
    new = specs.to_profile(specs.clean(r, p), p); new['status'] = 'image-measured-ring-v2'
    if pid in byid: prof['profiles'][byid[pid]] = new
    else: prof['profiles'].append(new)
    done.append(p['id'])
prof['profiles'] = [p for p in prof['profiles'] if p is not None]
prof['method'] = prof.get('method', '') + ' + ring v2 readings (areas/ring/readings)'
json.dump(prof, open(out, 'w'))
print(json.dumps({'reread': len(done), 'dropped': dropped, 'profiles': len(prof['profiles']), 'remapped': len(moved), 'shiftOver1m': sum(abs(d) > 1 for _, d in moved)}))
