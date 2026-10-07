"""The facade profiles for a ring build: an area's assembled profiles (regen/assemble.py) with the sides re-read for the
ring (areas/ring/readings, reading v2) in place of theirs. A re-read side keeps the reader's own colours (no photo
re-measurement): the readings judge daylight colour by eye, which the shaded or backlit photographs do not give.
Usage: profiles.py <area dir> <base profiles.json> <readings dir> <out.json>   (run from work/)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'regen'))
import specs
area, base, rdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
pieces = {p['id']: p for p in json.load(open(area / 'regen/sides/sides.json'))['pieces']}
prof = json.load(open(base)); byid = {p['id']: i for i, p in enumerate(prof['profiles'])}
done, dropped = [], []
for f in sorted(rdir.glob('*.json')):
    r = json.load(open(f)); p = pieces.get(r.get('id', f.stem))
    if p is None: print('no side record for', f.stem); continue
    pid = 'regen-' + p['id']
    if r.get('thisBuilding') is False or not r.get('materials'):
        if pid in byid: prof['profiles'][byid[pid]] = None; dropped.append(p['id'])
        continue
    new = specs.to_profile(specs.clean(r, p), p); new['status'] = 'image-measured-ring-v2'
    if pid in byid: prof['profiles'][byid[pid]] = new
    else: prof['profiles'].append(new)
    done.append(p['id'])
prof['profiles'] = [p for p in prof['profiles'] if p is not None]
prof['method'] = prof.get('method', '') + ' + ring v2 readings (areas/ring/readings)'
json.dump(prof, open(out, 'w'))
print(json.dumps({'reread': len(done), 'dropped': dropped, 'profiles': len(prof['profiles'])}))
