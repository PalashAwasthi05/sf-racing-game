"""Write the job files of one stage: which readings, overlays and photographs each agent gets and where it writes."""
import argparse, json, pathlib
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['correct', 'verify', 'fix'])
ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True); ap.add_argument('--per', type=int, default=8)
ap.add_argument('--only', help='json list of ids to include'); a = ap.parse_args()
root = pathlib.Path('regen/v1').resolve(); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
only = set(json.load(open(a.only))) if a.only else None
(root / a.dst).mkdir(exist_ok=True)
items = []
for f in sorted((root / a.src).glob('*.json')):
    if '.' in f.stem or f.stem not in byid or (only is not None and f.stem not in only): continue
    r = json.load(open(f)); p = byid[f.stem]
    if r.get('thisBuilding') is False or not r.get('materials'): continue
    v = p['views'][r.get('use', 0)]
    items.append({'id': f.stem, 'lengthM': p['lengthM'], 'heightM': p['heightM'], 'reading': str(f),
                  'overlay': str(root / 'overlays' / a.src / f'{f.stem}.jpg'), 'rect': str(root / 'sides/img' / v['rect']),
                  'persp': str(root / 'sides/img' / v['persp']), 'out': str(root / a.dst / f.name)})
items.sort(key=lambda i: i['id'])
n = 0
for k in range(0, len(items), a.per):
    json.dump(items[k:k + a.per], open(root / 'jobs' / f'{a.stage}-{k // a.per:02d}.json', 'w'), indent=1); n += 1
print(len(items), 'items in', n, 'jobs')
