"""Write the draft-reading jobs of an area: every side piece with a photograph and at least 30 m2 of wall, grouped by
building, `--per` pieces a job. A set-back tier (baseM) carries the note that its bottom edge is a roof.
Skips pieces that already have a reading. Usage: make_drafts.py [--per 10] [--min-area 30]   (REGEN_ROOT selects the area)"""
import argparse, json, os, pathlib
ap = argparse.ArgumentParser(); ap.add_argument('--per', type=int, default=10); ap.add_argument('--min-area', type=float, default=30.)
ap.add_argument('--prefix', default='draft', help='job file prefix (a later pass keeps its own files)')
ap.add_argument('--exclude', default=None, help='glob of earlier job files whose pieces are left out')
a = ap.parse_args()
root = pathlib.Path(os.environ.get('REGEN_ROOT', 'regen/v1')).resolve()
S = json.load(open(root / 'sides/sides.json'))['pieces']
for f in ('sides-tiers.json',):
    if (root / 'sides' / f).exists():
        have = {p['id'] for p in S}; S += [p for p in json.load(open(root / 'sides' / f))['pieces'] if p['id'] not in have]
(root / 'readings').mkdir(exist_ok=True); (root / 'jobs').mkdir(exist_ok=True)
queued = {i['id'] for f in root.glob('jobs/' + a.exclude) for i in json.load(open(f))} if a.exclude else set()
P = [p for p in S if p['views'] and p['lengthM'] * p['heightM'] >= a.min_area and p['id'] not in queued and not (root / 'readings' / f"{p['id']}.json").exists()]
P.sort(key=lambda p: (p['building'], -p['lengthM'] * p['heightM']))
for f in root.glob(f'jobs/{a.prefix}-*.json'): f.unlink()
jobs, cur = [], []
for p in P:
    item = {'id': p['id'], 'lengthM': p['lengthM'], 'heightM': p['heightM'],
            'views': [{'view': k, 'rect': str(root / 'sides/img' / v['rect']), **({'persp': str(root / 'sides/img' / v['persp'])} if k == 0 else {})} for k, v in enumerate(p['views'])],
            'out': str(root / 'readings' / f"{p['id']}.json")}
    if p.get('baseM'):
        item['context'] = (f"This wall is a set-back upper tier: it stands on a lower roof {p['baseM']:.0f} m above the street, so its bottom edge is "
                           "that roof, not the street. It has no shopfronts, entries or garage doors; storey 0 is its lowest storey above that roof.")
    cur.append(item)
    if len(cur) == a.per: jobs.append(cur); cur = []
if cur: jobs.append(cur)
for i, j in enumerate(jobs): json.dump(j, open(root / 'jobs' / f'{a.prefix}-{i:03d}.json', 'w'), indent=1)
print(json.dumps({'pieces': len(P), 'jobs': len(jobs), 'tiers': sum(bool(p.get('baseM')) for p in P)}))
