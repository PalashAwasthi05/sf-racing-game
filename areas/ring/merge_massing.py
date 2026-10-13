"""The massing agents' specs (areas/ring/massing/specs/*.json) into the generator's overrides
(areas/ring/massing-overrides.json), each checked first: its parts cover the building's outline with no gaps or overlaps,
every part passes massing_view's LiDAR check, and added buildings join the ring's build list (areas/ring/ring-ids.json).
A spec that fails is left out and said so.
Usage (run from work/): merge_massing.py [--dry]"""
import json, subprocess, sys
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
sys.path.insert(0, str(Path(__file__).resolve().parent))
import massing_view as mv

RING = Path('areas/ring'); dry = '--dry' in sys.argv
over = json.load(open(RING / 'massing-overrides.json')); ids = json.load(open(RING / 'ring-ids.json'))
D, _ = mv.dump(); report = []
for f in sorted((RING / 'massing/specs').glob('*.json')):
    j = json.load(open(f)); b = j.get('building', f.stem); src = b.split('_p')[0]
    if j.get('decision') != 'change': report.append({'building': b, 'merged': False, 'why': 'kept as it is'}); continue
    spec = dict(j.get('spec') or {}); problems = []
    if j.get('outline') and not spec.get('outline'): spec['outline'] = j['outline']        # a trimmed outline given beside the spec
    whole = Polygon(spec['outline']).buffer(0) if spec.get('outline') else unary_union([mv.poly(r) for r in mv.parts_of(src, D)])
    if spec.get('parts'):
        # a part drawn over another (a tower inside the main block): the earlier part gives way to the later ones
        parts = [dict(p) for p in spec['parts']]
        for i in range(len(parts)):
            g = Polygon(parts[i]['outline']).buffer(0)
            for q in parts[i + 1:]: g = g.difference(Polygon(q['outline']).buffer(0))
            if g.geom_type != 'Polygon': g = max(getattr(g, 'geoms', [g]), key=lambda q: q.area)
            parts[i]['outline'] = [[round(x, 3), round(y, 3)] for x, y in g.exterior.coords]
        spec['parts'] = parts
        json.dump({**j, 'spec': spec}, open(f, 'w'), indent=1)
        polys = [Polygon(p['outline']).buffer(0) for p in spec['parts']]; U = unary_union(polys)
        if abs(U.area - whole.area) > .03 * whole.area: problems.append(f'parts cover {U.area:.1f} m2 of a {whole.area:.1f} m2 outline')
        if sum(p.area for p in polys) > U.area * 1.02: problems.append('parts overlap')
        if any(p.area < 8 for p in polys): problems.append('a part under 8 m2')
        if len({p.get('id') for p in spec['parts'] if p.get('id')}) < len([p for p in spec['parts'] if p.get('id')]): problems.append('two parts share an id')
    out = subprocess.run([sys.executable, str(RING / 'massing_view.py'), 'check', src, str(f)], capture_output=True, text=True)
    try: rows = json.loads(out.stdout)['parts']
    except Exception: rows = []; problems.append('check failed to run: ' + out.stderr[-300:])
    if any(not r['pass'] for r in rows): problems.append('check fails: ' + ', '.join(f"{r['part']} below1 {r['below1']}" for r in rows if not r['pass']))
    if problems: report.append({'building': b, 'merged': False, 'why': '; '.join(problems)}); continue
    entry = {k: v for k, v in spec.items()}; entry['why'] = '; '.join(j.get('evidence', [])[:6]); entry['unmodelled'] = j.get('unmodelled', [])
    keep = {k: v for k, v in over.get(src, {}).items() if k in ('baseOffsetM', 'heightM', 'outline', 'baseWhy', 'roofFeatures') and k not in entry}   # earlier hand fixes and roof features stay unless replaced
    over[src] = {**keep, **entry}
    for a in j.get('adds') or []:
        if not a.get('id', '').startswith('add_') or '_p' in a['id']: report.append({'building': a.get('id'), 'merged': False, 'why': 'an added building needs an id add_... without _p'}); continue
        over[a['id']] = {'add': True, 'outline': a['outline'], 'wallTopM': a['wallTopM'], 'roof': a.get('roof', {'kind': 'flat'}), 'why': a.get('what', '')}
        if a['id'] not in ids: ids.append(a['id'])
    report.append({'building': b, 'merged': True, 'parts': len(spec.get('parts') or [1]), 'adds': len(j.get('adds') or []), 'check': [(r['part'], r['within0.6'], r['below1']) for r in rows]})
if not dry:
    json.dump(over, open(RING / 'massing-overrides.json', 'w'), indent=1); json.dump(ids, open(RING / 'ring-ids.json', 'w'), indent=1)
print(json.dumps(report, indent=1))
