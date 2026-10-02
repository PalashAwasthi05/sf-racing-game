"""What changed between two generator builds of one area: counts, facade walls (by building and reading), street
surfaces (area and where they differ) and placed street objects. Usage: compare_builds.py <old build> <new build>"""
import json, sys, collections
from pathlib import Path
from shapely.geometry import shape

old, new = Path(sys.argv[1]), Path(sys.argv[2])

def counts(b):
    ev = json.load(open(b / 'king-street-evidence.json'))['world']['counts']; return ev

co, cn = counts(old), counts(new)
print('counts that changed:')
for k in sorted(set(co) | set(cn)):
    if co.get(k) != cn.get(k) and k != 'generationSeconds': print(f'  {k}: {co.get(k)} -> {cn.get(k)}')

def walls(b):
    w = json.load(open(b / 'public/facade-evidence.json'))['walls']
    return collections.Counter((r['buildingId'], r.get('grammarId'), r.get('status'), round(r['lengthM'], 1), round(r['heightM'], 1)) for r in w)
wo, wn = walls(old), walls(new)
gone, added = wo - wn, wn - wo
print('facade walls: old', sum(wo.values()), 'new', sum(wn.values()), '| changed records: removed', sum(gone.values()), 'added', sum(added.values()))
by = collections.Counter(k[0] for k in list(gone) + list(added))
if by: print('  buildings with changed walls:', by.most_common(12))

go, gn = json.load(open(old / 'street-space-geometry.json')), json.load(open(new / 'street-space-geometry.json'))
for k in go:
    a, b = shape(go[k]), shape(gn[k]); d = a.symmetric_difference(b)
    if d.area > .5: print(f'  street {k}: {a.area:.0f} -> {b.area:.0f} m2, differs over {d.area:.1f} m2 at {[round(c, 1) for c in d.centroid.coords[0]]}')

def placed(b):
    return [(r['kind'], tuple(r.get('position') or []), r['status']) for r in json.load(open(b / 'street-space-audit.json'))['placements']]
po, pn = placed(old), placed(new)
moved = sum(1 for a, b in zip(po, pn) if a[1] and b[1] and max(abs(x - y) for x, y in zip(a[1], b[1])) > .002)
print('placements: old', len(po), 'new', len(pn), '| status changes', sum(a[0] != b[0] or a[2] != b[2] for a, b in zip(po, pn)), '| moved > 2 mm', moved)
