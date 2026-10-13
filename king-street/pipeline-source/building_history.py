"""How one source building looked in every published release, from each release's evidence: its parts and heights and,
per facing, which source drew the walls (a reading, inferred, blank) and in which material. Start any investigation of
a building the user has flagged here: a facing whose material or source flips while its evidence stays the same is
being decided by fallback rules. Usage: building_history.py <sourceId> [<sourceId> ...]"""
import collections, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent


def releases():
    out = []
    for r in ROOT.glob('*/release.json'):
        d = r.parent
        if (d / 'king-street-evidence.json').exists() or (d / 'public/facade-evidence.json').exists():
            out.append((r.stat().st_mtime, d, json.loads(r.read_text()).get('sha256', '')[:12]))
    return sorted(out)


def load(d):
    p = d / 'king-street-evidence.json'
    if p.exists():
        e = json.load(open(p)); return e['facadeWalls'], e['world']['buildings']
    e = json.load(open(d / 'public/facade-evidence.json')); return e['walls'], json.load(open(d / 'public/world.json'))['buildings']


def source(w):
    return 'reading ' + str(w.get('grammarId') or w.get('archetype')) if (w.get('edge') == 'grammar' or w.get('grammarId')) else str(w.get('status'))


def main(ids):
    for _, d, sha in releases():
        walls, blds = load(d); mine = lambda b: any(b == i or b.startswith(i + '_p') for i in ids)
        parts = [(b['id'], b.get('height'), round(b.get('area', 0))) for b in blds if mine(b['id'])]
        print(f'== {d.name} {sha} parts {parts}')
        facing = collections.defaultdict(collections.Counter)
        for w in walls:
            if not mine(w['buildingId']): continue
            ang = int(round(float(np.degrees(np.arctan2(w['normal'][1], w['normal'][0]))) / 45.) * 45) % 360
            facing[ang][(w['buildingId'][-3:], source(w), str(w.get('material')))] += w['lengthM']
        for ang in sorted(facing):
            top = ', '.join(f'{k[0]} {k[1]} {k[2]} {v:.0f} m' for k, v in facing[ang].most_common(4))
            print(f'   facing {ang:3d} deg: {top}')


if __name__ == '__main__':
    main(sys.argv[1:])
