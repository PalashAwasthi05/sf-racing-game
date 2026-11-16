"""Review rounds: Palash flags what is wrong in model renders beside their street photographs, and each flag becomes a
fix to the generator's rules for every building of its kind (feedback_heuristic_gaps). One round = a fresh sample of
well-read facades across the area plus every facade not judged "Matches well" in the previous round.
Usage (run from work/):
  make_round.py poses <round> <build label> [--n 24] [--prev prev-round.json]  -> king-street/public/compare/round<N>-poses.json
  (render it with the viewer driver: prefix r<N>)
  make_round.py page <round> <build label>                                      -> areas/review/round<N>/ (page + images)
Previous-round results: ArtifactData list of collection round<N-1>, saved as JSON files (one per facade)."""
import argparse, glob, hashlib, json, random
from pathlib import Path
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('step'); ap.add_argument('round', type=int); ap.add_argument('build')
ap.add_argument('--n', type=int, default=24); ap.add_argument('--prev'); a = ap.parse_args()
D = Path('areas/review'); R = Path('areas/pacific-heights/regen'); O = D / f'round{a.round}'
P = {p['side']: p for p in json.load(open(D / 'poses-area-all.json'))['poses'] if p['view'] == 0}
if a.step == 'poses':
    again = []
    for f in sorted(glob.glob(str(Path(a.prev) / '*.json'))) if a.prev else []:
        d = json.load(open(f)); d = d.get('data', d)
        if d.get('verdict') != 'good' and d.get('side') in P: again.append(d['side'])
    seen = set()
    for f in glob.glob(str(D / 'round*' / 'items.json')): seen |= {i['side'] for i in json.load(open(f))}
    pool = []
    for s, p in P.items():
        f = R / 'readings-v2' / f'{s}.json'
        if s in seen or not f.exists(): continue
        r = json.load(open(f))
        if r.get('confidence') in ('medium', 'high') and r.get('thisBuilding') is not False: pool.append(s)
    rng = random.Random(a.round * 7919); rng.shuffle(pool)
    pick, bld = [], set()
    for s in pool:                                   # one facade per building, spread over the area
        b = P[s]['building'].split('_p')[0]
        if b in bld: continue
        bld.add(b); pick.append(s)
        if len(pick) >= a.n: break
    sides = list(dict.fromkeys(again + pick)); O.mkdir(parents=True, exist_ok=True)
    json.dump({'sides': sides, 'again': again}, open(O / 'sides.json', 'w'))
    L = [[f"{s}__v0", P[s]['from'], P[s]['at'], P[s]['vfovDeg'], P[s]['height']] for s in sides]
    json.dump(L, open(f'king-street/public/compare/round{a.round}-poses.json', 'w'))
    print(len(again), 'again +', len(pick), 'new =', len(sides))
else:
    S = json.load(open(O / 'sides.json')); (O / 'img').mkdir(parents=True, exist_ok=True); items = []
    for s in S['sides']:
        p = P[s]; h = hashlib.md5(f'{a.round}{s}'.encode()).hexdigest()[:10]; cap = Path(f'areas/captures/r{a.round}__{s}__v0.png')
        if not cap.exists(): continue
        for src, tag in ((p['persp'], 'p'), (cap, 'm')):
            Image.open(src).convert('RGB').resize((720, round(720 * p['height'] / p['width']))).save(O / 'img' / f'{h}-{tag}.jpg', quality=78)
        items.append({'id': h, 'side': s, 'build': a.build, 'again': s in S['again'], 'photo': f'img/{h}-p.jpg', 'render': f'img/{h}-m.jpg'})
    json.dump(items, open(O / 'items.json', 'w'))
    html = open(D / 'review.html').read().replace('__ROUND__', str(a.round)).replace('__ITEMS__', json.dumps(items))
    open(O / 'review.html', 'w').write(html)
    json.dump([{'path': f'img/{f.name}'} for f in sorted((O / 'img').glob('*.jpg'))], open(O / 'files.json', 'w'))
    print(len(items), 'facades on the page')
