"""The viewer's render list for a poses file: [name, from, at, vfovDeg, height] per pose, for the browser-pane capture
driver (window.__run in the session notes). By default the before/after selection (before_after.py): per building the
side facing the park, else the first listed, at view 0; --all keeps every pose.
Usage: ba_list.py <poses.json> <out.json> [--all] [--views 0]   (run from work/; out usually king-street/public/compare/...)"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument('poses'); ap.add_argument('out'); ap.add_argument('--all', action='store_true'); ap.add_argument('--views', default='0'); a = ap.parse_args()
P = [p for p in json.load(open(a.poses))['poses'] if str(p['view']) in a.views.split(',')]
if not a.all:
    first = {}
    for p in P: first.setdefault(p['building'], p['side'])
    park = {p['building']: p['side'] for p in reversed(P) if p.get('facesPark')}
    P = [p for p in P if p['side'] == park.get(p['building'], first[p['building']])]
L = [[f"{p['side']}__v{p['view']}", p['from'], p['at'], p['vfovDeg'], p['height']] for p in P]
json.dump(L, open(a.out, 'w')); print(len(L), 'poses', sum(bool(p.get('registered')) for p in P), 'registered')
