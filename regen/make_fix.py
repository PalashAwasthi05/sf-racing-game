"""Worst-scoring sides -> fix jobs: each gets the model drawn beside its aligned photographs (wall frame), the reading's
overlay and clean photograph (reading frame), and the judge's measures, for a corrector to act on."""
import argparse, json, pathlib, sys, os
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import judge_facades_v27 as J, aligncorr, score_sides as S
ap = argparse.ArgumentParser(); ap.add_argument('--build', required=True); ap.add_argument('--scores', required=True); ap.add_argument('--share', type=float, default=.2)
ap.add_argument('--per', type=int, default=4); a = ap.parse_args()
root = pathlib.Path(os.environ.get('REGEN_ROOT', 'regen/v1')).resolve(); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
sc = json.load(open(a.scores)); al = json.load(open(root / 'alignment.json'))
def key(r): h = r.get('holdout', {}).get('score'); s = r.get('self', {}).get('score'); return h if h is not None else (s if s is not None else 9)
ranked = sorted(sc, key=lambda i: key(sc[i])); worst = ranked[:int(round(len(ranked) * a.share))]
B = J.load_build(a.build); (root / 'fixcmp').mkdir(exist_ok=True); (root / 'fixed').mkdir(exist_ok=True); items = []
for i in worst:
    p = byid[i]; n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]])
    wall = {'midpoint': (n * p['planeV'] + t * sum(p['frameU']) / 2).tolist(), 'normal': p['normal'], 'lengthM': p['lengthM'], 'heightM': p['heightM']}
    rend, _, _ = J.render_wall(B, wall, p['groundZ'], p['heightM'], p['lengthM'])
    tiles = [('model now', Image.fromarray(rend))]
    for vi, v in enumerate(p['views']):
        tiles.append((f'photo view {vi}', Image.fromarray(S.photo_in_wall_frame(p, v, aligncorr.choose((al.get(i) or {}).get(str(vi))), root / 'sides/img'))))
    w, h = tiles[0][1].size; sc_ = min(1., 700 / max(w, h)); tw, th = int(w * sc_), int(h * sc_)
    sheet = Image.new('RGB', (len(tiles) * (tw + 8), th + 22), (20, 20, 22)); d = ImageDraw.Draw(sheet)
    for k, (lab, im) in enumerate(tiles): sheet.paste(im.resize((tw, th)), (k * (tw + 8), 22)); d.text((k * (tw + 8) + 4, 5), lab, fill=(230, 230, 230))
    cmp_path = root / 'fixcmp' / f'{i}.jpg'; sheet.save(cmp_path, quality=85)
    r = json.load(open(root / 'final' / f'{i}.json')); u = r.get('use', 0) if isinstance(r.get('use'), int) else 0; v = p['views'][u]
    m = sc[i].get('holdout') or sc[i].get('self') or {}
    items.append({'id': i, 'lengthM': p['lengthM'], 'heightM': p['heightM'], 'reading': str(root / 'final' / f'{i}.json'),
                  'overlay': str(root / 'overlays' / 'final' / f'{i}.jpg'), 'rect': str(root / 'sides/img' / v['rect']), 'compare': str(cmp_path),
                  'check': {'score': m.get('score'), 'windowsOnSolidWall': m.get('falseOpenings'), 'photoDetailMissing': m.get('missingDetail')},
                  'out': str(root / 'fixed' / f'{i}.json')})
for k in range(0, len(items), a.per): json.dump(items[k:k + a.per], open(root / 'jobs' / f'fix-{k // a.per:02d}.json', 'w'), indent=1)
print(len(items), 'sides in', (len(items) + a.per - 1) // a.per, 'jobs; worst holdout', [round(key(sc[i]), 2) for i in worst[:5]])
