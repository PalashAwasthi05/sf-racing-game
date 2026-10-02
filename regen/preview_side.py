"""Preview what the generator will draw for one side from a reading, next to the side's aligned photographs, with the
judge's scores. For a corrector's edit-and-check loop: seconds per call, no world build.
Usage: preview_side.py <side id> <reading.json> <out.jpg>   (prints the scores as JSON)"""
import json, pathlib, sys, os
import numpy as np
from PIL import Image, ImageDraw
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import specs, colours, aligncorr, score_sides as S
import judge_facades_v27 as J
from pipeline import facade_grammar as G, material_recipes as MR
PPM = 12.

def rgb(material):
    base, _, hx = str(material).partition('#')
    if len(hx) == 6: return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))
    if str(material).startswith('#') and len(material) == 7: return tuple(int(material[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(MR.RECIPES.get(base, {}).get('color', [170, 170, 170]))

def draw(prof, L, H):
    W, Hp = max(8, int(L * PPM)), max(8, int(H * PPM)); Hs = float(prof.get('heightRefM') or H)
    im = Image.new('RGB', (W, Hp), rgb(prof['material'])); d = ImageDraw.Draw(im); op = Image.new('L', (W, Hp), 0); do = ImageDraw.Draw(op)
    X = lambda f: f * L * PPM; Z = lambda f: (H - f * Hs) * PPM
    for pn in prof.get('panels', []):
        x0, z0, x1, z1 = pn['rect']; d.rectangle([X(x0), Z(z1), X(x1), Z(z0)], fill=rgb(pn.get('material', '#999999')))
    for pr in prof.get('piers', []):
        x = pr['x'] * L; w = pr.get('widthM', .5); y0, y1 = pr.get('y', [0, 1]); d.rectangle([(x - w / 2) * PPM, Z(y1), (x + w / 2) * PPM, Z(y0)], fill=rgb(pr.get('material', '#999999')))
    for b in prof.get('bands', []):
        z = b['height'] * Hs; t = b.get('thicknessM', .2); d.rectangle([0, (H - z - t / 2) * PPM, W, (H - z + t / 2) * PPM], fill=rgb(b.get('material', '#999999')))
    for o in G.openings(prof, [], L, Hs, H):
        x0, z0, x1, z1 = o['rect']; st = o.get('style', {}); box = [X(x0), Z(z1), X(x1), Z(z0)]
        d.rectangle(box, fill=(28, 32, 36) if st.get('kind') != 'rollup' else (110, 112, 112), outline=rgb(st.get('frameMaterial', '#3a3d40')), width=2); do.rectangle(box, fill=255)
    return np.asarray(im), np.asarray(op) > 0

def main(i, reading, out):
    root = pathlib.Path(os.environ['REGEN_ROOT']).resolve() if os.environ.get('REGEN_ROOT') else HERE / 'v1'; p = {x['id']: x for x in json.load(open(root / 'sides/sides.json'))['pieces']}[i]
    al = json.load(open(root / 'alignment.json')).get(i)
    r = specs.clean(json.load(open(reading)), p); r = colours.measure(r, p, root / 'sides/img'); r, _ = aligncorr.correct(r, p, al)
    prof = specs.to_profile(r, p); L, H = float(p['lengthM']), float(p['heightM'])
    rend, op = draw(prof, L, H); cov = np.ones_like(op)
    tiles = [('preview of the model', rend)]; scores = {}
    for vi, v in enumerate(p['views']):
        ph = S.photo_in_wall_frame(p, v, aligncorr.choose((al or {}).get(str(vi))), root / 'sides/img')
        cf = root / 'codes' / f'{i}_{vi}.npz'; cd = np.load(cf)['codes'] if cf.exists() else None
        j = J.judge(ph.astype(float), rend, op, cov, H, cd)
        if j and not j.get('reliable'): j = None        # under 40 % of the wall can be judged in this photo: abstain
        scores[f'view{vi}'] = {k: j[k] for k in ('score', 'falseOpenings', 'missingDetail', 'openingCorr')} if j else None
        tiles.append((f'photo view {vi}  score {j["score"] if j else "n/a"}', ph))
    sc = min(1., 800 / max(rend.shape[1], rend.shape[0])); tw, th = int(rend.shape[1] * sc), int(rend.shape[0] * sc)
    sheet = Image.new('RGB', (len(tiles) * (tw + 8), th + 22), (20, 20, 22)); dr = ImageDraw.Draw(sheet)
    for k, (lab, a) in enumerate(tiles):
        sheet.paste(Image.fromarray(a.astype(np.uint8)).resize((tw, th)), (k * (tw + 8), 22)); dr.text((k * (tw + 8) + 4, 5), lab, fill=(235, 235, 235))
    sheet.save(out, quality=85)
    vals = [s['score'] for s in scores.values() if s]; scores['mean'] = round(float(np.mean(vals)), 3) if vals else None
    print(json.dumps(scores))

if __name__ == '__main__':
    main(*sys.argv[1:4])
