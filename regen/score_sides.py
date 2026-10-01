"""Score every side of a build against its photographs, in the wall's own frame.

Each view's rectified photograph is warped into wall metres through that view's LiDAR alignment (aligncorr.choose), the
build is drawn square-on in the same frame (judge_facades_v27.render_wall), and the judge's structure comparison scores
the pair. The view a reading was taken from scores the generator and the reading; the other view is a hold-out."""
import argparse, json, pathlib, sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import judge_facades_v27 as J, aligncorr
PPM = 12.

def photo_in_wall_frame(piece, v, corr, img_dir):
    im = np.asarray(Image.open(pathlib.Path(img_dir) / v['rect']).convert('RGB'), float); pad, ppm = v['rectPad'], v['ppm']
    L, H = float(piece['lengthM']), float(piece['heightM']); W, Hp = max(8, int(L * PPM)), max(8, int(H * PPM))
    u = (np.arange(W) + .5) / PPM; z = H - (np.arange(Hp) + .5) / PPM
    U, Z = np.meshgrid(u, z)
    if corr:
        U = corr['uc'] + (U - corr['uc']) / corr['s'] + corr['dx']; Z = corr['eye'] + (Z - corr['eye']) / corr['s'] + corr['dz']
    rows = pad + (H - Z) * ppm; cols = pad + U * ppm
    out = np.stack([map_coordinates(im[..., k], [rows, cols], order=1, mode='nearest') for k in range(3)], -1)
    return out.astype(np.uint8)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', required=True); ap.add_argument('--readings', default='final'); ap.add_argument('--out', required=True); a = ap.parse_args()
    root = pathlib.Path('regen/v1'); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
    al = json.load(open(root / 'alignment.json')) if (root / 'alignment.json').exists() else {}
    B = J.load_build(a.build); res = {}
    for f in sorted((root / a.readings).glob('*.json')):
        i = f.stem; p = byid.get(i)
        if not p or '.' in i: continue
        r = json.load(open(f))
        if r.get('thisBuilding') is False or not r.get('materials'): continue
        use = r.get('use', 0) if isinstance(r.get('use'), int) and 0 <= r.get('use', 0) < len(p['views']) else 0
        n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]])
        wall = {'midpoint': (n * p['planeV'] + t * sum(p['frameU']) / 2).tolist(), 'normal': p['normal'], 'lengthM': p['lengthM'], 'heightM': p['heightM']}
        rend, op, cov = J.render_wall(B, wall, p['groundZ'], p['heightM'], p['lengthM'])
        rec = {}
        for vi, v in enumerate(p['views']):
            corr = aligncorr.choose((al.get(i) or {}).get(str(vi)))
            ph = photo_in_wall_frame(p, v, corr, root / 'sides/img')
            cf = root / 'codes' / f'{i}_{vi}.npz'; cd = np.load(cf)['codes'] if cf.exists() else None
            j = J.judge(ph.astype(float), rend, op, cov, p['heightM'], cd)
            if j and not j.get('reliable'): j = None        # under 40 % of the wall can be judged in this photo: abstain
            rec['self' if vi == use else 'holdout'] = {'score': j['score'] if j else None, 'aligned': corr['kind'] if corr else None, **({k: j[k] for k in ('falseOpenings', 'missingDetail', 'openingCorr', 'reliable')} if j else {})}
        res[i] = rec
    json.dump(res, open(a.out, 'w'), indent=1)
    s = [x['self']['score'] for x in res.values() if x.get('self', {}).get('score') is not None]; h = [x['holdout']['score'] for x in res.values() if x.get('holdout', {}).get('score') is not None]
    print('sides', len(res), 'self mean', round(float(np.mean(s)), 3), 'holdout mean', round(float(np.mean(h)), 3) if h else None, 'n holdout', len(h))
