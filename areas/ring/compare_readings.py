"""Score a reading of a side against a reference reading of the same side (a model choice check: does a cheaper reader
reproduce the reference?). Windows are matched by centre (within 0.4 m); the score lists missed and extra windows, the
position and size error of the matched ones, storeys, bays, bands, the special features and the colour difference.
Usage: compare_readings.py <reference.json> <candidate.json>   (run from work/)"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'regen'))
import specs, colours


def windows(r):
    r = specs.lay_bay_windows(r)
    out = [c for b in r.get('banks') or [] for c in specs.bank_cells(b)]
    out += [(o['x0'], o['z0'], o['x1'], o['z1']) for o in r.get('openings') or []]
    return np.array(out, float).reshape(-1, 4)


def dE(a, b):
    try: return float(np.linalg.norm(colours._lab([int(a[i:i + 2], 16) for i in (1, 3, 5)]) - colours._lab([int(b[i:i + 2], 16) for i in (1, 3, 5)])))
    except Exception: return None


def main():
    ref, cand = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    A, B = windows(ref), windows(cand)
    ca, cb = (A[:, :2] + A[:, 2:]) / 2, (B[:, :2] + B[:, 2:]) / 2
    used, pairs = set(), []
    for i in np.argsort([0] * len(ca)):
        d = np.hypot(*(cb - ca[i]).T) if len(cb) else np.array([])
        for j in np.argsort(d):
            if d[j] > .4: break
            if j not in used: used.add(j); pairs.append((i, j, d[j])); break
    size = [abs((A[i, 2] - A[i, 0]) - (B[j, 2] - B[j, 0])) + abs((A[i, 3] - A[i, 1]) - (B[j, 3] - B[j, 1])) for i, j, _ in pairs]
    feats = lambda r: {k: len(r.get(k) or []) for k in ('bays', 'bands', 'rustication', 'quoins', 'fireEscapes', 'balconies', 'ornaments', 'piers')}
    prof = lambda r: sorted((b.get('kind'), b.get('profile')) for b in r.get('bands') or [])
    out = {'windows': {'reference': len(A), 'candidate': len(B), 'matched': len(pairs), 'missed': len(A) - len(pairs), 'extra': len(B) - len(pairs),
                       'meanCentreErrorM': round(float(np.mean([d for _, _, d in pairs])), 3) if pairs else None,
                       'meanSizeErrorM': round(float(np.mean(size)), 3) if size else None},
           'storeys': {'reference': ref.get('storeys'), 'candidate': cand.get('storeys')},
           'wallTopM': [ref.get('wallTopM'), cand.get('wallTopM')],
           'features': {'reference': feats(ref), 'candidate': feats(cand)},
           'bands': {'reference': prof(ref), 'candidate': prof(cand)},
           'bays': {'reference': [[b['x0'], b['x1'], b['z0'], b['z1'], b.get('shape')] for b in ref.get('bays') or []],
                    'candidate': [[b['x0'], b['x1'], b['z0'], b['z1'], b.get('shape')] for b in cand.get('bays') or []]},
           'mainColour': {'reference': [(m.get('family'), m.get('hex')) for m in ref.get('materials') or []],
                          'candidate': [(m.get('family'), m.get('hex')) for m in cand.get('materials') or []]}}
    rm = {m.get('family'): m.get('hex') for m in ref.get('materials') or []}; cm = {m.get('family'): m.get('hex') for m in cand.get('materials') or []}
    out['colourDeltaE'] = {f: round(dE(rm[f], cm[f]), 1) for f in rm if f in cm and dE(rm[f], cm[f]) is not None}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
