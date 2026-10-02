"""Stage 3/5: snap every reading of a stage onto its photograph and draw the overlays the next stage looks at.
Invalid or empty readings are listed, not guessed at."""
import argparse, json, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import specs, snap
ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.environ.get('REGEN_ROOT', 'regen/v1')); ap.add_argument('--src', default='readings'); ap.add_argument('--dst', default='snapped'); a = ap.parse_args()
root = pathlib.Path(a.root); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
(root / a.dst).mkdir(exist_ok=True); (root / 'overlays' / a.dst).mkdir(parents=True, exist_ok=True)
stats = {'snapped': 0, 'notThisBuilding': [], 'invalid': [], 'improved': 0, 'elements': 0}
for f in sorted((root / a.src).glob('*.json')):
    if '.' in f.stem: continue            # test copies (x.haiku.json)
    i = f.stem
    if i not in byid: continue
    try: r = json.load(open(f))
    except Exception as e: stats['invalid'].append([i, str(e)[:80]]); continue
    if r.get('thisBuilding') is False or not r.get('materials'): stats['notThisBuilding'].append(i); json.dump(r, open(root / a.dst / f.name, 'w')); continue
    try:
        r = specs.clean(r, byid[i])
        r2, rep = snap.snap_reading(r, byid[i], root / 'sides/img')
        specs.overlay(r2, byid[i], root / 'sides/img', root / 'overlays' / a.dst / f'{i}.jpg')
        specs.to_profile(r2, byid[i])
    except Exception as e: stats['invalid'].append([i, repr(e)[:120]]); continue
    json.dump(r2, open(root / a.dst / f.name, 'w'), indent=1)
    stats['snapped'] += 1; stats['elements'] += len(rep); stats['improved'] += sum(1 for x in rep if x[3])
json.dump(stats, open(root / f'{a.dst}-stats.json', 'w'), indent=1); print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in stats.items()}))
