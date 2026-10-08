"""Flag ring readings whose vertical scale disagrees with the LiDAR wall: a rectified photograph can come out squashed
or stretched (a camera's pitch or the wall's plane misjudged), and every height read on it is then off by the same
factor. Compares each reading's wall top with the side's LiDAR wall height (the frame it was rectified to), and its
storey pitch with a city storey (2.6-4.6 m).
Usage: scale_check.py   (run from work/)"""
import glob, json, os
import numpy as np
pieces = {p['id']: p for p in json.load(open('areas/pacific-heights/regen/sides/sides.json'))['pieces']}
rows = []
for f in sorted(glob.glob('areas/ring/readings/*.json')):
    r = json.load(open(f)); sid = os.path.basename(f)[:-5]; p = pieces.get(sid)
    if not p or r.get('thisBuilding') is False: continue
    st = [z for z in (r.get('storeys') or []) if isinstance(z, (int, float))]
    pitch = float(np.median(np.diff(st[1:]))) if len(st) >= 4 else None
    top = r.get('wallTopM'); ratio = round(top / p['heightM'], 2) if isinstance(top, (int, float)) and top > 0 else None
    flag = []
    if ratio is not None and (ratio < .85 or ratio > 1.12): flag.append(f'top/LiDAR {ratio}')
    if pitch is not None and not (2.6 <= pitch <= 4.6): flag.append(f'storey pitch {pitch:.2f}')
    rows.append((sid, r.get('confidence'), round(p['heightM'], 1), top, ratio, None if pitch is None else round(pitch, 2), '; '.join(flag)))
for row in rows:
    if row[-1]: print(row)
print(len(rows), 'readings,', sum(bool(r[-1]) for r in rows), 'flagged')
