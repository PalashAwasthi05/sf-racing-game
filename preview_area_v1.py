"""Local comparison preview of another area's build: the candidate's model chunked as preview_full_candidate_v26.py
does it, plus that build's own world data (terrain grid, radii, street views, label) beside the manifest, which names
it, so the viewer opens the area with its own ground and camera stations instead of the live area's. The live release
files are not touched. View with http://127.0.0.1:3001/?manifest=/compare/<label>.json, or pick the area from the
title menu (compare/areas.json: each area opens its newest previewed build).
Usage: preview_area_v1.py <candidate dir> <label>   (run from the project root, as v26 is)"""
import json, shutil, subprocess, sys
from pathlib import Path
cand, label = Path(sys.argv[1]), sys.argv[2]
out = subprocess.run([sys.executable, 'work/preview_full_candidate_v26.py', str(cand), label], capture_output=True, text=True, check=True).stdout
compare = Path('work/king-street/public/compare'); world = compare / f'{label}-world.json'
shutil.copyfile(cand / 'public/world.json', world)
manifest = json.loads((compare / f'{label}.json').read_text()); manifest['world'] = f'/compare/{label}-world.json'
(compare / f'{label}.json').write_text(json.dumps(manifest))
# The viewer's area menu (compare/areas.json): this area now opens its newest build; the published world stays first.
import datetime
index = compare / 'areas.json'
areas = json.loads(index.read_text())['areas'] if index.exists() else [{'id': 'king-street', 'label': '188 King Street', 'detail': 'SoMa · published release', 'manifest': None}]
area, place = cand.resolve().parent.name, json.loads((cand / 'public/world.json').read_text()).get('label')
entry = {'id': area, 'label': area.replace('-', ' ').title(), 'detail': ' · '.join(filter(None, [place, f'build {cand.name}, {datetime.date.today():%b} {datetime.date.today().day}'])),
         'manifest': f'/compare/{label}.json'}
index.write_text(json.dumps({'areas': [a for a in areas if a['id'] != area] + [entry]}, indent=1))
print(out.strip().splitlines()[-1], '+ world', world.name, '+ area menu', area)
