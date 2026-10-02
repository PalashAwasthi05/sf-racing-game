"""Local comparison preview of another area's build: the candidate's model chunked as preview_full_candidate_v26.py
does it, plus that build's own world data (terrain grid, radii, street views, label) beside the manifest, which names
it, so the viewer opens the area with its own ground and camera stations instead of the live area's. The live release
files are not touched. View with http://127.0.0.1:3001/?manifest=/compare/<label>.json
Usage: preview_area_v1.py <candidate dir> <label>   (run from the project root, as v26 is)"""
import json, shutil, subprocess, sys
from pathlib import Path
cand, label = Path(sys.argv[1]), sys.argv[2]
out = subprocess.run([sys.executable, 'work/preview_full_candidate_v26.py', str(cand), label], capture_output=True, text=True, check=True).stdout
compare = Path('work/king-street/public/compare'); world = compare / f'{label}-world.json'
shutil.copyfile(cand / 'public/world.json', world)
manifest = json.loads((compare / f'{label}.json').read_text()); manifest['world'] = f'/compare/{label}-world.json'
(compare / f'{label}.json').write_text(json.dumps(manifest))
print(out.strip().splitlines()[-1], '+ world', world.name)
