"""Local comparison preview of a whole candidate (no layer merging): chunk <candidate>/materials/candidate.glb into the
viewer's public folder and write public/compare/<label>.json, without touching the live release files or the
candidate's final/. View with http://127.0.0.1:3001/?manifest=/compare/<label>.json.
Usage: preview_full_candidate_v26.py <candidate> <label>"""
import sys, gzip, hashlib, json
from pathlib import Path
cand = Path(sys.argv[1]); label = sys.argv[2]; public = Path('work/king-street/public'); out = public / 'compare'; out.mkdir(exist_ok=True)
raw = (cand / 'materials/candidate.glb').read_bytes(); sha = hashlib.sha256(raw).hexdigest(); packed = gzip.compress(raw, compresslevel=6, mtime=0); chunks = []
for i, s in enumerate(range(0, len(packed), 4400000)):
    n = f'world-{sha[:12]}-{i:02}.bin'; (public / n).write_bytes(packed[s:s + 4400000]); chunks.append('/' + n)
(out / f'{label}.json').write_text(json.dumps(dict(compression='gzip', contentHash=sha[:12], chunks=chunks, compressedBytes=len(packed), uncompressedBytes=len(raw))))
print(json.dumps({'label': label, 'sha': sha[:12]}))
