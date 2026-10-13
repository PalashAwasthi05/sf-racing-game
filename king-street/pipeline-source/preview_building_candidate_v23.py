"""Local comparison preview: merge a building candidate into the v0.21 street world and write its chunks plus a
manifest under public/compare/<label>.json, without touching the live release files. View with
http://127.0.0.1:3001/?manifest=/compare/<label>.json. Usage: preview_building_candidate_v23.py <candidate> <label>"""
import sys,gzip,hashlib,json
from pathlib import Path
from replace_scene_layers_v21 import replace_scene_layers
cand=Path(sys.argv[1]);label=sys.argv[2];public=Path('work/king-street/public');out=public/'compare';out.mkdir(exist_ok=True)
final=cand/'final';final.mkdir(exist_ok=True)
rep=replace_scene_layers(Path('work/street-v21-release/final/candidate.glb'),cand/'materials/candidate.glb','buildings,facades,roofs',final/'candidate.glb',final/'report.json')
assert rep['preservationHashCheck']
raw=(final/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=6,mtime=0);chunks=[]
for i,s in enumerate(range(0,len(packed),4400000)):
    n=f'world-{sha[:12]}-{i:02}.bin';(public/n).write_bytes(packed[s:s+4400000]);chunks.append('/'+n)
(out/f'{label}.json').write_text(json.dumps(dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw))))
print(json.dumps({'label':label,'sha':sha[:12]}))
