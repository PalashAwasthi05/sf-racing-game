import json,hashlib
from pathlib import Path
pub=Path('work/king-street/public');m=json.load(open(pub/'model.json'));digest=hashlib.sha256(Path('outputs/king-street-pilot.glb').read_bytes()).hexdigest()[:12];paths=[]
for i,old in enumerate(m['chunks']):
 name=f'king-street-{digest}-{i:02d}.bin';(pub/old.lstrip('/')).rename(pub/name);paths.append('/'+name)
m['chunks']=paths;m['contentHash']=digest;json.dump(m,open(pub/'model.json','w'))
p=Path('work/build_world.py');s=p.read_text().replace('import gzip','import gzip,hashlib').replace("packed=(PUBLIC/'king-street.glb.gz').read_bytes();chunks=[]", "packed=(PUBLIC/'king-street.glb.gz').read_bytes();chunks=[];digest=hashlib.sha256(glb).hexdigest()[:12]").replace("name=f'king-street-{i//(4*1024*1024):02d}.bin'", "name=f'king-street-{digest}-{i//(4*1024*1024):02d}.bin'").replace("{'compression':'gzip','chunks':chunks", "{'compression':'gzip','contentHash':digest,'chunks':chunks");p.write_text(s)
p=Path('work/king-street/app/world-view.ts');s=p.read_text().replace("fetch('/model.json')", "fetch('/model.json',{cache:'no-cache'})");p.write_text(s)
p=Path('work/king-street/app/king-pilot.tsx');s=p.read_text().replace("fetch('/world.json')", "fetch('/world.json',{cache:'no-cache'})");p.write_text(s)
