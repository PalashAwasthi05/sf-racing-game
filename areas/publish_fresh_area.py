"""Publish only an explicit comparison manifest and immutable asset files.
Existing release and area-menu manifests are preserved byte for byte.
"""
import argparse,gzip,hashlib,json,os,shutil,struct,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--area',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--label',required=True);p.add_argument('--memory-gb',type=float,default=16);a=p.parse_args()
root=Path(__file__).resolve().parents[2];os.chdir(root);public=root/'work/king-street/public';compare=public/'compare';compare.mkdir(exist_ok=True);label=a.label
if not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in label):raise ValueError('Unsafe label')
manifest=compare/f'{label}.json';before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in public.rglob('*.json') if f!=manifest}
source_path=a.candidate/'materials/candidate.glb';raw_len=source_path.stat().st_size
with source_path.open('rb') as source:
 header=source.read(20);model=json.loads(source.read(struct.unpack_from('<I',header,12)[0]));source.seek(0);digest=hashlib.file_digest(source,'sha256').hexdigest()
chunks=[]
with tempfile.TemporaryFile() as packed:
 with gzip.GzipFile(fileobj=packed,mode='wb',compresslevel=6,mtime=0) as compressed, source_path.open('rb') as source:shutil.copyfileobj(source,compressed,1024*1024)
 packed_len=packed.tell();packed.seek(0);i=0
 while data:=packed.read(4400000):
  name=f'world-{digest[:12]}-{i:02}.bin';(public/name).write_bytes(data);chunks.append('/'+name);i+=1
world=json.loads((a.candidate/'public/world.json').read_text());world['previewRevision']=label
(compare/f'{label}-world.json').write_text(json.dumps(world,separators=(',',':')))
manifest.write_text(json.dumps({'compression':'gzip','contentHash':digest[:12],'chunks':chunks,'compressedBytes':packed_len,'uncompressedBytes':raw_len,'world':f'/compare/{label}-world.json'}))
# Keep source frame files local to the input area for the source-based street furniture compilers.
shutil.copyfile(a.candidate/'public/world.json',a.area/'world.json')
config=json.loads((a.area/'config.json').read_text());shutil.copyfile(a.candidate/f"{config['outputPrefix']}-evidence.json",a.area/'evidence.json')
py=sys.executable;env={**os.environ,'SFW_AREA':str(a.area.resolve())}
for script in ['parking','overhead']:
 subprocess.run([py,'work/watch_mem.py',str(a.memory_gb),str(a.candidate/'publish.log'),py,f'work/areas/{script}.py',str(a.area),label,'--glb',str(a.candidate/'materials/candidate.glb')],env=env,check=True)
subprocess.run([py,'work/watch_mem.py',str(a.memory_gb),str(a.candidate/'publish.log'),py,'work/king-street/pipeline-source/package_startup.py',str(manifest)],check=True)
evidence=json.loads((a.area/'evidence.json').read_text());donors=sum(bool(f.get('donorId')) for f in evidence['facadeWalls']);landmarks=json.loads((a.candidate/'landmark-audit.json').read_text())
assert donors==0 and not landmarks, 'Fresh scene must not contain donors or landmark overrides'
triangles=sum(model['accessors'][q['indices']]['count']//3 for m in model.get('meshes',[]) for q in m['primitives'] if q.get('mode',4)==4 and 'indices' in q)
proof={'label':label,'manifest':f'/compare/{label}.json','sourceSha256':digest,'counts':world['counts'],'buildingRecords':len(world['buildings']),'streetViews':len(world['streetViews']),'meshCount':len(model.get('meshes',[])),'materialCount':len(model.get('materials',[])),'embeddedImages':len(model.get('images',[])),'triangles':triangles,'heightRangeM':[min(b['height'] for b in world['buildings']),max(b['height'] for b in world['buildings'])],'genericOnly':True,'donorProfiles':donors,'landmarkOverrides':len(landmarks),'imageVerifiedFacadeWalls':world['counts']['imageFittedWalls'],'materialOnlyProfiles':sum(bool(p.get('materialOnly')) for p in evidence.get('profiles',[])),'observationSourceRecords':len(evidence.get('observationSources',{})),'emittedPhotoBoundaryFragments':sum((w.get('openingEvidence') or {}).get('explicitBoundaries',0) for w in evidence['facadeWalls']),'emittedMappedEntranceFragments':sum((w.get('openingEvidence') or {}).get('mappedEntrances',0) for w in evidence['facadeWalls']),'facingsWithEmittedPhotoBoundaries':len({w.get('grammarId') for w in evidence['facadeWalls'] if (w.get('openingEvidence') or {}).get('explicitBoundaries',0)>0}),'appearanceStatus':'Source-derived skyline and streets; paired-image openings and mapped entrances where supported, inferred layouts elsewhere. Absolute facade placement is unverified.','otherAreaManifestsPreserved':all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in before.items() if not f.endswith(f'/{label}-world.json') and not f.endswith(f'/{label}-cars.json') and not f.endswith(f'/{label}-overhead.json'))}
(a.candidate/'scene-proof.json').write_text(json.dumps(proof,indent=2));assert proof['otherAreaManifestsPreserved'];print(json.dumps(proof))
