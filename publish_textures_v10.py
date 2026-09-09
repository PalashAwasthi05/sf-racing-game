from pathlib import Path
import json,gzip,hashlib,shutil,struct
root=Path(__file__).resolve().parents[1];work=root/'work';out=root/'outputs';public=work/'king-street/public'
candidate=work/'detailed-v10';report=json.loads((candidate/'report.json').read_text());raw=(candidate/'candidate.glb').read_bytes();digest=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=6,mtime=0)
assert gzip.decompress(packed)==raw
history=work/'history/before-v10-preview';history.mkdir(parents=True,exist_ok=True)
for name in ('world.json','model.json','generation.json'):
 if not (history/name).exists():shutil.copy2(public/name,history/name)
chunks=[]
for i,start in enumerate(range(0,len(packed),4400000)):
 name=f'world-{digest[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4400000]);chunks.append('/'+name)
w=json.loads((history/'world.json').read_text());w['previewRevision']='STREET STUDY / AUTOMATIC MATERIALS';w['updateSummary']='Automatic material matching and recoloring now updates 115 mesh surfaces using shared scanned materials. Surface patterns and finishes must match before color is considered. Material labels remain inherited, not verified against street photos; plaster repeat scale is provisional.'
w['counts'].update(automaticMaterialMeshes=report['updatedMeshes'],sharedMaterialAssets=len(report['uniqueAssignedAssets']),materialVariants=report['materialVariants'])
(public/'world.json').write_text(json.dumps(w,separators=(',',':')))
g=json.loads((history/'generation.json').read_text());g.update(pipelineVersion='detailed-baseline-0.7 + roof-repair-0.9 + automatic-materials-0.10',outputSha256=digest,textureAssignment=report)
(public/'generation.json').write_text(json.dumps(g,separators=(',',':')))
(public/'model.json').write_text(json.dumps(dict(compression='gzip',contentHash=digest[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)),separators=(',',':')))
shutil.copy2(candidate/'candidate.glb',out/'world-building-update.glb')
(out/'world-building-update-release.json').write_text(json.dumps(dict(revision='0.10',outputSha256=digest,geometryPreserved=report['geometryMeshesPreserved'],textureReport='texture-assignment-report.json',localPreview='http://127.0.0.1:3001/?revision='+digest[:12],hostedSiteUpdated=False),indent=2))
print(json.dumps(dict(revision=digest[:12],compressedBytes=len(packed),requests=len(report['newTextureRequests']))))
