from pathlib import Path
import json,hashlib,gzip,shutil
root=Path(__file__).resolve().parents[1];candidate=root/'work/material-final-v13';public=root/'work/king-street/public';out=root/'outputs'
report=json.loads((candidate/'report.json').read_text());raw=(candidate/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==report['outputSha256'];packed=gzip.compress(raw,compresslevel=6,mtime=0)
history=root/'work/history/before-v13-preview';history.mkdir(parents=True,exist_ok=True)
for name in ['world.json','generation.json','model.json']:
 if not (history/name).exists():shutil.copy2(public/name,history/name)
chunks=[]
for i,start in enumerate(range(0,len(packed),4400000)):
 name=f'world-{sha[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4400000]);chunks.append('/'+name)
w=json.loads((public/'world.json').read_text());w['previewRevision']='WORK IN PROGRESS / MATERIALS';w['updateSummary']=f"Material regeneration updated {report['counts']['updated']} surfaces with shared metric color, relief and roughness maps. Color variants reuse material structure. Glossy surfaces now reflect nearby surroundings. Material assignments and facade layouts still inherit earlier input; they are not yet verified against street photographs. Earlier roof and support repairs remain in place."
w['counts'].update(materialUpdatedMeshes=report['counts']['updated'],newMaterialStructures=report['newStructureAssets'],materialVariantsV13=report['colorVariants'])
g=json.loads((public/'generation.json').read_text());g.update(outputSha256=sha,materialRegeneration=report,readyForFullFeedbackReview=False)
(public/'world.json').write_text(json.dumps(w,separators=(',',':')));(public/'generation.json').write_text(json.dumps(g,separators=(',',':')))
(public/'model.json').write_text(json.dumps(dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)),separators=(',',':')))
shutil.copy2(candidate/'candidate.glb',out/'world-building-update.glb');shutil.copy2(candidate/'report.json',out/'material-regeneration-report.json')
(out/'world-building-update-release.json').write_text(json.dumps(dict(revision='0.13',readyForFullFeedbackReview=False,outputSha256=sha,localPreview='http://127.0.0.1:3001/?revision='+sha[:12],hostedSiteUpdated=False),indent=2))
print(json.dumps({'revision':sha[:12],'updated':report['counts']['updated'],'compressedBytes':len(packed)}))
