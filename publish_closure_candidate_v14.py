"""Publish the v0.14 envelope-closure candidate to the local preview (no network publishing).

Same packaging as publish_material_candidate_v13.py: gzip the final GLB into 4.4 MB chunks named by
content hash, write model.json, refresh world.json from the regenerated pilot manifest while keeping the
preview's accumulated review fields, and keep the previous preview manifests under work/history.
"""
from pathlib import Path
import json,hashlib,gzip,shutil,argparse
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='work/detailed-v14');ap.add_argument('--summary',required=True);a=ap.parse_args()
cand=root/a.candidate;final=cand/'final';public=root/'work/king-street/public';out=root/'outputs'
report=json.loads((final/'report.json').read_text());raw=(final/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==report['outputSha256']
packed=gzip.compress(raw,compresslevel=6,mtime=0)
history=root/'work/history/before-v14-preview';history.mkdir(parents=True,exist_ok=True)
for name in ['world.json','generation.json','model.json']:
 if not (history/name).exists():shutil.copy2(public/name,history/name)
chunks=[]
for i,start in enumerate(range(0,len(packed),4400000)):
 name=f'world-{sha[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4400000]);chunks.append('/'+name)
for old in public.glob('world-*.bin'):
 if '/'+old.name not in chunks:old.unlink()
previous=json.loads((public/'world.json').read_text());regenerated=json.loads((cand/'public/world.json').read_text())
w=regenerated;w['previewRevision']='WORK IN PROGRESS / ENVELOPE CLOSURE';w['updateSummary']=a.summary
w['counts'].update({k:v for k,v in previous['counts'].items() if k not in w['counts']})
g=json.loads((public/'generation.json').read_text());g.update(outputSha256=sha,envelopeClosure={'revision':'0.14','pilotWorldJson':str(cand/'public/world.json'),'finalReport':report.get('counts')},readyForFullFeedbackReview=False)
(public/'world.json').write_text(json.dumps(w,separators=(',',':')));(public/'generation.json').write_text(json.dumps(g,separators=(',',':')))
(public/'model.json').write_text(json.dumps(dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)),separators=(',',':')))
shutil.copy2(final/'candidate.glb',out/'world-building-update.glb');shutil.copy2(final/'report.json',out/'material-regeneration-report.json')
(out/'world-building-update-release.json').write_text(json.dumps(dict(revision='0.14',readyForFullFeedbackReview=False,outputSha256=sha,localPreview='http://localhost:3001/?revision='+sha[:12],hostedSiteUpdated=False),indent=2))
print(json.dumps({'revision':sha[:12],'chunks':len(chunks),'compressedBytes':len(packed)}))
