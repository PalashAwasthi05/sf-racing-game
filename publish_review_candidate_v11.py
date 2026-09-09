"""Expose a local review candidate with an explicit incomplete status."""
from pathlib import Path
import argparse,json,gzip,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--stage',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];public=root/'work/king-street/public';out=root/'outputs';candidate=Path(a.candidate);raw=(candidate/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=6,mtime=0);chunks=[]
history=root/'work/history/before-v11-preview';history.mkdir(parents=True,exist_ok=True)
for name in ('model.json','world.json','generation.json'):
 if not (history/name).exists():shutil.copy2(public/name,history/name)
for i,start in enumerate(range(0,len(packed),4400000)):
 name=f'world-{sha[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4400000]);chunks.append('/'+name)
(public/'model.json').write_text(json.dumps(dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)),separators=(',',':')))
w=json.loads((history/'world.json').read_text());w['previewRevision']='WORK IN PROGRESS / SUPPORT REPAIR';w['updateSummary']='A support-boundary correction is under street-level review. Full facade reconstruction remains incomplete: no calibrated area-wide images or derived facade observations have entered the generator. The detailed scene still contains inherited layouts.';(public/'world.json').write_text(json.dumps(w,separators=(',',':')))
g=json.loads((history/'generation.json').read_text());g.update(outputSha256=sha,reviewStage=a.stage,readyForFullFeedbackReview=False,supportRepair=json.loads((candidate/'report.json').read_text()));(public/'generation.json').write_text(json.dumps(g,separators=(',',':')))
(out/'world-building-update-release.json').write_text(json.dumps(dict(revision='0.11',stage=a.stage,readyForFullFeedbackReview=False,outputSha256=sha,localPreview='http://127.0.0.1:3001/?revision='+sha[:12]+'&reviewStep=80',hostedSiteUpdated=False),indent=2))
if a.stage=='area-pass':shutil.copy2(candidate/'candidate.glb',out/'world-building-update.glb')
print(sha[:12])
