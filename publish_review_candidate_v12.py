"""Publish a local surface-repair review with honest imagery/inference status."""
from pathlib import Path
import argparse,json,gzip,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--stage',choices=['section-test','area-pass'],required=True);p.add_argument('--imagery',type=Path,default=Path('work/imagery-v12'));a=p.parse_args();root=Path(__file__).resolve().parents[1];public=root/'work/king-street/public';out=root/'outputs';raw=(a.candidate/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=6,mtime=0);chunks=[]
for i,start in enumerate(range(0,len(packed),4400000)):
 name=f'world-{sha[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4400000]);chunks.append('/'+name)
(public/'model.json').write_text(json.dumps(dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)),separators=(',',':')))
report=json.loads((a.candidate/'report.json').read_text());imagery=json.loads((a.imagery/'report.json').read_text())
w=json.loads((public/'world.json').read_text());w['previewRevision']='WORK IN PROGRESS / SURFACE REPAIR';w['updateSummary']=f"Reversed roof surfaces have been corrected across the area. Roof fills require support at their own height, and new wall replacements require complete surface coverage. The active imagery selection contains {imagery['selectedImages']} Mapillary images, including {imagery['recentImagesWithin730Days']} recent views. Alignment to this world and automatic facade inference remain incomplete. The detailed scene still contains inherited facade layouts."
w['counts'].update(roofCoverageRepairs=report.get('patchesAdded',0),reorientedFaces=report.get('facesReoriented',0),acquiredImages=imagery['selectedImages'],registeredImages=imagery['worldRegistrationsValidated'])
(public/'world.json').write_text(json.dumps(w,separators=(',',':')))
g=json.loads((public/'generation.json').read_text());g.update(outputSha256=sha,reviewStage=a.stage,readyForFullFeedbackReview=False,surfaceRepair=report,imageryAcquisition=imagery);(public/'generation.json').write_text(json.dumps(g,separators=(',',':')))
if a.stage=='area-pass':
 shutil.copy2(a.candidate/'candidate.glb',out/'world-building-update.glb')
 (out/'world-building-update-release.json').write_text(json.dumps(dict(revision='0.12',stage=a.stage,readyForFullFeedbackReview=False,outputSha256=sha,localPreview='http://127.0.0.1:3001/?revision='+sha[:12],hostedSiteUpdated=False),indent=2))
 (out/'surface-repair-report.json').write_text(json.dumps(report,indent=2))
print(sha[:12])
