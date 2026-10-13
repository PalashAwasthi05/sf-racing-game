"""Package a validated street-layer replacement for the local viewer only."""
import argparse,gzip,hashlib,json,os,shutil
from pathlib import Path
from validate_street_candidate_v21 import validate
STREET_COUNTS={'roadSegments','crosswalks','trees','trashCans','signals','smallPoles','hydrants','lamps','signPanels','sidewalkAreaM2','mappedStairways','cyclewayFeatures','mappedBarrierSections','bollards','cycleSeparatorPosts','observedPaintPatches'}
def atomic_json(path,value):
 temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,separators=(',',':')));os.replace(temp,path)
def publish(candidate,public):
 validate(candidate)
 report=json.loads((candidate/'final/report.json').read_text())
 if not report['preservationHashCheck']:raise ValueError('Unchanged geometry failed preservation check')
 raw=(candidate/'final/candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if sha!=report['output']['sha256']:raise ValueError('Candidate hash differs from preservation report')
 previous=json.loads((public/'world.json').read_text());new=json.loads((candidate/'public/world.json').read_text())
 history=candidate/'previous-preview';history.mkdir(exist_ok=True)
 for name in ['world.json','model.json','generation.json']:
  if not (history/name).exists():shutil.copy2(public/name,history/name)
 packed=gzip.compress(raw,compresslevel=6,mtime=0);chunks=[]
 for i,start in enumerate(range(0,len(packed),4400000)):
  name=f'world-{sha[:12]}-{i:02}.bin';target=public/name
  if not target.exists():target.write_bytes(packed[start:start+4400000])
  chunks.append('/'+name)
 # Retain old chunks: clients already loading the previous manifest must finish.
 for key in STREET_COUNTS:previous['counts'][key]=new['counts'].get(key,0)
 previous['previewRevision']='0.21 / STREET SPACE AND FURNITURE'
 previous['updateSummary']='Street layers regenerated from shared surface ownership, curb geometry, mapped inventories and bounded object placement. Existing detailed buildings and roof work preserved.'
 previous['streetRevision']={'revision':'0.21','contentHash':sha[:12],'audit':'/street-space-audit.json','accuracy':'Inventory and inferred placement; not certified as-built.'}
 generation=json.loads((public/'generation.json').read_text());generation['outputSha256']=sha
 generation['streetSpace']={'revision':'0.21','preservedGeometry':True,'replacedLayers':report['layers'],'candidate':str(candidate.resolve())}
 generation['readyForFullFeedbackReview']=False
 shutil.copy2(candidate/'street-space-audit.json',public/'street-space-audit.json')
 atomic_json(public/'world.json',previous);atomic_json(public/'generation.json',generation)
 atomic_json(public/'model.json',dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)))
 release=dict(revision='0.21',outputSha256=sha,localPreview='http://127.0.0.1:3001/?revision='+sha[:12],hostedSiteUpdated=False,buildingGeometryPreserved=True)
 atomic_json(candidate/'release.json',release);print(json.dumps(release))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--public',type=Path,default=Path('work/king-street/public'));a=ap.parse_args();publish(a.candidate,a.public)
