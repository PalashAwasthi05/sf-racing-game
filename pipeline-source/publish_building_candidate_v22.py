"""Local release adapter: replace building layers, preserve current streets byte-for-byte."""
import argparse,gzip,hashlib,json,os,shutil
from pathlib import Path
from replace_scene_layers_v21 import replace_scene_layers
from publish_street_candidate_v21 import atomic_json,STREET_COUNTS

def publish(candidate,base,public):
 final=candidate/'final';final.mkdir(exist_ok=True)
 report=replace_scene_layers(base,candidate/'materials/candidate.glb','buildings,facades,roofs',final/'candidate.glb',final/'report.json')
 if not report['preservationHashCheck']:raise ValueError('Street preservation failed')
 previous=json.loads((public/'world.json').read_text());world=json.loads((candidate/'public/world.json').read_text())
 history=candidate/'previous-preview';history.mkdir(exist_ok=True)
 for name in ['world.json','model.json','generation.json']:
  if not (history/name).exists():shutil.copy2(public/name,history/name)
 for key in STREET_COUNTS:world['counts'][key]=previous['counts'].get(key,0)
 for key in ['streetRevision','streetViews']:world[key]=previous.get(key)
 raw=(final/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=6,mtime=0);chunks=[]
 for i,start in enumerate(range(0,len(packed),4400000)):
  name=f'world-{sha[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4400000]);chunks.append('/'+name)
 world['previewRevision']='0.22 / SHARED BUILDING ENVELOPE'
 world['updateSummary']='All building layers regenerated with shared wall/roof heights, continuous supported slopes and surface-scoped facade materials. Street layers preserved. Facade source coverage remains incomplete.'
 generation={'pipelineVersion':'0.22','fullFacadeRegeneration':True,'legacyFacadeGeometryPreserved':False,'outputSha256':sha,'readyForFullFeedbackReview':False,'independentRealismGate':'NOT PASSED','streetLayersPreserved':True,'candidate':str(candidate.resolve()),'remaining':'Most facade inputs remain inferred; roof height disagreements remain.'}
 for name in ['roof-envelope-audit.json','height-audit.json']:
  if (candidate/name).exists():shutil.copy2(candidate/name,public/name)
 shutil.copy2(candidate/'public/facade-evidence.json',public/'facade-evidence.json')
 atomic_json(public/'world.json',world);atomic_json(public/'generation.json',generation)
 atomic_json(public/'model.json',dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)))
 result={'sha256':sha,'localPreview':'http://127.0.0.1:3001/?revision='+sha[:12],'preservedStreets':True};atomic_json(candidate/'release.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('work/street-v21-release/final/candidate.glb'));p.add_argument('--public',type=Path,default=Path('work/king-street/public'));a=p.parse_args();publish(a.candidate,a.base,a.public)
