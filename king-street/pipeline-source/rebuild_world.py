"""Full-region release wrapper: snapshot inputs, rebuild all geometry, reject drift."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent

def snapshot():
 files=[root/'build_world.py',Path(__file__)]+[p for p in (root/'pipeline').rglob('*') if p.suffix in ['.py','.json']]+[p for p in (root/'data').iterdir() if p.is_file()]+[p for p in (root/'material-assets').rglob('*') if p.is_file()]
 out={}
 for p in sorted(files):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
  out[str(p.relative_to(root))]=h.hexdigest()
 return out
before=snapshot();start=time.time();subprocess.run([sys.executable,str(root/'build_world.py')],check=True);after=snapshot()
if before!=after:raise RuntimeError('Generation inputs changed during rebuild; output cannot be released')
world=json.load(open(root/'king-street/public/world.json'));glb=root.parent/'outputs/king-street-pilot.glb';ids=sorted(b['id'] for b in world['buildings']);e=json.load(open(root.parent/'outputs/king-street-evidence.json'))
assert len(ids)==len(set(ids))==len(e['foundations'])
assert all(x['maximumUnfilledGapM']<=.001 for x in e['foundations'])
manifest={'pipelineVersion':world['pipelineVersion'],'fullRegionRebuild':True,'inputFingerprint':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'sourceFiles':before,'outputSha256':hashlib.sha256(glb.read_bytes()).hexdigest(),'regeneratedBuildingIds':ids,'foundationChecks':len(e['foundations']),'elapsedSecondsIncludingExport':round(time.time()-start,2),'inputDrift':False,'independentRealismGate':'NOT PASSED'}
(root.parent/'outputs/king-street-generation-manifest.json').write_text(json.dumps(manifest,indent=2));(root/'king-street/public/generation.json').write_text(json.dumps(manifest,separators=(',',':')))
print('FULL_REGION_MANIFEST',len(ids),manifest['inputFingerprint'])
