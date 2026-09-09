"""Full-area rebuild with source fingerprints, uniform rules and explicit release gates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from general_world.run import execute


def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def rebuild(input_path,output):
 p=Path(input_path).resolve();root=Path(__file__).resolve().parent;d=json.load(open(p));files=[p,p.parent/d['groundFile'],p.parent/d['surfacePointsFile'],root/'build_world.py',Path(__file__).resolve(),*sorted((root/'general_world').glob('*.py'))]
 def snapshot():return {str(f.relative_to(root)) if f.is_relative_to(root) else f.name:digest(f) for f in files}
 before=snapshot();report=execute(p,output);after=snapshot()
 if before!=after:raise RuntimeError('Inputs changed while rebuilding; candidate is invalid')
 manifest={'fullAreaRebuild':True,'manualProfileDependencies':0,'inputDrift':False,'inputFingerprint':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'sourceFiles':before,'outputHash':report['outputHash'],'appearanceGate':report['appearanceGate'],'releaseAllowed':report['publicationApprovedByQualityGate']}
 (Path(output)/'generation-manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(rebuild(a.input,a.output)))
