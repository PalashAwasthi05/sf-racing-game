"""Replay the complete candidate with permuted identities and order; compare geometry."""
import sys,json,hashlib,collections
from pathlib import Path
import numpy as np,trimesh
sys.path.insert(0,str(Path(__file__).parent));from general_world.run import execute
p=Path('work/normalized-v8/input.json');d=json.load(open(p));rng=np.random.default_rng(41)
for i,b in enumerate(d['buildings']):b['key']='opaque-'+str(i);b['name']='unused label';b['address']='unused metadata'
rng.shuffle(d['buildings']);rng.shuffle(d['surfaces']);q=p.with_name('relabeled.json');q.write_text(json.dumps(d));execute(q,'work/candidate-v8-relabel')
def signatures(file):
 s=trimesh.load(file,force='scene');result=[]
 for m in s.geometry.values():
  tr=np.round(m.triangles,4);order=np.lexsort((tr[:,:,2],tr[:,:,1],tr[:,:,0]),axis=1);tr=np.take_along_axis(tr,order[:,:,None],axis=1).reshape(-1,9);tr=tr[np.lexsort(tr[:,::-1].T[::-1])];result.append(hashlib.sha256(tr.tobytes()).hexdigest())
 return sorted(result)
a=signatures('work/candidate-v8/candidate.glb');b=signatures('work/candidate-v8-relabel/candidate.glb');assert a==b
report={'wholeAreaIdNameAndOrderPermutation':True,'geometriesCompared':len(a),'manualProfileDependencies':0,'appearanceValidated':False}
Path('outputs/general-world-area-validation.json').write_text(json.dumps(report,indent=2));print(report)
