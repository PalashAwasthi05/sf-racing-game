"""Exact active-output/semantic comparison for matching-input roof scenes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,trimesh
from areas.ring.splice import source


def hashes(scene,identity):
 result={}
 for name,m in scene.geometry.items():
  if name.split('__')[0] not in ('buildings','facades','roofs','foundations') or source(name)!=identity:continue
  result[name]=hashlib.sha256(np.asarray(m.vertices,dtype='<f8').tobytes()+np.asarray(m.faces,dtype='<i8').tobytes()).hexdigest()
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('before','after','profiles','selection','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);scenes=[trimesh.load(q/'king-street-pilot.glb',force='scene',process=False) for q in (a.before,a.after)];audits=[{r['buildingId']:r for r in json.load(open(q/'roof-envelope-audit.json'))} for q in (a.before,a.after)];profiles=json.load(open(a.profiles))['profiles'];rows=[]
 for identity in json.load(open(a.selection)):
  graphs=[x.get(identity,{}) for x in audits];shells=[g.get('roofShell',{}) for g in graphs];gable=[g.get('sourceGableOwnership',{}) for g in shells];proofs=[p for p in profiles if (p.get('match')or{}).get('partId')==identity];transition=gable[1].get('transition')
  if not transition:
   transition=next((v['transition'] for v in shells[1].get('sourceGablePriorValidation',[]) if isinstance(v,dict) and isinstance(v.get('transition'),dict)),None)
  ha,hb=[hashes(s,identity) for s in scenes];changed=ha!=hb;old_active=bool(gable[0].get('appliedBoundaryVertices'));new_active=bool(gable[1].get('appliedBoundaryVertices'));decision='unchanged'
  if old_active and not new_active:decision='unsafe-prior-rejected-native-shell-restored'
  elif new_active and changed:decision='source-boundary-and-unmeasured-transition-recomposed'
  row={'featureId':identity,'activeOutputChanged':changed,'decision':decision,'r30GableStatus':gable[0].get('status'),'r30AppliedBoundaryVertices':gable[0].get('appliedBoundaryVertices',0),'finalGableStatus':gable[1].get('status'),'finalAppliedBoundaryVertices':gable[1].get('appliedBoundaryVertices',0),'finalTransition':transition,'nativePlanesByteEqual':shells[0].get('structuralPlaneSurface',{}).get('planes')==shells[1].get('structuralPlaneSurface',{}).get('planes'),'sourceComponentsByteEqual':graphs[0].get('observation',{}).get('features',{}).get('accepted')==graphs[1].get('observation',{}).get('features',{}).get('accepted'),'beforeGeometryHashes':ha,'afterGeometryHashes':hb,'sourceProvenance':[{'profileId':p['id'],'sourceIds':p.get('sourceIds',[]),'reading':p.get('readingProvenance',{}).get('sourceReading'),'frameRegistration':p.get('sourceFrameRegistration')} for p in proofs]};rows.append(row)
 changed=[r['featureId'] for r in rows if r['activeOutputChanged']];(a.output/'all-changed-active-feature-ids.json').write_text(json.dumps(changed,indent=2)+'\n');(a.output/'audit.json').write_text(json.dumps({'status':'exact-matching-R30-versus-final-active-output-comparison','beforeRaw':str(a.before),'afterRaw':str(a.after),'beforeGlbSha256':hashlib.sha256((a.before/'king-street-pilot.glb').read_bytes()).hexdigest(),'afterGlbSha256':hashlib.sha256((a.after/'king-street-pilot.glb').read_bytes()).hexdigest(),'records':rows,'changedActiveFeatureCount':len(changed),'allChangedActiveFeatureIds':changed,'productionChanges':0,'qualification':'Matching frozenR30 source geometry data, no massing override. Final adds verified frame-scale metadata and keeps all cleanup semantic results. Rejected provisional roof parts restore native rather than silently retaining theold source warp.'},indent=2)+'\n');print(json.dumps({'changed':len(changed),'ids':changed,'decisions':[(r['featureId'],r['decision']) for r in rows if r['activeOutputChanged']]},indent=2))
if __name__=='__main__':main()
