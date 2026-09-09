"""Apply the same support rule to a supplied detailed scene; identities only join data."""
import argparse,json,time
from pathlib import Path
import numpy as np,trimesh
from shapely.geometry import Polygon,shape
from shapely.ops import unary_union
from general_world.member_support import clip_long_members
from general_world.mesh_hygiene import remove_empty_faces
from general_world.uv_transfer import apply_source_material
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--footprints',required=True);p.add_argument('--output',required=True);p.add_argument('--test-owner');a=p.parse_args();start=time.time()
s=trimesh.load(a.scene,force='scene',process=False);footprints={r['id']:(shape(r['geometry']) if 'geometry' in r else unary_union([Polygon(ring) for ring in r['rings']])).buffer(0) for r in json.loads(Path(a.footprints).read_text())};records=[];new=[]
for name,m in list(s.geometry.items()):
 owner=m.metadata.get('featureId');layer=m.metadata.get('layer');h=remove_empty_faces(m)
 if a.test_owner and owner!=a.test_owner:continue
 if layer not in ('facades','buildings','roofs') or owner not in footprints:continue
 parts,report=clip_long_members(m,footprints[owner],missingSupportEvidence=m.metadata.get('evidence')!='approved-measured-boundary',measuredSupport=m.metadata.get('measuredSupport') is True)
 report.update(mesh=name,emptyFacesRemoved=h['emptyFacesRemoved']);records.append(report)
 changed=[part for part in parts if part['changed']]
 if not changed:continue
 mask=np.ones(len(m.faces),bool)
 for i,part in enumerate(changed):
  mask[part['originalFaceIndices']]=False;n=part['mesh']
  if len(n.faces):
   n=apply_source_material(n,m);n.metadata['supportRule']='bounded-unverified-member';new.append((name+'__supported_'+str(i),n))
 m.update_faces(mask)
for name,m in new:
 s.add_geometry(m,geom_name=name,node_name=name,transform=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s.export(out/'candidate.glb')
r={'allAreaMeshesProcessed':not bool(a.test_owner),'examinedMeshes':len(records),'sourceFootprintCount':len(footprints),'changedMeshes':sum(x['changed'] for x in records),'clippedMembers':sum(x['offending']-x['failedClips'] for x in records),'failedClips':sum(x['failedClips'] for x in records),'records':records,'elapsedSeconds':round(time.time()-start,2),'geometryMeaning':'support-boundary correction to inherited detail; not image-based reconstruction'};(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='records'}))
