"""External area adapter for the general measured open-cap orientation rule."""
import argparse,json,time
from pathlib import Path
import numpy as np,trimesh
from pyproj import Transformer
from shapely.geometry import shape,Point,Polygon
from shapely.ops import unary_union
from general_world.surface_orientation import find_reversed_caps
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--footprints',required=True);p.add_argument('--normalized-input',required=True);p.add_argument('--world',required=True);p.add_argument('--output',required=True);p.add_argument('--test-owner');a=p.parse_args();start=time.time()
path=Path(a.normalized_input);d=json.loads(path.read_text());w=json.loads(Path(a.world).read_text());frame=d['coordinateFrame'];project=Transformer.from_crs(4326,frame['projectedCrs'],always_xy=True);origin=np.array(project.transform(w['center']['lon'],w['center']['lat']));delta=np.r_[np.array(frame['originProjectedM'])-origin,frame['verticalOriginM']-w['verticalOrigin']];cloud=np.load(path.parent/d['surfacePointsFile'])['points'][:,:3]+delta;ground=np.load(path.parent/d['groundFile'])['points'][:,:3]+delta
s=trimesh.load(a.scene,force='scene',process=False);footprints={r['id']:shape(r['geometry']) if 'geometry' in r else unary_union([Polygon(ring) for ring in r['rings']]) for r in json.loads(Path(a.footprints).read_text())}
for feature in w['buildings']:
 url=feature.get('sourceUrl','')
 if url.startswith('https://www.openstreetmap.org/way/'):
  key='osm_'+url.rsplit('/',1)[-1]
  if key in footprints:footprints[feature['id']]=footprints[key]
records=[];new=[]
for name,m in list(s.geometry.items()):
 owner=m.metadata.get('featureId')
 if m.metadata.get('layer') not in ('buildings','roofs') or owner not in footprints or (a.test_owner and owner!=a.test_owner):continue
 if not np.any(m.face_normals[:,2]<-.98):continue
 fp=footprints[owner];lo=np.array(fp.bounds[:2]);hi=np.array(fp.bounds[2:]);pc=cloud[((cloud[:,:2]>=lo)&(cloud[:,:2]<=hi)).all(axis=1)];gp=ground[((ground[:,:2]>=lo-2)&(ground[:,:2]<=hi+2)).all(axis=1)]
 if not len(gp):continue
 others=[other for other in s.geometry.values() if other is not m and other.metadata.get('featureId')==owner and other.metadata.get('layer') in ('buildings','roofs','facades')]
 ids,report=find_reversed_caps(m,others,fp,pc,minimum_roof_z=float(np.median(gp[:,2]))+3.)
 report.update(mesh=name,owner=owner);records.append(report)
 if len(ids):
  part=m.submesh([ids],append=True,repair=False);part.invert();part.metadata={**m.metadata,'surfaceOrientation':'point-supported-open-cap-reversal'};new.append((name+'__orientation_repair',part));keep=np.ones(len(m.faces),bool);keep[ids]=False;m.update_faces(keep)
for name,m in new:s.add_geometry(m,geom_name=name,node_name=name,transform=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s.export(out/'candidate.glb');report={'allAreaMeshesExamined':not bool(a.test_owner),'meshesExamined':len(records),'meshesChanged':len(new),'facesReoriented':sum(len(m.faces) for _,m in new),'reorientedAreaM2':sum(m.area for _,m in new),'newSurfacesAdded':0,'records':records,'elapsedSeconds':time.time()-start};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}))
