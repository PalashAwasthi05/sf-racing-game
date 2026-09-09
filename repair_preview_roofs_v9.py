"""Area data adapter: fit every existing cap, preserve other preview content."""
import argparse,json,time
from pathlib import Path
import numpy as np
import trimesh
from pyproj import Transformer
from general_world.scene_roofs import repair_scene

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--scene',required=True);p.add_argument('--world',required=True);p.add_argument('--output',required=True);a=p.parse_args()
start=time.perf_counter();path=Path(a.input);d=json.loads(path.read_text());w=json.loads(Path(a.world).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
frame=d['coordinateFrame'];project=Transformer.from_crs('EPSG:4326',frame['projectedCrs'],always_xy=True);origin=np.array(project.transform(w['center']['lon'],w['center']['lat']))
cloud=np.load(path.parent/d['surfacePointsFile'])['points'];cloud[:,:2]+=np.array(frame['originProjectedM'])-origin;cloud[:,2]+=frame['verticalOriginM']-w['verticalOrigin']
s=trimesh.load(a.scene,force='scene',process=False);T=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float);s.apply_transform(T)
r=repair_scene(s,cloud);r['elapsedSeconds']=time.perf_counter()-start;r['coordinateOffsetM']=[*(np.array(frame['originProjectedM'])-origin),frame['verticalOriginM']-w['verticalOrigin']]
s.apply_transform(np.linalg.inv(T));s.export(out/'candidate.glb');(out/'report.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='records'}))
