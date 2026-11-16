"""Apply the standard texture stages to an already-generated section and retain
only its building/roof/facade layers. Does not edit the default local world."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import trimesh
from areas.ring.splice import source
def main():
 ap=argparse.ArgumentParser()
 for k in ('raw','output','only','area'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();work=Path(__file__).resolve().parents[1];env={**os.environ,'PYTHONPATH':str(work),'SFW_AREA':str(a.area.resolve())}
 def run(s,*args):subprocess.run([sys.executable,s,*map(str,args)],check=True,env=env)
 run('apply_textures_v10.py','--scene',a.raw/'king-street-pilot.glb','--output',a.raw/'textures')
 run('areas/slim_textures.py',a.raw/'textures/candidate.glb',a.raw/'textures/candidate-slim.glb')
 run('apply_materials_v13.py','--scene',a.raw/'textures/candidate-slim.glb','--observations','regen/v1/no-material-observations.json','--output',a.raw/'materials')
 s=trimesh.load(a.raw/'materials/candidate.glb',force='scene',process=False);out=trimesh.Scene();ids=set(json.loads(a.only.read_text()))
 for node in s.graph.nodes_geometry:
  if node.split('__')[0] not in ('buildings','facades','roofs') or source(node) not in ids:continue
  transform,geom=s.graph[node];out.add_geometry(s.geometry[geom],node_name=node,geom_name=geom,transform=transform)
 if not out.geometry:raise ValueError('empty building section')
 (a.output/'materials').mkdir(parents=True,exist_ok=False);(a.output/'public').mkdir();g=a.output/'materials/candidate.glb';out.export(g);shutil.copyfile(a.raw/'public/world.json',a.output/'public/world.json')
 (a.output/'section.json').write_text(json.dumps({'sha256':hashlib.sha256(g.read_bytes()).hexdigest(),'buildingIds':sorted(ids),'meshes':len(out.geometry),'fullWorldRebuilt':False},indent=1))
if __name__=='__main__':main()
