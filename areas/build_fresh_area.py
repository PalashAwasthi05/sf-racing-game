"""Build an independent area with reusable priors, source geometry and scanned materials.
Does not read another area's massing, profiles or model. Every heavy stage runs alone.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--area',required=True,type=Path);p.add_argument('--build',default='generic-v1');p.add_argument('--label',required=True);p.add_argument('--memory-gb',type=float,default=6.4);p.add_argument('--resume',action='store_true');p.add_argument('--profiles','--regen-profiles',dest='profiles',type=Path);p.add_argument('--material-observations',type=Path,default=Path('work/regen/v1/no-material-observations.json'));a=p.parse_args()
root=Path(__file__).resolve().parents[2];os.chdir(root);area=a.area.resolve();out=area/a.build;out.mkdir(parents=True,exist_ok=True)
config=json.loads((area/'config.json').read_text());prefix=config['outputPrefix'];py=str(root/'work/geoenv/bin/python');env={**os.environ,'PYTHONPATH':str(root/'work'),'PYTHONUNBUFFERED':'1','GEN_TRACE':'1','WATCH_LOG_S':'15','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
def filehash(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
tracked=[area/'config.json',*[area/'data'/n for n in ('osm.xml','points.npz','buildings.json','row.json','trees.json','signs.json')],root/'work/generate_detailed_v14.py',root/'work/pipeline/facades.py',root/'work/pipeline/materials.py',root/'work/pipeline/surface_grade.py',root/'work/pipeline/street_space.py']
if a.profiles:tracked.append(a.profiles.resolve())
tracked.extend(root/'work/pipeline'/n for n in ('facade_grammar.py','facade_lint.py','spatial_rules.py','vegetation.py','canopy.py'))
tracked.extend([root/'work/pipeline/project_point_cloud.py',root/'work/regen/bounded_dsm.py'])
tracked.append(a.material_observations.resolve())
merge_helper=root/'work/pipeline/merge_static_glbs.py'
if merge_helper.exists():tracked.append(merge_helper)
fingerprints={str(p):filehash(p) for p in tracked};record=out/'build-inputs.json'
reuse=a.resume and record.exists() and json.loads(record.read_text())==fingerprints
record.write_text(json.dumps(fingerprints,indent=2))
stages=[('geometry',out/f'{prefix}-pilot.glb',[py,'work/generate_detailed_v14.py','--output',str(out),'--area',str(area),'--generic-only','--bounded-scene']),('textures',out/'textures/candidate.glb',[py,'work/apply_textures_v10.py','--bounded','--scene',str(out/f'{prefix}-pilot.glb'),'--output',str(out/'textures')]),('slim',out/'textures/candidate-slim.glb',[py,'work/areas/slim_textures.py',str(out/'textures/candidate.glb'),str(out/'textures/candidate-slim.glb')]),('materials',out/'materials/candidate.glb',[py,'work/apply_materials_v13.py','--scene',str(out/'textures/candidate-slim.glb'),'--observations',str(a.material_observations.resolve()),'--output',str(out/'materials')])]
if a.profiles:
 stages[0][2].extend(['--regen-profiles',str(a.profiles.resolve())])
for name,product,cmd in stages:
 marker=out/f'{name}.complete'
 if reuse and marker.exists() and product.exists():continue
 print('STAGE',name,flush=True)
 subprocess.run([py,'work/watch_mem.py',str(a.memory_gb),str(out/'build.log'),*cmd],env=env,check=True)
 marker.write_text('complete\n')
 if name=='geometry' and merge_helper.exists():
  fingerprints[str(merge_helper)]=filehash(merge_helper);record.write_text(json.dumps(fingerprints,indent=2))
subprocess.run([py,'work/areas/publish_fresh_area.py','--area',str(area),'--candidate',str(out),'--label',a.label,'--memory-gb',str(a.memory_gb)],env=env,check=True)
