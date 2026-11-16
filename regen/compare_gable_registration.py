"""Roof-only source-wide replay of staged registration/cleaning corrections."""
import argparse,importlib.util,json,types,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon
import pipeline
from pipeline import roof_shell,roof_source_parts,roof_assemblies,roof_front_ownership,spatial_rules
from pipeline.facade_lint import clean
from regen.staged_gable_ownership import roof_gable_ownership as after_helper
from regen.rebuild_roof_support_review import parameters,contexts


def load_old(path=None):
 p=path or Path('regen/staged_gable_ownership/roof_gable_ownership-before-snap.py');s=importlib.util.spec_from_file_location('old_source_gable_helper',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def mesh_delta(a,b):
 distance,index=cKDTree(b.vertices[:,:2]).query(a.vertices[:,:2]);common=distance<1e-6;delta=abs(a.vertices[common,2]-b.vertices[index[common],2]);keys=lambda mesh:{tuple(map(tuple,tri[np.lexsort((tri[:,1],tri[:,0])),:2].round(6))):tri[np.lexsort((tri[:,1],tri[:,0])),2] for tri in mesh.triangles};ka,kb=keys(a),keys(b);shared=ka.keys()&kb.keys();changed=sum(abs(ka[k]-kb[k]).max()>1e-6 for k in shared)
 return {'commonXYVertices':int(common.sum()),'changedCommonXYVertices':int((delta>1e-6).sum()),'maximumCommonVertexHeightChangeM':float(delta.max()) if len(delta) else 0.,'removedTopologyVertices':int((~common).sum()),'addedTopologyVertices':int((cKDTree(a.vertices[:,:2]).query(b.vertices[:,:2])[0]>1e-6).sum()),'changedCommonRoofFaces':int(changed),'removedRoofFaces':len(ka.keys()-kb.keys()),'addedRoofFaces':len(kb.keys()-ka.keys())}

def grade_metrics(mesh):
 e=mesh.edges_unique;d=mesh.vertices[e[:,0]]-mesh.vertices[e[:,1]];h=np.linalg.norm(d[:,:2],axis=1);g=abs(d[:,2])/np.maximum(h,1e-12);use=h>1e-8
 return {'maximumGrade':float(g[use].max()),'edgesOverStructuralGrade3_7':int((g[use]>3.7+1e-6).sum())}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('build','before-profiles','after-profiles','observations','selection','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--before-helper',type=Path);p.add_argument('--after-helper',type=Path);p.add_argument('--roof-module',type=Path)
 a=p.parse_args();
 global roof_shell
 if a.roof_module:
  module=types.ModuleType('pipeline.roof_shell');module.__package__='pipeline';module.__file__=str(a.roof_module);exec(compile(a.roof_module.read_text(),str(a.roof_module),'exec'),module.__dict__);roof_shell=module;pipeline.roof_shell=module;sys.modules['pipeline.roof_shell']=module
 a.output.mkdir(parents=True,exist_ok=True);raw_profiles=[json.loads(q.read_text())['profiles'] for q in (a.before_profiles,a.after_profiles)];before=[];after=[]
 for old,new in zip(*raw_profiles):
  q,issues=clean(old);before.append(q if issues else old);after.append(clean(new)[0])
 profiles_by_mode=[before,after];helpers=[load_old(a.before_helper),load_old(a.after_helper) if a.after_helper else after_helper];obs=json.loads(a.observations.read_text());obs=obs.get('observations',obs);masses=json.loads((a.build/'massing-dump.json').read_text());selection=set(json.loads(a.selection.read_text()));clouds=np.load(a.build/'rooftop-structure-returns.npz');audits={r['buildingId']:r for r in json.loads((a.build/'roof-envelope-audit.json').read_text())};previous=a.output/'audit.json';report=json.loads(previous.read_text())['records'] if previous.exists() else [];completed={r['building'] for r in report};selected_masses=[m for m in masses if (m['id'] in selection or m.get('sourceId') in selection) and m['id'] not in completed];context=contexts(a.build.parent,selected_masses,clouds)
 for mass in masses:
  identity=mass['id'];source_id=mass.get('sourceId',identity)
  if identity in completed or (identity not in selection and source_id not in selection):continue
  foot=Polygon(mass['outline'][0]);ctx=context[identity];raw=ctx[contains_xy(foot,ctx[:,0],ctx[:,1])&(ctx[:,2]>mass['z0']+3)];x0,y0,x1,y1=foot.bounds;nx=max(2,int(np.ceil((x1-x0)/1.5)));ny=max(2,int(np.ceil((y1-y0)/1.5)));inner=raw[contains_xy(foot.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,x0,y1,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,x0,y1,1.5,nx,ny);boundary=raw[keep];base=audits[identity]['baseRoofZ'];results=[];statuses=[]
  for profiles,helper in zip(profiles_by_mode,helpers):
   pipeline.roof_gable_ownership=helper;constraints,host=parameters(obs,profiles,identity,mass['z0']);native=roof_shell.build(foot,inner,base,boundary_points=boundary,eave_constraints=constraints,host_spec=host);prior=[];preparation=[];front=[]
   for profile in profiles:
    match=profile.get('match')or{};part=match.get('partId')
    if not (str(part)==identity or (part is None and str(match.get('featureId'))==source_id)):continue
    if profile.get('gableOutline'):
     item,detail=helper.prepare_gable_prior(profile,mass['z0']);preparation.append(detail)
     if item is not None and item['traceUZ'][:,1].min()>=base-1e-8:prior.append(item)
    for observation in obs:
     if profile in roof_assemblies._profiles_for_side(profiles,observation.get('side'),identity,source_id):
      item,_=roof_front_ownership.prepare_front_prior(observation,profile,mass['z0'],base)
      if item is not None:front.append(item)
   parts,_=roof_source_parts.resolve(obs,profiles,identity,source_id,foot);fixed,_=roof_assemblies.resolve(obs,profiles,identity,source_id,foot,mass['z0'],native['surface']);state=native.get('nativeSupportState');status={'status':'no-admissible-prior','preparation':preparation};candidate=native
   if prior and not front and not fixed and state and not native['audit'].get('observedHostForm',{}).get('status','').startswith('applied-'):
    proposed=helper.solve(state['xy'],state['faces'],state['z'],state['known'],state['unknown'],prior,foot,protected_domains=[p['polygon'] for p in parts]);status=proposed[3]
    if status.get('appliedBoundaryVertices'):
     knots,_=helper.outline_mesh_points(prior,foot);candidate=roof_shell.build(foot,inner,base,boundary_points=boundary,eave_constraints=constraints,host_spec=host,gable_priors=prior,source_outline_points=knots,source_owned_domains=[p['polygon'] for p in parts]);status=candidate['audit']['sourceGableOwnership']
   results.append(candidate['surface']);statuses.append(status)
  delta=mesh_delta(*results);record={'building':identity,'before':statuses[0],'after':statuses[1],'delta':delta,'gradeBefore':grade_metrics(results[0]),'gradeAfter':grade_metrics(results[1]),'qualification':'Common fixed native points are not lowered; replay uses rounded exported footprints and exact saved returns, so not a full-scene acceptance.'};report.append(record)
  if any(delta[k] for k in ['changedCommonXYVertices','removedTopologyVertices','addedTopologyVertices']):
   for label,mesh in zip(['before','after'],results):mesh.export(a.output/f'{identity}-{label}.ply')
  print(json.dumps({'building':identity,**delta}),flush=True)
  (a.output/'audit.json').write_text(json.dumps({'status':'external-roof-only-source-wide-registration-follow-up','productionChanges':0,'beforeProfiles':str(a.before_profiles),'afterProfiles':str(a.after_profiles),'records':report},indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)))
 pipeline.roof_gable_ownership=after_helper
if __name__=='__main__':main()
