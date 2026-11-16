"""Trace remaining roof shoulders to measured support vs unknown perimeter."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon,Point,LineString
import trimesh
from pipeline import roof_shell,spatial_rules
from regen.rebuild_roof_support_review import parameters


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('build','profiles','observations','after','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--profile',required=True);a=p.parse_args();profile=next(q for q in json.loads(a.profiles.read_text())['profiles'] if q['id']==a.profile);identity=(profile.get('match')or{}).get('partId')or profile['match']['featureId'];mass=next(q for q in json.loads((a.build/'massing-dump.json').read_text()) if q['id']==identity);g=Polygon(mass['outline'][0]);clouds=np.load(a.build/'rooftop-structure-returns.npz');ctx=clouds[identity+'__context'];raw=ctx[contains_xy(g,ctx[:,0],ctx[:,1])&(ctx[:,2]>mass['z0']+3)];x0,y0,x1,y1=g.bounds;nx=max(2,int(np.ceil((x1-x0)/1.5)));ny=max(2,int(np.ceil((y1-y0)/1.5)));inner=raw[contains_xy(g.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,x0,y1,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,x0,y1,1.5,nx,ny);boundary=raw[keep];obs=json.loads(a.observations.read_text());obs=obs.get('observations',obs);profiles=json.loads(a.profiles.read_text())['profiles'];constraints,host=parameters(obs,profiles,identity,mass['z0']);shell=roof_shell.build(g,inner,mass['baseRoofZ'],boundary_points=boundary,eave_constraints=constraints,host_spec=host);state=shell['nativeSupportState'];xy=state['xy'];known=state['known'];distance,index=cKDTree(shell['surface'].vertices[:,:2]).query(xy);native_z=shell['surface'].vertices[index,2];after=trimesh.load(a.after,force='scene',process=False);vertices=np.vstack([m.vertices for k,m in after.geometry.items() if k.startswith('roofs__'+identity+'__')]);tree=cKDTree(vertices[:,:2]);dist,ind=tree.query(xy);after_z=vertices[ind,2];n=np.array(profile['normal']);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]]);u=xy@t;depth=profile['planeV']-xy@n;trace=np.array(profile['parapets'][0]['top']);span=profile['frameU'][0]+trace[[0,-1],0]*(profile['frameU'][1]-profile['frameU'][0]);coords=list(g.exterior.coords);rows=[]
 for i,(aa,bb) in enumerate(zip(coords,coords[1:])):
  line=LineString([aa,bb]);on=np.array([line.distance(Point(q))<1e-7 for q in xy]);inside=on&(u>=span[0]-1e-6)&(u<=span[1]+1e-6);near=on&(depth>=-.1)&(depth<=3.);front=on&(abs(depth)<.1)
  rows.append({'edgeIndex':i,'a':list(aa),'b':list(bb),'lengthM':line.length,'boundaryVertices':int(on.sum()),'measuredVertices':int((on&known).sum()),'unknownVertices':int((on&~known).sum()),'sourceURangeM':[float(u[on].min()),float(u[on].max())],'inwardDepthRangeM':[float(depth[on].min()),float(depth[on].max())],'nativeHeightRangeM':(native_z[on]-mass['z0']).tolist(),'afterHeightRangeM':(after_z[on]-mass['z0']).tolist(),'measuredMask':known[on].tolist(),'nearFront3mVertices':int(near.sum()),'nearFront3mUnknownVertices':int((near&~known).sum()),'insideObservedSpanVertices':int(inside.sum()),'frontPlaneVertices':int(front.sum()),'maximumCandidateXYMatchDistanceM':float(dist[on].max())})
 result={'status':'read-only-measured-versus-unknown-boundary-audit','building':identity,'profileId':profile['id'],'sourceObservedAbsoluteU':span.tolist(),'edges':rows,'structuralPlanes':shell['audit']['structuralPlaneSurface']['planes'],'productionChanges':0,'qualification':'Source camera/XY remains conditional. High boundary closure is not a wall observation. No measured plane is removed based on projection alone.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
 for r in rows:
  if r['nearFront3mVertices']:print({k:r[k] for k in ['edgeIndex','lengthM','boundaryVertices','measuredVertices','unknownVertices','sourceURangeM','inwardDepthRangeM','insideObservedSpanVertices','frontPlaneVertices']})
if __name__=='__main__':main()
