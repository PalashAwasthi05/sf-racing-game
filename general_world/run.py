"""Generic CLI over normalized external geometry. No geography or authored profiles."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import shape,Point,LineString
from scipy.spatial import cKDTree
from .policy import Policy
from .geometry import reconstruct_building,supported_stairs,polygons
from .evidence import contiguous_frames,inspect_record


def execute(input_path,output_dir,policy=None):
 policy=policy or Policy();start=time.perf_counter();path=Path(input_path);d=json.loads(path.read_text());out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 if d.get('schema')!='normalized-world-input/v1':raise ValueError('Input must be normalized geometry observations')
 if d.get('manualProfiles') or d.get('overrides'):raise ValueError('Authored feature overrides are prohibited')
 ground=np.load(path.parent/d['groundFile'])['points'];tree=cKDTree(ground[:,:2])
 def terrain(x,y):
  x,y=np.broadcast_arrays(x,y);p=np.c_[x.ravel(),y.ravel()];dist,idx=tree.query(p,k=min(8,len(ground)));z=np.median(ground[idx,2],axis=-1) if idx.ndim>1 else ground[idx,2];return z.reshape(x.shape)
 cloud=np.load(path.parent/d['surfacePointsFile'])['points'];scene=trimesh.Scene();records=[];queue=[];stair_records=[];source_index=d.get('images',{});detailed=0
 for item in d['buildings']:
  g=shape(item['geometry']);a,b,c,e=g.bounds;mask=(cloud[:,0]>=a)&(cloud[:,0]<=c)&(cloud[:,1]>=b)&(cloud[:,1]<=e)
  from shapely import contains_xy
  p=cloud[mask];p=p[contains_xy(g,p[:,0],p[:,1])]
  mesh,record=reconstruct_building(g,p,terrain,policy,item.get('heightM'));record['key']=item['key'];records.append(record)
  if mesh is not None:
   mesh.visual=trimesh.visual.ColorVisuals(mesh,vertex_colors=[164,171,176,255]);mesh.metadata={'layer':'buildings','featureId':item['key'],'evidence':'unresolved-appearance'};scene.add_geometry(mesh,node_name='solid_'+item['key'])
  for i,frame in enumerate(contiguous_frames(g)):
   # The same request schema is emitted for every boundary, without a name lookup.
   mid=(np.array(frame['a'])+frame['b'])/2;n=np.array(frame['normal']);visible=not frame['courtyard'];area=frame['lengthM']*min(12,item.get('heightM') or 12)
   queue.append({'key':item['key']+':'+str(i),'buildingKey':item['key'],'frame':frame,'visibleArea':float(area) if visible else 0,'attempts':0,'fields':['surface masks','opening boundaries','material scale and albedo','depth changes','ground interfaces','literal text and carrier'],'status':'needs-automatic-image-evidence','minimumIndependentViews':policy.minimum_independent_views,'target':[*mid,4.0],'preferredCamera':list(mid+n*15)})
 # Drape basic mapped surfaces; they are geometry references, not surveyed paint.
 from .surfaces import drape
 for item in d.get('surfaces',[]):
  mesh=drape(shape(item['geometry']),terrain,item.get('offsetM',.03))
  if mesh is not None:
   mesh.visual=trimesh.visual.ColorVisuals(mesh,vertex_colors=item.get('color',[80,85,89,255]));mesh.metadata={'layer':item['layer'],'evidence':'mapped-surface-proxy'};scene.add_geometry(mesh,node_name=item['key'])
 for item in d.get('stairs',[]):
  made,r=supported_stairs(LineString(item['path']),item.get('widthM',2),item['count'],terrain,rail=item.get('handrail',False));stair_records.append({'key':item['key'],**r})
  for j,m in enumerate(made):
   component=m.metadata.get('component','tread');m.visual=trimesh.visual.ColorVisuals(m,vertex_colors=[65,70,73,255] if component=='handrail' else [172,176,174,255]);m.metadata={'layer':'steps','component':component,'evidence':'terrain-derived-approximation'};scene.add_geometry(m,node_name=item['key']+':'+str(j))
 # Detail compiler is deliberately not replaced with a plausible template when image inference is absent.
 verdicts=[inspect_record(r,source_index,policy,d['snapshotDate']) for r in d.get('derivedDetails',[])]
 if any(r['accepted'] for r in verdicts):raise RuntimeError('Accepted automatic details require the image-surface mesher; no placeholder detail renderer is substituted')
 scene.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float));glb=scene.export(file_type='glb');(out/'candidate.glb').write_bytes(glb)
 report={'schema':'generation-report/v1','allFeaturesUseSameRules':True,'manualProfilesRead':0,'buildingsAttempted':len(records),'closedBuildings':sum(r.get('watertight',False) for r in records),'buildingsUnresolved':sum(r.get('status')!='closed-proxy-solid' for r in records),'facadesReconstructedFromImages':detailed,'inventedWindows':0,'inventedFacadeMaterials':0,'inventedSignage':0,'strayDecorativeObjectsGenerated':0,'records':records,'stairs':stair_records,'detailVerdicts':verdicts,'appearanceGate':'FAIL: automatic image reconstruction not connected','publicationApprovedByQualityGate':False,'inputHash':hashlib.sha256(path.read_bytes()).hexdigest(),'outputHash':hashlib.sha256(glb).hexdigest(),'elapsedSeconds':round(time.perf_counter()-start,2),'triangles':sum(len(m.faces) for m in scene.geometry.values())}
 (out/'report.json').write_text(json.dumps(report,indent=2));(out/'acquisition-queue.json').write_text(json.dumps({'surfaces':queue,'automaticQueue':True,'manualBuildingInstructions':False},indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['records','stairs','detailVerdicts']}));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();execute(a.input,a.output)
