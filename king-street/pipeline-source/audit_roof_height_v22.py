import re
"""Compare rendered roof heights against sampled measured heights, not plan coverage alone."""
import json,argparse
from pathlib import Path
import numpy as np
from shapely import polygons,points,STRtree,contains_xy
from shapely.geometry import shape
from replace_scene_layers_v21 import _read_glb
from pipeline.roof_envelope import prepare_grid
from pipeline.spatial_rules import surface_raster

def audit(scene,inputs,output):
 d=_read_glb(scene);j=d.json
 def acc(i):
  a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];typ={5123:'u2',5125:'u4',5126:'f4'}[a['componentType']]
  return np.frombuffer(d.binary,dtype=typ,count=a['count']*n,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'],n)
 roof={}
 # Generator and texture adapters use a flat scene and one coordinate transform.
 for node in j['nodes']:
  if 'mesh' not in node:continue
  mesh=j['meshes'][node['mesh']];meta=mesh.get('extras',{});name=mesh.get('name','')
  if meta.get('layer',name.split('__')[0])!='roofs':continue
  fid=meta.get('featureId') or name.split('__')[1].split('___')[0]
  fid=re.sub(r'_p\d+$','',fid)   # a split part's roof belongs to its source footprint
  M=np.array(node.get('matrix',np.eye(4).T.ravel()),float).reshape(4,4).T
  for p in mesh['primitives']:
   v=acc(p['attributes']['POSITION']);v=v@M[:3,:3].T+M[:3,3]
   v=v[:,[0,2,1]]*np.array([1,-1,1]) # GLB y-up -> source ENU
   t=v[acc(p['indices']).ravel().reshape(-1,3)]
   normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);t=t[normal[:,2]>.001]
   roof.setdefault(fid,[]).append(t)
 records=[];data=json.loads((inputs/'building-roof-inputs.json').read_text());cloud=np.load(inputs/'building-roof-points.npz')
 for row in data['buildings']:
  fid=row['id'];g=shape(row['geometry']);p=cloud[fid]
  if fid not in roof or len(p)<50:continue
  # Reference medians reject narrow rooftop structures using the same general
  # surface-return selection, but do not use the generated tier heights.
  xmin,ymin,xmax,ymax=g.bounds;step=2. if g.area>18000 else 1.5
  rr,mask=surface_raster(p,xmin,ymax,step,max(2,int(np.ceil((xmax-xmin)/step))),max(2,int(np.ceil((ymax-ymin)/step))))
  if rr is None:continue
  grid,known,xy,_=prepare_grid(rr,g,xmin,ymax,step)
  if grid is None:continue
  xy=xy[known.ravel()];target=grid[known]
  triangles=np.concatenate(roof[fid]);polys=polygons(triangles[:,:,:2]);pairs=STRtree(polys).query(points(xy),predicate='intersects')
  actual=np.full(len(xy),-np.inf)
  if pairs.shape[1]:
   qi,ti=pairs;tr=triangles[ti];q=xy[qi];a=tr[:,0,:2];b=tr[:,1,:2]-a;c=tr[:,2,:2]-a;v=q-a;den=b[:,0]*c[:,1]-b[:,1]*c[:,0]
   u=(v[:,0]*c[:,1]-v[:,1]*c[:,0])/den;w=(b[:,0]*v[:,1]-b[:,1]*v[:,0])/den
   z=tr[:,0,2]+u*(tr[:,1,2]-tr[:,0,2])+w*(tr[:,2,2]-tr[:,0,2]);np.maximum.at(actual,qi,z)
  ok=np.isfinite(actual);err=actual[ok]-target[ok]
  records.append({'buildingId':fid,'samples':len(xy),'missingRoofSamples':int((~ok).sum()),'medianAbsoluteErrorM':round(float(np.median(abs(err))),2) if len(err) else None,'p90AbsoluteErrorM':round(float(np.quantile(abs(err),.9)),2) if len(err) else None,'roofMoreThan4mBelowMeasured':int((err < -4).sum()),'roofMoreThan4mAboveMeasured':int((err>4).sum())})
 out={'scene':str(scene),'records':records,'samples':sum(r['samples'] for r in records),'missingRoofSamples':sum(r['missingRoofSamples'] for r in records),'roofMoreThan4mBelowMeasured':sum(r['roofMoreThan4mBelowMeasured'] for r in records),'roofMoreThan4mAboveMeasured':sum(r['roofMoreThan4mAboveMeasured'] for r in records),'caveat':'Point clouds include occlusions and obsolete structures; this measures disagreement, not certified as-built error.'}
 output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--inputs',type=Path,default=Path('work/building-v22-inputs'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.scene,a.inputs,a.output)
