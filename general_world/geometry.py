"""Measured support -> closed solids. No facade templates, material aliases or labels."""
import math
import numpy as np
import trimesh
import manifold3d
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon,Point,shape
from shapely.ops import unary_union
from shapely import contains_xy
from scipy.ndimage import label,median_filter
from rasterio.features import shapes
from rasterio.transform import from_origin

def polygons(g):
 if g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 return [p for q in getattr(g,'geoms',[]) for p in polygons(q)]

def extruded(g,bottom,top):
 parts=[]
 for p in polygons(g):
  if p.area<.001:continue
  p=orient(p,sign=1);contours=[np.array(r.coords)[:-1] for r in [p.exterior,*p.interiors]]
  solid=manifold3d.CrossSection(contours).extrude(top-bottom);raw=solid.to_mesh64();m=trimesh.Trimesh(np.asarray(raw.vert_properties)[:,:3],np.asarray(raw.tri_verts),process=False);m.apply_translation([0,0,bottom]);parts.append(m)
 return parts

def union_solids(meshes):
 if not meshes:raise ValueError('No supported solid')
 if any(not m.is_volume for m in meshes):raise ValueError('Input is not a closed positive solid')
 m=trimesh.boolean.union(meshes,engine='manifold',check_volume=True) if len(meshes)>1 else meshes[0]
 if not m.is_volume or not m.is_watertight:raise ValueError('Solid union failed topology gate')
 return m

def canonical_frame(footprint):
 candidates=[]
 for p in polygons(footprint):
  for a,b in zip(p.exterior.coords,list(p.exterior.coords)[1:]):
   v=np.array(b)-a;length=np.linalg.norm(v)
   if length>.01:candidates.append((length,v/length))
 u=max(candidates,key=lambda x:x[0])[1];v=np.array([-u[1],u[0]]);origin=np.array(footprint.centroid.coords[0]);return origin,np.array([u,v])

def supported_roof_layers(footprint,points,terrain,policy):
 """Preserve observed islands and gaps. Never nearest-fill unknown roof cells."""
 from shapely.ops import transform
 origin,R=canonical_frame(footprint)
 def into(x,y,z=None):
  a=np.stack([np.asarray(x)-origin[0],np.asarray(y)-origin[1]],axis=-1)@R.T;return a[...,0],a[...,1]
 local=transform(into,footprint);p=points.copy();p[:,:2]=(p[:,:2]-origin)@R.T
 a,b,c,d=local.bounds;cell=policy.roof_cell_m;nx=max(1,math.ceil((c-a)/cell));ny=max(1,math.ceil((d-b)/cell));raster=np.full((ny,nx),np.nan)
 if len(p):
  ix=np.clip(((p[:,0]-a)/cell).astype(int),0,nx-1);iy=np.clip(((d-p[:,1])/cell).astype(int),0,ny-1)
  groups={}
  for k,z in zip(iy*nx+ix,p[:,2]):groups.setdefault(int(k),[]).append(z)
  for k,zs in groups.items():
   if len(zs)>=policy.minimum_roof_samples:
    z=np.sort(zs);upper=z[max(0,len(z)//2):];spread=np.quantile(upper,.9)-np.quantile(upper,.1)
    if spread<=1.2:raster[k//nx,k%nx]=np.median(upper)
 yy,xx=np.mgrid[:ny,:nx];inside=contains_xy(local,a+(xx+.5)*cell,d-(yy+.5)*cell);raster[~inside]=np.nan
 support=np.isfinite(raster);valid=raster[support]
 if len(valid)<policy.minimum_roof_component_cells:return [],{'observedRoofFraction':float(support.sum()/max(1,inside.sum())),'roofStatus':'insufficient-roof-support'}
 base=float(np.quantile(valid,.15));q=np.where(support,np.round((raster-base)/policy.roof_height_step_m)*policy.roof_height_step_m,0);layers=[]
 def out(x,y,z=None):
  a=np.stack([x,y],axis=-1)@R+origin;return a[...,0],a[...,1]
 for h in sorted(float(x) for x in np.unique(q) if x>0):
  mask=(support&(q>=h));components,count=label(mask);keep=np.zeros_like(mask)
  for i in range(1,count+1):
   piece=components==i
   if piece.sum()>=policy.minimum_roof_component_cells:keep|=piece
  gs=[shape(g) for g,v in shapes(keep.astype('uint8'),mask=keep,transform=from_origin(a,d,cell,cell)) if v]
  if gs:
   g=transform(out,unary_union(gs).intersection(local)).intersection(footprint)
   if not g.is_empty:layers.append((g,base+h))
 return layers,{'baseRoofM':base,'observedRoofFraction':float(support.sum()/max(1,inside.sum())),'roofStatus':'point-supported-islands','unsupportedCellsFilled':0}

def reconstruct_building(footprint,points,terrain,policy,fallback_height=None):
 boundary=[]
 for p in polygons(footprint):
  for r in [p.exterior,*p.interiors]:
   for d in np.arange(0,r.length,.5):boundary.append(r.interpolate(d).coords[0])
 xy=np.array(boundary);ground=np.asarray(terrain(xy[:,0],xy[:,1]));datum=float(np.median(ground));bottom=float(ground.min()-policy.ground_clearance_m)
 if points is None:points=np.empty((0,3))
 points=np.asarray(points);roof=points[points[:,2]>datum+2] if len(points) else points
 layers,report=supported_roof_layers(footprint,roof,terrain,policy)
 top=report.get('baseRoofM')
 if top is None:
  if fallback_height is None:return None,{**report,'status':'unresolved-height','detailStatus':'missing-images'}
  top=datum+float(fallback_height);report['roofStatus']='catalog-height-proxy'
 if top<=float(ground.max())+.25:return None,{**report,'status':'inconsistent-ground-height','detailStatus':'missing-images'}
 solids=extruded(footprint,bottom,top)
 for g,z in layers:
  if z>top+.01:solids.extend(extruded(g,top-.02,z))
 mesh=union_solids(solids)
 return mesh,{**report,'status':'closed-proxy-solid','detailStatus':'missing-images','foundationBottomM':bottom,'minimumBoundaryGroundM':float(ground.min()),'maximumBoundaryGroundM':float(ground.max()),'watertight':bool(mesh.is_watertight),'positiveVolume':bool(mesh.is_volume),'components':len(mesh.split(only_watertight=False)),'windowsInvented':0,'materialsInvented':0,'authoredProfilesUsed':0}

def supported_stairs(path,width,count,terrain,top_rise=None,rail=False):
 """Measured profile must support the treads; handrails use tread datums, not terrain."""
 ps=np.array([path.interpolate(d).coords[0] for d in np.linspace(0,path.length,count+1)]);z=np.asarray(terrain(ps[:,0],ps[:,1]));reverse=z[-1]<z[0]
 if reverse:ps=ps[::-1];z=z[::-1]
 rise=(z[-1]-z[0]) if top_rise is None else float(top_rise)
 if rise<.08:return [],{'accepted':False,'reason':'unresolved-stair-rise'}
 levels=z[0]+np.arange(1,count+1)*rise/count
 if (z[1:]>levels+.08).any() or np.any(np.diff(z)<-.1):return [],{'accepted':False,'reason':'nonmonotonic-or-inconsistent-support'}
 made=[]
 for i in range(count):
  v=ps[i+1]-ps[i];length=np.linalg.norm(v);bottom=min(z[i],z[i+1])-.15;top=levels[i]
  if length<.01 or top<=bottom:return [],{'accepted':False,'reason':'invalid-tread'}
  m=trimesh.creation.box([length+.005,width,top-bottom]);angle=math.atan2(v[1],v[0]);T=trimesh.transformations.rotation_matrix(angle,[0,0,1]);T[:3,3]=[*((ps[i]+ps[i+1])/2),(bottom+top)/2];m.apply_transform(T);m.metadata['component']='tread';made.append(m)
 if rail:
  def member(a,b):
   v=np.array(b)-a;length=np.linalg.norm(v)
   if length<.01:return
   m=trimesh.creation.cylinder(radius=.03,height=length,sections=8);T=trimesh.geometry.align_vectors([0,0,1],v/length);T[:3,3]=(np.array(a)+b)/2;m.apply_transform(T);m.metadata['component']='handrail';made.append(m)
  for sign in [-1,1]:
   tops=[]
   for i,level in enumerate(levels):
    v=ps[i+1]-ps[i];u=v/max(np.linalg.norm(v),.001);n=np.array([-u[1],u[0]])*sign;xy=(ps[i]+ps[i+1])/2+n*(width/2-.08);base=[*xy,level];top=[*xy,level+1];member(base,top);tops.append(top)
   for a,b in zip(tops,tops[1:]):member(a,b)
 return made,{'accepted':True,'treadTops':levels.tolist(),'handrailDatums':(levels+1).tolist() if rail else [],'source':'terrain-profile; metric rise remains estimated'}
