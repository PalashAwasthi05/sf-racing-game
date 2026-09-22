"""Whole-area topology rules: one surface owner, supported foundations, roof bands.
Inputs are source geometry/observations. No address-specific corrections.
"""
import numpy as np
import trimesh
from shapely.geometry import Point,Polygon,shape
from shapely.ops import unary_union
from rasterio.features import shapes

def reconcile(candidates):
 audit=[]
 for c in candidates:
  g=c['g'];o=c.get('osm')
  if not o:continue
  q=o['g'];inter=g.intersection(q).area;union=g.union(q).area;iou=inter/max(union,1)
  if c.get('old') and iou>=.62 and .6<q.area/max(g.area,1)<1.6:
   c['g']=q;c['fp']='OSM outline reconciled with municipal identity'
   audit.append({'rule':'outline-reconciliation','buildingId':c['id'],'iou':round(iou,4),'oldAreaM2':round(g.area,2),'selectedAreaM2':round(q.area,2),'sourceAgeVerified':False})
 # Current explicit outlines own overlapping regions; old fallback pieces cannot
 # independently draw another wall/building over the same space.
 ordered=sorted(candidates,key=lambda c:(0 if c['fp'].startswith('OSM') else 1,c['id']))
 owned=Polygon();out=[]
 for c in ordered:
  overlap=c['g'].intersection(owned).area
  if overlap>.01:
   before=c['g'].area;c['g']=c['g'].difference(owned).buffer(0)
   audit.append({'rule':'single-building-owner','buildingId':c['id'],'removedOverlapM2':round(overlap,3),'beforeAreaM2':round(before,3),'remainingAreaM2':round(c['g'].area,3)})
  if c['g'].area<4:continue
  owned=owned.union(c['g']);out.append(c)
 for i,a in enumerate(out):
  for b in out[i+1:]:
   if a['g'].intersection(b['g']).area>.001:raise ValueError('Overlapping ownership survived reconciliation')
 return out,audit

def foundation(ctx,c,datum):
 """Continuous foundation below every terrain edge sample, independent of slope."""
 g=c['g'];samples=[]
 for p in ctx['polys'](g):
  for ring in [p.exterior,*p.interiors]:
   for d in np.arange(0,ring.length,.75):samples.append(ring.interpolate(d).coords[0])
 xy=np.array(samples);z=ctx['elev'](xy[:,0],xy[:,1]);bottom=float(z.min()-.3)
 if datum>bottom:ctx['extrude'](g,bottom,datum-bottom+.01,'foundations__'+c['id'],'stone')
 return {'buildingId':c['id'],'datumM':datum,'minimumBoundaryGroundM':float(z.min()),'maximumBoundaryGroundM':float(z.max()),'foundationBottomM':bottom,'maximumUnfilledGapM':max(0,bottom-float(z.min())),'originalMaximumGapM':max(0,datum-float(z.min())),'edgeSamples':len(z)}

def _parts(g):
 if g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 return [p for p in getattr(g,'geoms',[]) if p.geom_type=='Polygon' and not p.is_empty]

def roof_bands(quant,transform,footprint,minimum_area=25,minimum_width_m=3.0,simplify_m=.2,cell_m=None,level_step_m=1.0,measured_mask=None):
 """Nested superlevel regions: adjacent roof tiers never own the same vertical span.

 A tier is a roof volume, so it must be at least `minimum_width_m` wide everywhere: signs, screens,
 parapets, railings and pipe runs read as raised strips a cell or two wide and are opened away here. The
 width test runs after the clip to the footprint, so clipping cannot leave a sliver at the outline. Outlines
 are simplified at `simplify_m` (the raster resolution): a diagonal tier edge then runs straight along the
 building instead of stepping cell by cell, which would give it a wall block per stair. Because each tier is
 simplified on its own, a shared edge drifts by up to a cell; the strip of the parent tier left around a tier
 that is narrower than a cell (`cell_m`), or any remaining piece of it smaller than four cells, is not a roof of
 its own and joins the tier, whose edge then snaps to the parent's; the roof outline is the parent of the first
 tier. Every part of a tier must hold at least four cells. `quant` contains integer level bins; `level_step_m`
 converts a bin to the physical height returned in `bottom` and `top`. When `measured_mask` is supplied,
 candidate tiers and their narrow edge joins are clipped to cells with measured roof support. This keeps
 unknown regions unresolved without deciding that a known equipment void is a hole to be filled. Without
 these two rules a building's caps
 broke into thousands of two-triangle slivers and cell-wide parts were extruded to full tier height as poles.
 """
 cell=cell_m if cell_m else max(simplify_m,.5)
 if not np.isfinite(level_step_m) or level_step_m<=0:raise ValueError('level_step_m must be positive and finite')
 q=np.asarray(quant)
 if q.ndim!=2:raise ValueError('quant must be a 2-D raster')
 # Quantisation is explicit: a 2.5 m physical step is represented by bins [..,1,2,..],
 # never by truncating a physical value and then treating it as metres.
 q=np.rint(q).astype(np.int64)
 support=None
 if measured_mask is not None:
  measured=np.asarray(measured_mask,dtype=bool)
  if measured.shape!=q.shape:raise ValueError('measured_mask must have the same shape as quant')
  support_parts=[shape(g) for g,v in shapes(measured.astype('uint8'),mask=measured,transform=transform) if v]
  support=unary_union(support_parts) if support_parts else Polygon()
 levels=sorted(int(x) for x in np.unique(q) if x>0);last=0.;result=[]
 for level in levels:
  mask=(q>=level).astype('uint8')
  if measured_mask is not None:mask &= measured.astype('uint8')
  parts=[shape(g) for g,v in shapes(mask,mask=mask.astype(bool),transform=transform) if v]
  g=unary_union(parts).buffer(.45).buffer(-.45).intersection(footprint)
  g=g.buffer(-minimum_width_m/2).buffer(minimum_width_m/2).simplify(simplify_m,preserve_topology=True).intersection(footprint)
  if measured_mask is not None:g=g.intersection(support)
  # The parent of the first tier is the roof outline itself, so the same rules hold at the building's edge.
  parent=result[-1]['geometry'] if result else footprint;g=g.intersection(parent)
  if g.is_empty:continue
  # Keep only narrow shared strips which are actually adjacent to this measured tier. The old
  # empty-wide fallback promoted the whole parent, including disconnected/unmeasured regions.
  strip=parent.difference(g)
  wide=unary_union([p for p in _parts(strip.buffer(-cell/2,join_style=2).buffer(cell/2,join_style=2)) if p.area>=4*cell*cell])
  narrow=strip.difference(wide)
  joins=[]
  for p in _parts(narrow):
   if p.area<1e-9 or p.distance(g)>cell+1e-6:continue
   if measured_mask is not None:p=p.intersection(support)
   if not p.is_empty:joins.append(p)
  if joins:g=unary_union([g,*joins]).buffer(0)
  g=unary_union([p for p in _parts(g) if p.area>=4*cell*cell])
  if g.area>=minimum_area:
   top=float(level*level_step_m)
   result.append({'bottom':last,'top':top,'geometry':g});last=top
 # Consecutive tiers with the same outline are one tier: the lower level owned no roof of its own.
 merged=[]
 for r in result:
  if merged and merged[-1]['geometry'].difference(r['geometry']).area<cell*cell:r['bottom']=merged[-1]['bottom'];merged[-1]=r
  else:merged.append(r)
 result=merged
 # Explicit top surfaces exclude any higher volume, eliminating buried duplicate caps.
 for i,r in enumerate(result):r['cap']=r['geometry'].difference(result[i+1]['geometry']) if i+1<len(result) else r['geometry']
 return result

def perimeter_strip(ctx,c,profiles,depth=1.0):
 from .envelopes import matches,frame
 strips=[]
 for p in profiles:
  if not matches(c,p):continue
  try:f=frame(c,p)
  except ValueError:continue
  u,n=f['tangent'],f['normal'];a=f['left'];b=a+f['width'];plane=f['plane']
  strips.append(Polygon([u*x+n*y for x,y in [(a-.01,plane+.3),(b+.01,plane+.3),(b+.01,plane-depth),(a-.01,plane-depth)]]))
 return unary_union(strips)

def remove_duplicate_triangles(mesh):
 """Remove coincident triangles within a semantic/material batch after UV merging."""
 q=np.round(mesh.triangles,5)
 # Order vertices lexicographically so winding and index identity do not conceal duplicates.
 order=np.lexsort((q[:,:,2],q[:,:,1],q[:,:,0]),axis=1);q=np.take_along_axis(q,order[:,:,None],axis=1).reshape(-1,9)
 _,idx=np.unique(q,axis=0,return_index=True);removed=len(mesh.faces)-len(idx)
 if removed:mesh.update_faces(np.sort(idx));mesh.remove_unreferenced_vertices()
 return removed


def surface_level(zs,gap_m=.8,minimum_share=.15,maximum_spread_m=.6):
 """Roof height of one raster cell from its returns: the highest DENSE, THIN layer.

 A roof surface answers as a dense layer a few decimetres thick (a 15-degree slope spreads .4 m over a
 1.5 m cell). Signs, antennas, railings and plant scatter sparse returns over metres of height above that
 layer; they are not roof surface and must not raise it. Returns are split into layers at vertical gaps,
 and the highest layer holding at least `minimum_share` of the cell's returns with a 10-90% spread under
 `maximum_spread_m` is the surface. None when no layer qualifies (the cell is filled from its neighbours).
 """
 z=np.sort(np.asarray(zs,float));n=len(z)
 if n==0:return None
 cuts=np.where(np.diff(z)>gap_m)[0]+1;layers=np.split(z,cuts)
 thin=[layer for layer in layers if len(layer)>=max(3,minimum_share*n) and (len(layer)<5 or np.quantile(layer,.9)-np.quantile(layer,.1)<=maximum_spread_m)]
 if thin:return float(np.quantile(thin[-1],.8))          # the highest thin layer is the surface
 # No thin layer: a steep roof spreads its returns over the cell (up to ~1.6 m at 45 degrees) and is
 # still a surface; a taller spread is a vertical structure alone in the cell, and the cell is unknown.
 dominant=max(layers,key=len)
 if np.quantile(dominant,.9)-np.quantile(dominant,.1)<=1.6:return float(np.quantile(dominant,.8))
 return None

def surface_raster(points,xmin,ymax,step,nx,ny):
 """Height raster of roof SURFACES plus the mask of returns that belong to one.

 Every cell is judged by surface_level; returns above a cell's surface by more than the layer gap are
 vertical structures and are excluded from every later height statistic (building height, wall raise).
 """
 rr=np.full((ny,nx),np.nan);keep=np.zeros(len(points),bool)
 ix=np.clip(((points[:,0]-xmin)/step).astype(int),0,nx-1);iy=np.clip(((ymax-points[:,1])/step).astype(int),0,ny-1);key=iy*nx+ix
 order=np.argsort(key,kind='stable');sorted_key=key[order];bounds=np.r_[0,np.where(np.diff(sorted_key))[0]+1,len(order)]
 for start,stop in zip(bounds[:-1],bounds[1:]):
  idx=order[start:stop];level=surface_level(points[idx,2])
  if level is None:continue
  k=int(sorted_key[start]);rr[k//nx,k%nx]=level;keep[idx[points[idx,2]<=level+.8]]=True
 return rr,keep


def rooftop_panels(points,surface_mask,rr,xmin,ymax,step,context=None,minimum_returns=40,minimum_length_m=8.,maximum_width_m=2.,minimum_height_m=3.,minimum_face_m=2.,maximum_face_m=8.,minimum_density=2.,beside_occupancy=.35,above_density=.5,audit=None):
 """Thin, tall, linear structures standing on a roof: sign panels, not roof volume and not nothing.

 Returns the surface rule dropped are grouped in plan. A group is a panel when it has the measured
 signature of a sign face and nothing else: it runs along one axis (length >= `minimum_length_m`, 10-90%
 width <= `maximum_width_m`, length >= 6x width), its top stands `minimum_height_m` above the roof under
 it, its returns span `minimum_face_m` to `maximum_face_m` vertically (a face, not a parapet line, a pipe
 or a wall) and number
 `minimum_density` per metre. The wall of a taller roof section or of a neighbour has the same face; it
 differs in having surface at its own top height right beside it, so `context` (all returns around the
 building, inside the footprint or not) is searched 1-8 m to either side of the face at the top height
 (returns occupying more than `beside_occupancy` of that band in plan are a roof, not air; another wall
 face there occupies only a line) and along the face line above its top
 (more than `above_density` returns per metre means the wall goes on). Either way the group is not a sign.
 """
 from scipy.ndimage import label
 above=points[~surface_mask]
 if len(above)<minimum_returns:return []
 ny,nx=rr.shape;ix=np.clip(((above[:,0]-xmin)/step).astype(int),0,nx-1);iy=np.clip(((ymax-above[:,1])/step).astype(int),0,ny-1)
 # Group at half the raster step so a sign face a few metres from a wall face stays its own group.
 fine=step/2;fx=((above[:,0]-xmin)/fine).astype(int);fy=((ymax-above[:,1])/fine).astype(int);fx-=fx.min();fy-=fy.min()
 occupied=np.zeros((fy.max()+1,fx.max()+1),bool);occupied[fy,fx]=True
 labels,count=label(occupied,structure=np.ones((3,3)));group=labels[fy,fx];panels=[]
 def note(**kw):
  if audit is not None:audit.append({k:(round(float(v),2) if isinstance(v,(float,np.floating)) else v) for k,v in kw.items()})
 for k in range(1,count+1):
  sel=group==k;q=above[sel]
  if len(q)<minimum_returns:continue
  xy=q[:,:2];centre=xy.mean(axis=0);_,_,vt=np.linalg.svd(xy-centre,full_matrices=False);axis=vt[0];proj=(xy-centre)@vt.T
  length=float(np.quantile(proj[:,0],.98)-np.quantile(proj[:,0],.02));width=float(np.quantile(proj[:,1],.9)-np.quantile(proj[:,1],.1))
  facts=dict(returns=int(len(q)),lengthM=length,widthM=width,centre=np.round(centre,1).tolist())
  if length<minimum_length_m or width>maximum_width_m or length<6*width or len(q)<minimum_density*length:note(rejected='shape',**facts);continue
  under=rr[iy[sel],ix[sel]];under=under[np.isfinite(under)]
  if not len(under):
   # A sign can hide the roof in every cell it stands in; read the roof from the cells around it.
   from scipy.ndimage import binary_dilation
   ring=np.zeros(rr.shape,bool);ring[iy[sel],ix[sel]]=True;near=rr[binary_dilation(ring,structure=np.ones((3,3)),iterations=2)];under=near[np.isfinite(near)]
   if not len(under):under=rr[np.isfinite(rr)]
   if not len(under):continue
  roof=float(np.median(under));top=float(np.quantile(q[:,2],.95));bottom=max(roof,float(np.quantile(q[:,2],.10))-.3);facts.update(roofZ=roof,bottomZ=bottom,topZ=top)
  # A sign is a panel a few metres tall; a thin face taller than `maximum_face_m` is a wall or a screen.
  if top-roof<minimum_height_m or top-bottom<minimum_face_m or top-bottom>maximum_face_m:note(rejected='height',**facts);continue
  if context is not None and len(context):
   rel=context[:,:2]-centre;along=rel@axis;across=rel@vt[1];inside=np.abs(along)<=length/2
   # Surface at or above the face top, 1-8 m to either side: the face is a wall under a higher roof.
   # A roof fills that band in plan; another wall face there is only a line, so plan occupancy decides.
   band=inside&(np.abs(across)>1.0)&(np.abs(across)<=8.0)&(context[:,2]>=top-1.5)&(context[:,2]<=top+3.0)
   cells=set(zip(np.floor(along[band]).astype(int).tolist(),np.floor(across[band]).astype(int).tolist()))
   facts.update(besideOccupancy=len(cells)/(2*7.0*max(length,1e-9)))
   if facts['besideOccupancy']>beside_occupancy:note(rejected='roof-beside-top',**facts);continue
   # Returns continuing up the same line above the face: a wall that goes on, not a sign that ends.
   higher=inside&(np.abs(across)<=1.5)&(context[:,2]>top+1.5);facts.update(abovePerM=higher.sum()/length)
   if facts['abovePerM']>above_density:note(rejected='wall-continues-above',**facts);continue
  note(rejected=None,**facts)
  panels.append({'centre':centre,'axis':axis,'lengthM':length,'widthM':width,'roofZ':roof,'bottomZ':bottom,'topZ':top,'returns':int(len(q))})
 return panels


def sloped_cells(rr,step,minimum_slope=.08,maximum_slope=1.0):
 """Raster cells that lie on a continuous roof slope rather than on a flat plateau or at a step.

 A wall is built only where the returns show one: a jump between neighbouring cells. Across a gradual
 gradient (between `minimum_slope` and `maximum_slope` metres per metre, about 5 to 45 degrees) the roof
 is one surface; quantising it into levels would invent a staircase of walls the LiDAR never measured.
 The gradient is read on the filled, unfiltered raster so a real step stays a sharp jump.
 """
 filled=np.where(np.isfinite(rr),rr,np.nanmedian(rr));gy,gx=np.gradient(filled,step);grade=np.hypot(gx,gy)
 return (grade>=minimum_slope)&(grade<=maximum_slope)
