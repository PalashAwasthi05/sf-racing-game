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

def roof_bands(quant,transform,footprint,minimum_area=25):
 """Nested superlevel regions: adjacent roof tiers never own the same vertical span."""
 levels=sorted(int(x) for x in np.unique(quant) if x>0);last=0;result=[]
 for level in levels:
  mask=(quant>=level).astype('uint8');parts=[shape(g) for g,v in shapes(mask,mask=mask.astype(bool),transform=transform) if v]
  g=unary_union(parts).buffer(.45).buffer(-.45).simplify(.2).intersection(footprint)
  if result:g=g.intersection(result[-1]['geometry'])
  if g.area>=minimum_area:
   result.append({'bottom':last,'top':level,'geometry':g});last=level
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
