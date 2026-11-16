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
  if overlap>.001:   # the same tolerance the check below holds (row houses share walls to the millimetre)
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
 return [q for p in getattr(g,'geoms',[]) if not p.is_empty for q in (_parts(p) if p.geom_type in ('MultiPolygon','GeometryCollection') else [p] if p.geom_type=='Polygon' else [])]

def _overlay(op,a,b):
 """A shapely overlay that falls back to a fixed-precision (1 mm snap-rounded) overlay when GEOS cannot resolve
 near-coincident edges (a large outline with long collinear walls failed 'unable to determine overlay result')."""
 import shapely
 try:return getattr(a,op)(b)
 except shapely.errors.GEOSException:
  from shapely import make_valid
  area=lambda g:unary_union([q for q in _parts(make_valid(g))]) if not g.is_empty else g   # polygons only: no stray lines
  return getattr(shapely,op)(area(a),area(b),grid_size=1e-3)

def regularise(g,cell,axis_tolerance_deg=15.,original=None):
 """A tier outline made of walls along the raster's axes. A raster cell straddling a wall reads either side of it, so a
 straight wall alternated between two cell lines; simplified, the jogs became walls a few degrees off true and short
 returns between them. Edges within `axis_tolerance_deg` of an axis are turned onto it; parallel walls a jog of at most a
 cell apart become one wall at their length-weighted line. Oblique edges (a chord across a stepped survey edge, a real
 diagonal wall) stay, moved onto the least-squares line of the `original` boundary they simplify: the simplified chord
 of a cell staircase runs through its outer corners, half a step outside the wall. A ring that does not come out a valid
 polygon of nearly the same area is kept as it was."""
 samples=None
 if original is not None and not original.is_empty:
  from shapely import segmentize
  samples=np.concatenate([np.asarray(r.coords) for q in _parts(segmentize(original,.25)) for r in [q.exterior]+list(q.interiors)])
 out=[]
 for p in _parts(g):
  rings=[]
  for ring in [p.exterior]+list(p.interiors):
   xy=np.asarray(ring.coords)[:-1]
   r=_regular_ring(xy,cell,axis_tolerance_deg,samples) if len(xy)>=4 else None
   rings.append(r if r is not None else xy)
  q=Polygon(rings[0],rings[1:])
  if not q.is_valid:q=q.buffer(0)
  ref=original.intersection(p.buffer(cell,join_style=2)).area if original is not None else p.area   # the cells it stands for
  if q.is_empty or q.geom_type!='Polygon' or abs(q.area-ref)>.08*ref+cell*cell:q=p
  out.append(q)
 return unary_union(out) if out else g

def _regular_ring(xy,jog,tolerance_deg,samples=None):
 tol=np.radians(tolerance_deg);edges=[]
 n=len(xy)
 for i in range(n):
  a,b=xy[i],xy[(i+1)%n];d=b-a;L=float(np.hypot(*d))
  if L<1e-6:continue
  ang=abs(np.arctan2(d[1],d[0]))%np.pi
  if min(ang,np.pi-ang)<=tol:edges.append(['h',(a[1]+b[1])/2,L,a,b])
  elif abs(ang-np.pi/2)<=tol:edges.append(['v',(a[0]+b[0])/2,L,a,b])
  else:
   if L<=1.5*jog:
    # a single cell cut off a corner (the chord across one missing corner cell) is below the raster's resolution: the
    # corner is square unless a finer reading says otherwise
    edges.append(['c',None,L,a,b]);continue
   if samples is not None and L>=2*jog:
    # the oblique chord onto the mean line of the boundary it replaced (samples along its middle, within a cell or so)
    t=d/L;nrm=np.array([-t[1],t[0]]);rel=samples-a;al=rel@t;off=rel@nrm;m=(al>.1*L)&(al<.9*L)&(np.abs(off)<1.5*jog)
    if m.sum()>=4:
     shift=float(np.mean(off[m]));a=a+nrm*shift;b=b+nrm*shift
   edges.append(['o',None,L,a,b])
 out_=[]
 for i,e in enumerate(edges):
  if e[0]=='c':
   p_,n_=edges[i-1],edges[(i+1)%len(edges)]
   if {p_[0],n_[0]}=={'h','v'}:continue          # between two perpendicular walls: the corner is theirs
   e=['o',None,e[2],e[3],e[4]]
  out_.append(e)
 edges=out_
 if len(edges)<4:return None
 def merge(e,f):
  L=e[2]+f[2];return [e[0],(e[1]*e[2]+f[1]*f[2])/L,L,e[3],f[4]]
 changed=True
 while changed and len(edges)>=4:
  changed=False
  for i in range(len(edges)):
   e,f=edges[i],edges[(i+1)%len(edges)]
   # neighbours on one axis (an edge turned onto the axis beside another): one wall
   if e[0]==f[0]!='o' and abs(e[1]-f[1])<=jog+1e-9:
    edges[i]=merge(e,f);del edges[(i+1)%len(edges)];changed=True;break
   # a jog: a short return between two parallel walls a cell or less apart
   g_=edges[(i+2)%len(edges)]
   if e[0]!='o' and g_[0]==e[0] and f[0] not in ('o',e[0]) and f[2]<=jog+1e-9 and abs(e[1]-g_[1])<=jog+1e-9 and len(edges)>4:
    m=merge(e,g_);j=(i+1)%len(edges);k=(i+2)%len(edges)
    for idx in sorted({j,k},reverse=True):del edges[idx]
    edges[i if i<len(edges) else 0]=m
    changed=True;break
 if len(edges)<4:return None
 pts=[]
 for i in range(len(edges)):
  e,f=edges[i],edges[(i+1)%len(edges)]
  if e[0]=='h' and f[0]=='v':pts.append((f[1],e[1]))
  elif e[0]=='v' and f[0]=='h':pts.append((e[1],f[1]))
  elif e[0]==f[0]=='o':
   a,b,c_,d_=[np.asarray(v,float) for v in (e[3],e[4],f[3],f[4])];r_=b-a;s_=d_-c_;den=r_[0]*s_[1]-r_[1]*s_[0]
   pts.append(tuple(a+r_*(((c_-a)[0]*s_[1]-(c_-a)[1]*s_[0])/den)) if abs(den)>1e-9 else tuple((b+c_)/2))
  elif e[0]==f[0]:
   # two parallel walls a jog apart that stayed apart: a short square return at the joint
   x=e[4] if True else None
   pts.append((e[4][0],e[1]) if e[0]=='h' else (e[1],e[4][1]));pts.append((e[4][0],f[1]) if e[0]=='h' else (f[1],e[4][1]))
  else:
   # an axis wall meeting an oblique edge: where the oblique edge's line crosses the wall's line
   ax,ob=(e,f) if e[0]!='o' else (f,e);a,b=np.asarray(ob[3],float),np.asarray(ob[4],float);d=b-a
   if ax[0]=='h':t=(ax[1]-a[1])/d[1] if abs(d[1])>1e-9 else None
   else:t=(ax[1]-a[0])/d[0] if abs(d[0])>1e-9 else None
   pts.append(tuple(a+d*t) if t is not None and -1.<=t<=2. else tuple(e[4]))
 return np.asarray(pts,float)

def _small_pockets(mask,measured,maximum_cells=4):
 """Unknown cells in holes of `mask` that it encloses completely, up to `maximum_cells` per hole (a 2 x 2 pocket).
 A notch open to the outside is never a pocket: filling notches pulled unknown cells beside a wall into the block."""
 from scipy.ndimage import binary_fill_holes,label
 holes=binary_fill_holes(mask)&~mask
 if not holes.any():return holes
 lab,n=label(holes);sizes=np.bincount(lab.ravel());small=(sizes<=maximum_cells);small[0]=False
 return small[lab]&~measured

class OutlineFrame:
 """The outline's own axes, for its roof raster.

 A roof raster laid north-up over a footprint at 45 degrees cuts every wall into a staircase of cells: tier outlines
 kept about 300 vertices each, their walls stood as scalloped zig-zags, and two tiers of one vertical wall never had
 the same outline, so a straight block became a stack of boxes. The frame turns about the footprint's centroid by the
 angle of the long side of its minimum rotated rectangle (the rule building_parts.split uses): `points` and `geom`
 take world data into the frame, `world_points`, `world_vectors`, `world` and `world_mesh` bring results back. An
 outline already on the grid axes keeps the identity."""
 def __init__(self,footprint):
  import math
  self.angle=0.;self.ox=self.oy=0.
  try:rect=footprint.minimum_rotated_rectangle
  except Exception:return
  if rect.geom_type!='Polygon' or rect.area<1e-6:return
  xy=np.asarray(rect.exterior.coords)[:4];e=xy[1]-xy[0] if np.linalg.norm(xy[1]-xy[0])>=np.linalg.norm(xy[2]-xy[1]) else xy[2]-xy[1]
  ang=math.atan2(e[1],e[0])
  if abs(math.sin(2*ang))<1e-3:return
  o=footprint.centroid;self.angle=float(ang);self.ox,self.oy=float(o.x),float(o.y)
 @property
 def identity(self):return self.angle==0.
 def _turn(self,xy,ang,about=True):
  xy=np.array(xy,dtype=float,copy=True)
  if self.identity or len(xy)==0:return xy
  c,s=np.cos(ang),np.sin(ang);ox,oy=(self.ox,self.oy) if about else (0.,0.)
  x=xy[...,0]-ox;y=xy[...,1]-oy;xy[...,0]=ox+x*c-y*s;xy[...,1]=oy+x*s+y*c
  return xy
 def points(self,p):return self._turn(p,-self.angle)
 def world_points(self,p):return self._turn(p,self.angle)
 def world_vectors(self,v):return self._turn(v,self.angle,about=False)
 def geom(self,g):
  from shapely import affinity
  return g if self.identity or g is None or g.is_empty else affinity.rotate(g,-self.angle,origin=(self.ox,self.oy),use_radians=True)
 def world(self,g):
  from shapely import affinity
  return g if self.identity or g is None or g.is_empty else affinity.rotate(g,self.angle,origin=(self.ox,self.oy),use_radians=True)
 def world_mesh(self,mesh):
  if mesh is None or self.identity:return mesh
  mesh=mesh.copy();v=np.asarray(mesh.vertices).copy();v[:,:2]=self.world_points(v[:,:2]);mesh.vertices=v;return mesh

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
 # the domain as polygons only: a difference with slope patches can leave stray lines, and a mixed collection breaks overlays
 if footprint.geom_type not in ('Polygon','MultiPolygon') or not footprint.is_valid:
  from shapely import make_valid
  footprint=unary_union(_parts(make_valid(footprint)))
 q=np.asarray(quant)
 if q.ndim!=2:raise ValueError('quant must be a 2-D raster')
 # Quantisation is explicit: a 2.5 m physical step is represented by bins [..,1,2,..],
 # never by truncating a physical value and then treating it as metres.
 q=np.rint(q).astype(np.int64)
 support=None;usable=None;lvl=q
 if measured_mask is not None:
  measured=np.asarray(measured_mask,dtype=bool)
  if measured.shape!=q.shape:raise ValueError('measured_mask must have the same shape as quant')
  # A cell holding no roof surface of its own (its returns straddle a wall or a penthouse face, or none came back) that
  # borders measured roof stands at the lowest level of its measured neighbours: it belongs to every level that
  # surrounds it and to none above. Left out of all of them, the ring of such cells round a plant room cut a notch into
  # every level below it, open to the edge where the room stood near a wall. Filled values (prepare_grid copies the
  # nearest measured cell into unknown ones) never make a level; unknown cells with no measured neighbour stay out.
  big=np.iinfo(np.int64).max;qm=np.pad(np.where(measured,q,big),1,constant_values=big);H,W=q.shape;near=np.full(q.shape,big)
  for (dy,dx) in ((0,1),(1,0),(1,1),(1,-1)):
   # measured on both sides of the cell along one of four directions: the cell lies between them (a ring round a
   # plant room, a wall's straddling cells), not beyond them (a channel, the edge of the survey, an isolated return)
   a=qm[1+dy:1+dy+H,1+dx:1+dx+W];b=qm[1-dy:1-dy+H,1-dx:1-dx+W];both=(a<big)&(b<big);near=np.where(both,np.minimum(near,np.minimum(a,b)),near)
  adjacent=(~measured)&(near<big);lvl=np.where(measured,q,np.where(adjacent,near,-1));usable=measured|adjacent
  # An unknown pocket a cell or two wide inside measured roof (a missed return, a dark hatch) is roof, not a hole:
  # punched out of the tiers, each one became a pit with four walls no return had shown.
  closed=usable|_small_pockets(usable,measured)
  support_parts=[shape(g) for g,v in shapes(closed.astype('uint8'),mask=closed,transform=transform) if v]
  support=unary_union(support_parts) if support_parts else Polygon()
  # Support is closed at the tier width: a notch of unknown cells narrower than a tier along the edge of measured roof
  # is no region of its own. Clipped to the bare cells, every tier kept a 1.5 m tooth at each such cell.
  # The ground beyond the outline counts as closed for this, so a notch at the building's edge closes like a pocket.
  if not support.is_empty:
   r_=max(cell,minimum_width_m)/2-.05;beyond=_overlay('difference',footprint.envelope.buffer(4*cell,join_style=2),footprint)
   support=_overlay('intersection',unary_union([support,beyond]).buffer(r_,join_style=2).buffer(-r_,join_style=2),footprint)
 levels=sorted(int(x) for x in np.unique(q) if x>0);last=0.;result=[]
 for level in levels:
  mask=(q>=level).astype('uint8')
  if measured_mask is not None:
   own=(lvl>=level)&usable;own|=_small_pockets(own,measured);mask=own.astype('uint8')
  parts=[shape(g) for g,v in shapes(mask,mask=mask.astype(bool),transform=transform) if v]
  # Square joins: the raster runs along the outline's axes (OutlineFrame), so a tier's corners are right angles; round
  # joins turned each into an arc of 1.5 m radius, a fan of short walls at every corner.
  g=_overlay('intersection',unary_union(parts).buffer(.45,join_style=2).buffer(-.45,join_style=2),footprint)
  g=_overlay('intersection',g.buffer(-minimum_width_m/2,join_style=2).buffer(minimum_width_m/2,join_style=2),footprint)
  if measured_mask is not None:g=_overlay('intersection',g,support)
  # Simplified after every clip (a clip to the cells brings their staircase back), then made of walls along the frame.
  g=regularise(g.simplify(simplify_m,preserve_topology=True),cell,original=g)
  # A regularised wall a few millimetres off the outline's own edge leaves a sliver that would join as a 'narrow strip'
  # and run the whole edge as a hair-thin spike: walls within 5 cm of the outline are put on it.
  from shapely import snap
  g=snap(g,footprint,.05);g=_overlay('intersection',g if g.is_valid else g.buffer(0),footprint)
  # The parent of the first tier is the roof outline itself, so the same rules hold at the building's edge.
  parent=result[-1]['geometry'] if result else footprint;g=_overlay('intersection',g,parent)
  if g.is_empty:continue
  # Keep only narrow shared strips which are actually adjacent to this measured tier. The old
  # empty-wide fallback promoted the whole parent, including disconnected/unmeasured regions.
  # A strip of the parent is a roof of its own (a terrace or a setback) only when it is as wide as a tier must be; a
  # narrower one is the same wall read by two levels (a cell or two of raster drift, a parapet, a cornice), and the tier
  # takes it, so a vertical wall's levels share one outline and merge below. Joined at a cell's width, the levels of one
  # straight block kept 1-2 m of drift between them and stood as three stacked boxes, each with its own cornice.
  strip=_overlay('difference',parent,g);opening=max(cell,minimum_width_m)/2-.05   # strictly narrower than a tier
  wide=unary_union([p for p in _parts(strip.buffer(-opening,join_style=2).buffer(opening,join_style=2)) if p.area>=4*cell*cell])
  narrow=_overlay('difference',strip,wide)
  joins=[]
  for p in _parts(narrow):
   if p.area<1e-9 or p.distance(g)>cell+1e-6:continue
   p=p.buffer(-.01,join_style=2).buffer(.01,join_style=2)   # hair-thin slivers of float drift are no strip
   if p.is_empty:continue
   # A strip joins only beside the tier (within a tier's width of it): a long thin remnant of the domain touching the
   # tier at one end (the rim left round a draped courtyard ring) spread a small tower's top along the whole outline.
   p=_overlay('intersection',p,g.buffer(max(cell,minimum_width_m),join_style=2))
   if p.is_empty or p.area<1e-6:continue
   # Only the same wall read twice joins: a strip whose measured cells stand clearly lower than this level (more than a
   # bin below: a canopy, a lower storey or ledge along the building's edge) is a roof of its own however narrow, and
   # joined it stood the tier's full height over it. Cells straddling the wall read in between and still join. Every
   # cell the strip touches counts, so a sliver thinner than a cell answers to the returns under it too.
   # The parts of a strip no wider than a cell are the wall's own straddling cells (their returns mix the two sides) and
   # always join; a part at least a cell wide joins only when it stands at this level.
   core=p.buffer(-(cell/2+.02),join_style=2).buffer(cell/2+.02,join_style=2)
   if measured_mask is not None and not core.is_empty:
    from rasterio.features import geometry_mask
    thin=_overlay('difference',p,core)
    inside=~geometry_mask([core],out_shape=q.shape,transform=transform,all_touched=True)&measured
    if inside.sum()>=1 and float(np.mean(lvl[inside]>=level-1))<.5:
     p=thin
     if p.is_empty or p.area<1e-6:continue
   if measured_mask is not None:p=_overlay('intersection',p,support)
   if not p.is_empty:joins.append(p)
  if joins:g=unary_union([g,*joins]).buffer(0)
  # A last tidy once the strips are joined (their unions leave repeated points, sub-cell jogs and corner chips): the
  # outline is made of walls again and put back inside its parent.
  from shapely import snap
  if not g.is_empty:
   g=regularise(g.simplify(simplify_m/4,preserve_topology=True),cell);g=snap(g,parent,.05)
   g=_overlay('intersection',g if g.is_valid else g.buffer(0),parent)
  g=unary_union([p for p in _parts(g) if p.area>=4*cell*cell])
  if g.area>=minimum_area:
   top=float(level*level_step_m)
   result.append({'bottom':last,'top':top,'geometry':g});last=top
 # Consecutive tiers with the same outline are one tier: the lower level owned no roof of its own.
 merged=[]
 for r in result:
  if merged and _overlay('difference',merged[-1]['geometry'],r['geometry']).area<cell*cell:r['bottom']=merged[-1]['bottom'];merged[-1]=r
  else:merged.append(r)
 result=merged
 # Explicit top surfaces exclude any higher volume, eliminating buried duplicate caps.
 for i,r in enumerate(result):r['cap']=_overlay('difference',r['geometry'],result[i+1]['geometry']) if i+1<len(result) else r['geometry']
 return result

def measured_tiers(bands,rr,known,transform,base,first_rise_m=3.,next_rise_m=2.5):
 """Tier heights from the returns under each tier's own roof, and tiers kept by that measured rise.

 The 2.5 m bins find where the tiers are; a bin's nominal height does not set them. A main roof 3.35 m above a lower
 strip rounded to the first bin, was dropped as rooftop plant, and the block stood 3 m under its measured roof. A tier
 whose own roof rises less than a storey above the roof it stands on (3 m above the base roof, 2.5 m above another
 tier, as the bins allowed before) is plant, a parapet or a screen, and becomes part of the roof below it."""
 from rasterio.features import geometry_mask
 kept=[]
 for i,b in enumerate(bands):
  above=bands[i+1]['geometry'] if i+1<len(bands) else None
  cap=_overlay('difference',b['geometry'],above) if above is not None else b['geometry']
  inside=(~geometry_mask([cap],out_shape=rr.shape,transform=transform)&known) if not cap.is_empty else np.zeros(rr.shape,bool)
  rise=float(np.median(rr[inside]-base)) if inside.sum()>=4 else float(b['top'])
  below=kept[-1]['top'] if kept else 0.
  if rise-below<(next_rise_m if kept else first_rise_m):continue
  kept.append({**b,'bottom':below,'top':round(rise,3),'nominalTop':b['top']})
 return settle_tops(kept,rr,known,transform,base,next_rise_m)

def settle_tops(kept,rr,known,transform,base,next_rise_m=2.5,minimum_cap_m2=25.):
 """Each kept tier stands at the median of the roof it actually carries: its region less the next kept tier's, the caps
 of the tiers dropped between them included. Judged on its own cap alone, a tier whose cap was a thin ring of edge cells
 stood at that ring's height, and the main roof above it (dropped as rising too little over the ring) was drawn with it,
 metres low. A tier that settles within next_rise_m of the one below joins it."""
 from rasterio.features import geometry_mask
 kept=[dict(b) for b in kept];changed=True
 while changed:
  changed=False
  for i,b in enumerate(kept):
   cap=_overlay('difference',b['geometry'],kept[i+1]['geometry']) if i+1<len(kept) else b['geometry'];b['cap']=cap
   if rr is not None and not cap.is_empty:
    inside=~geometry_mask([cap],out_shape=rr.shape,transform=transform)&known
    if inside.sum()>=4:b['top']=round(float(np.median(rr[inside]-base)),3)
  for i in range(1,len(kept)):
   if kept[i]['top']-kept[i-1]['top']<next_rise_m:
    # too little rise over the tier below once both are settled: one roof (the lower region holds the upper one)
    del kept[i];changed=True;break
  if changed:continue
  for i in range(len(kept)-1):
   if kept[i]['cap'].area<minimum_cap_m2:
    # a tier whose own roof is a sliver beside the tier above (two levels reading one wall) is that tier: the tier
    # above takes its outline, so the sliver does not fall to the roof below
    kept[i+1]['geometry']=kept[i]['geometry'];del kept[i];changed=True;break
 for i,b in enumerate(kept):b['bottom']=kept[i-1]['top'] if i else 0.
 for i,b in enumerate(kept):b['cap']=_overlay('difference',b['geometry'],kept[i+1]['geometry']) if i+1<len(kept) else b['geometry']
 return kept

def keep_tiers(tiers,maximum,rr=None,known=None,transform=None,base=None):
 """At most `maximum` tiers, chosen by the size of each one's own roof (the part no higher tier covers). The budget
 kept the lowest tiers and cut the rest off the top: two slices of one block's wall stayed while the block's set-back top
 storey went, and the building stood a storey short. A tier whose own roof is a sliver beside the tier above adds nothing
 anyone sees; dropped, its roof becomes part of the roof below it and the next tier stands on that one."""
 kept=[dict(t) for t in tiers]
 def own(i):return _overlay('difference',kept[i]['geometry'],kept[i+1]['geometry']).area if i+1<len(kept) else kept[i]['geometry'].area
 dropped=False
 while maximum is not None and len(kept)>max(0,int(maximum)):
  del kept[int(np.argmin([own(i) for i in range(len(kept))]))];dropped=True
 if dropped and rr is not None:return settle_tops(kept,rr,known,transform,base)
 for i,b in enumerate(kept):
  b['bottom']=kept[i-1]['top'] if i else 0.
  b['cap']=_overlay('difference',b['geometry'],kept[i+1]['geometry']) if i+1<len(kept) else b['geometry']
 return kept

def corner_radius(points,corner,ex,ey,top_z,reach=7.,radii=None,minimum_cut_m2=.4,empty_share=.15,max_wall_shift=None):
 """Radius of a rounded building corner measured in the returns: 0 when the corner is square or unmeasured.

 `corner` is the tier outline's corner and ex, ey unit vectors along its two walls into the building (all in the
 outline frame); `points` are returns (N, 3) in that frame. The walls' true lines are where this tier's roof returns
 stop beyond the corner zone (a raster edge can sit a cell off); the radius is the largest one whose cut-off area (the
 square corner less the quarter disc) holds almost no returns while the roof beside it is dense. A rounded 1980s office
 block's corners were drawn square: the returns showed a 2.5 m radius at every one. With `max_wall_shift`, a corner
 whose measured wall lines stand further than that from `corner` is unmeasured (0): the cut is made at `corner`, so the
 walls it rounds must be the ones the returns show."""
 import math
 radii=np.arange(.5,5.01,.25) if radii is None else radii
 rel=np.asarray(points,float)[:,:2]-corner;x=rel@ex;y=rel@ey;z=np.asarray(points,float)[:,2]
 m=(x>-2.)&(y>-2.)&(x<reach)&(y<reach)&(z>=top_z-1.5);x,y=x[m],y[m]
 if len(x)<40:return 0.
 fx,fy=x[(y>4.)&(y<reach)],y[(x>4.)&(x<reach)]
 if len(fx)<10 or len(fy)<10:return 0.
 sx,sy=float(np.quantile(fx,.02)),float(np.quantile(fy,.02))
 if max_wall_shift is not None and max(abs(sx),abs(sy))>max_wall_shift:return 0.
 x=x-sx;y=y-sy
 dens=((x>2.)&(x<reach)&(y>2.)&(y<reach)).sum()/max(1e-6,(reach-2.)**2)
 if dens<2.:return 0.
 # the outermost returns stand about half a return spacing inside the wall line (or the quantile's share of the strip)
 edge=max(.02*reach,.5/math.sqrt(dens));x=x+edge;y=y+edge
 best=0.
 for r in radii:
  area=(1-math.pi/4)*r*r
  if area<minimum_cut_m2:continue
  cut=(x>=0)&(y>=0)&(x<=r)&(y<=r)&((x-r)**2+(y-r)**2>r*r)
  if cut.sum()/area<empty_share*dens:best=float(r)
  else:break
 return best


def wall_corners(q,between_m=12.,long_m=3.,tolerance_m=1.5):
 """Convex right-angle corners of polygon q's exterior: a long edge (`long_m` or more) and
 the next long edge perpendicular to it, with what lies between (a chamfer, an arc of facets, small jogs) spanning at
 most `between_m` from the end of the one to the start of the other. The corner is the crossing of the two lines; ex,
 ey point along the walls into the building, L1, L2 are the walls' lengths and G1, G2 how far each stops short of the
 crossing: (corner, ex, ey, L1, L2, G1, G2)."""
 xy=np.asarray(q.simplify(.05,preserve_topology=True).exterior.coords)[:-1];n=len(xy)
 if n<3:return []
 ccw=q.exterior.is_ccw;d=[xy[(i+1)%n]-xy[i] for i in range(n)];L=[float(np.linalg.norm(v)) for v in d]
 longs=[i for i in range(n) if L[i]>=long_m]
 out=[]
 for k,i in enumerate(longs):
  u=d[i]/L[i];end=xy[(i+1)%n];cands=[]
  for s in range(1,len(longs)):
   j=longs[(k+s)%len(longs)]
   if j==i or float(np.linalg.norm(xy[j]-end))>between_m:break
   v=d[j]/L[j]
   if abs(float(u@v))>.2:continue
   cross=u[0]*v[1]-u[1]*v[0]
   if (cross>0)!=ccw:break                        # a concave turn: no convex corner here
   try:t,w=np.linalg.solve(np.array([u,-v]).T,xy[j]-xy[i])
   except np.linalg.LinAlgError:continue
   # the crossing stands past the end of the one wall and before the start of the other (a corner, not the far side of
   # a narrow wing)
   if t<L[i]-tolerance_m or -w<-tolerance_m:continue
   cands.append((j,xy[i]+u*t))
  if not cands:continue
  # the nearest such wall, or a longer piece of the same wall line further on (a jog broke it), not a parallel wall
  # standing further in
  j0=cands[0][0];v0=d[j0]/L[j0];n0=np.array([-v0[1],v0[0]])
  j,b=max((cc for cc in cands if abs(float((xy[cc[0]]-xy[j0])@n0))<=tolerance_m),key=lambda cc:L[cc[0]])
  # how far each wall stops short of the crossing (the chamfer and jogs run that far)
  out.append((b,-u,d[j]/L[j],L[i],L[j],max(0.,float(np.linalg.norm(b-end))),max(0.,float(np.linalg.norm(xy[j]-b)))))
 return out

def round_tier_corners(tiers,points,base,outer=None,minimum_radius_m=1.,maximum_share=.4,between_m=12.,margin_m=1.5,records=None):
 """Fillet the convex corners of tier outlines (in the outline frame) where the returns show a rounded corner, and the
 same corner of every tier standing flush under it, so a rounded block is rounded from its foot to its top. The cap of
 each tier is recomputed. `points` are returns (N, 3) in the frame.

 A corner is where two long walls (3 m or more) meet at a right angle, at the crossing of their lines, with at most
 `between_m` of short edges between them: the raster draws a rounded corner as a chamfer, and canopy and tree returns
 beside an entrance broke a rounded block's corner into 2-4 m jogs, so no single vertex was a right angle and the corner
 stood square. Within r + `margin_m` of the corner a flush tier is replaced by the rounded corner: the square less the
 quarter disc is cut, the chamfer and jogs are filled out to the wall lines and whatever stands up to the margin outside
 them goes.
 A tier is flush at a corner when nothing of it stands beyond the wall lines by more than the margin within r + margin
 of the corner: a lower terrace running on past the corner keeps its wall, while an entrance canopy's jog 7 m along the
 wall does not stop the block above it being rounded (it did: the tier below was rounded alone and stood open).

 Corners on the outer boundary (`outer`, a line) are rounded too: a block standing flush on its lot outline kept square
 corners there. Along the outline the outline's own wall draws the face (facades.compile), so each such corner is
 returned in `records` (dicts: corner, ex, ey, radiusM, marginM, bottomZ = base of the lowest tier cut, outerAxes = which
 of ex/ey runs along the outline, cut = the cut polygon): the outline wall stops at bottomZ over the arc's span and the
 tier draws the arc."""
 from shapely.geometry import Point,Polygon,MultiPolygon
 import math
 def quadrant(b,ex,ey,lo,hi):return Polygon([b+ex*lo+ey*lo,b+ex*hi+ey*lo,b+ex*hi+ey*hi,b+ex*lo+ey*hi])
 cuts=[]
 for t in tiers:
  top=base+t['top']
  for q in _parts(t['geometry']):
   for b,ex,ey,L1,L2,G1,G2 in wall_corners(q,between_m):
    r=corner_radius(points,b,ex,ey,top,max_wall_shift=margin_m)
    r=min(r,maximum_share*min(L1,L2))
    if r<minimum_radius_m:continue
    disc=Point(*(b+(ex+ey)*r)).buffer(r,quad_segs=6)
    same=[k for k,cc in enumerate(cuts) if float(np.hypot(*(np.asarray(cc['corner'])-b)))<max(r,cc['radiusM'])+margin_m and max(float(cc['ex']@ex+cc['ey']@ey),float(cc['ex']@ey+cc['ey']@ex))>1.8]   # one corner seen from several tiers: one cut
    # tiers run bottom to top: a corner several tiers show is measured on the highest (canopies and trees stand lower)
    if same:cuts.pop(same[0])
    # within R of the corner a flush tier becomes the rounded corner itself: straight walls to the arc, the arc, nothing
    # outside the wall lines (a chamfer already inside the arc was left as it was, with the canopy's jogs)
    R=min(max(r+margin_m,G1+.1,G2+.1),min(L1,L2))
    rel=np.asarray(points,float)[:,:2]-b;u_,v_=rel@ex,rel@ey;hi=np.asarray(points,float)[:,2]>=top-1.5
    ref=(hi&(u_>R)&(u_<R+4.)&(v_>R)&(v_<R+4.)).sum()/16.
    inner=hi&(u_>=0)&(v_>=0)&(u_<=R)&(v_<=R)&~((u_<r)&(v_<r)&((u_-r)**2+(v_-r)**2>r*r))
    supported=bool(ref>0 and inner.sum()/max(1e-6,R*R-(1-math.pi/4)*r*r)>=.5*ref)
    cuts.append({'corner':b,'ex':ex,'ey':ey,'radiusM':float(r),'marginM':margin_m,'cut':quadrant(b,ex,ey,-margin_m,r).difference(disc),
                 'supported':supported,'box':quadrant(b,ex,ey,-margin_m,R),'fill':quadrant(b,ex,ey,0.,R).difference(quadrant(b,ex,ey,0.,r).difference(disc))})
 if not cuts:return tiers,[]
 def polygons(g):
  q=_parts(g);return Polygon() if not q else q[0] if len(q)==1 else MultiPolygon(q)
 def flush(g,cc):
  b,ex,ey,r,m=cc['corner'],cc['ex'],cc['ey'],cc['radiusM'],cc['marginM']
  if g.boundary.distance(Point(*b))>=r+.3:return False      # a chamfered corner stands up to r inside the crossing
  # the rounded corner is roof in the returns at the measured height (a crossing out beyond a diagonal wall is not)
  if not cc['supported']:return False
  # near the corner itself (a jog further along a wall is not a terrace running on past the corner)
  beyond=Point(*b).buffer(r+m).difference(quadrant(b,ex,ey,-m,r+m))
  return _overlay('intersection',g,beyond).area<.5
 out=[];bottoms=[None]*len(cuts)
 for t in tiers:
  g=t['geometry']
  for k,cc in enumerate(cuts):
   if flush(g,cc):
    g=_overlay('union',_overlay('difference',g,cc['box']),cc['fill'])
    g=polygons(g.buffer(-.02,join_style=2).buffer(.02,join_style=2))   # the fill's edges on the wall lines leave zero-width spurs
    z=base+t['bottom'];bottoms[k]=z if bottoms[k] is None else min(bottoms[k],z)
  out.append({**t,'geometry':g})
 for i,t in enumerate(out):t['cap']=polygons(_overlay('difference',t['geometry'],out[i+1]['geometry'])) if i+1<len(out) else t['geometry']
 if records is not None and outer is not None:
  for cc,z in zip(cuts,bottoms):
   b,m=cc['corner'],cc['marginM']
   if z is None or outer.distance(Point(*b))>=m+.5:continue
   axes=[ax for ax in ('ex','ey') if outer.distance(Point(*(b+cc[ax]*(cc['radiusM']+1.))))<m+.5]
   if axes:records.append({**cc,'bottomZ':float(z),'outerAxes':axes})
 return out,[cc['corner'] for cc in cuts]

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


def rooftop_panels(points,surface_mask,rr,xmin,ymax,step,context=None,minimum_returns=40,minimum_length_m=8.,maximum_width_m=2.,minimum_height_m=3.,minimum_face_m=2.,maximum_face_m=8.,minimum_density=2.,beside_occupancy=.35,above_density=.5,volume_beside=.25,audit=None):
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
   # A sign face stands in air: beside it, at its own height, there is nothing. A tree crown overhanging the roof edge
   # (this LiDAR does not class vegetation) or any other volume fills that band on one side or both; the crowns of a
   # street's trees over a row of houses read as 10-30 m long, 1-2 m wide 'panels' and were drawn as black slabs.
   level=inside&(np.abs(across)>1.0)&(np.abs(across)<=6.0)&(context[:,2]>=bottom+1.0)&(context[:,2]<=top-.5)
   sides=[set(zip(np.floor(along[level&(across*sg>0)]).astype(int).tolist(),np.floor(across[level&(across*sg>0)]).astype(int).tolist())) for sg in (1,-1)]
   facts.update(volumeBeside=max(len(c) for c in sides)/(5.0*max(length,1e-9)))
   if facts['volumeBeside']>volume_beside:note(rejected='volume-beside',**facts);continue
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
