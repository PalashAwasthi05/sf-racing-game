"""Compile observation-derived wall regions and window topology into metric geometry.
Profiles are data; no geographic IDs or addresses are interpreted here.
"""
import math
import numpy as np
from shapely.geometry import Point,Polygon,box
from shapely.ops import unary_union
from .surface_evidence import admissible,resolve_material
from . import bays, facade_detail, facade_wall_domain, facade_bands, facade_recesses, door_leaves


RIBBON_PIER_M=.3   # lights of a bank standing at most this far apart are one band of glazing per storey


# the sill under every window: its projection from the wall face (game_depth.py deepens it)
from .game_depth import SILL_PROJECTION_M as SILL

def measured_bay_extent(record, width, frame_height):
 """Metric extent of a directly measured bay in its source facade frame.

 A bay's photographed top is independent evidence for that projecting volume;
 a lower LiDAR roof return on the host mass must not truncate its upper windows.
 The host wall itself remains limited by its own roof/domain boundary.
 """
 x0,z0,x1,z1=map(float,record['rect'])
 return {'x0':x0*width,'x1':x1*width,'z0':z0*frame_height,'z1':z1*frame_height}

def ribbon_bank(group,W,region,dx,nx):
 """Whether a repeat bank is ribbon glazing: its lights stand at most RIBBON_PIER_M apart, so what divides them is a
 mullion, not wall. Drawn light by light, each with its own four-sided frame and sill and a sliver of wall between, a
 tile office block's continuous bands of glass came out as a grid of punched windows. A style may say so itself."""
 st=group.get('style') or {}
 if 'ribbon' in st:return bool(st['ribbon'])
 # arched, pointed, or segmental heads are separate openings however close (a pair of wide arches is not a band), and a band
 # needs three lights at least
 if st.get('kind','window')!='window' or head_shape(st) is not None or W is None or nx<3:return False
 return dx*(region[2]-region[0])*W*(1-st.get('openingRatio',.7))<=RIBBON_PIER_M

def _ribbons(ops,W):
 """Each storey of a ribbon bank's lights, consecutive columns joined into one opening: the joints between lights
 become mullions (`jointFractions`, `jointWidthM` the width the lights stood apart), the lights' own mullions stay."""
 out=[];rows={}
 for op in ops:rows.setdefault((op['group'],op['iy']),[]).append(op)
 for key in sorted(rows):
  row=sorted(rows[key],key=lambda o:o['ix']);runs=[[row[0]]]
  for o in row[1:]:
   if o['ix']==runs[-1][-1]['ix']+1:runs[-1].append(o)
   else:runs.append([o])
  for run in runs:
   x0=run[0]['rect'][0];x1=run[-1]['rect'][2];w=x1-x0;st=dict(run[0]['style'])
   if len(run)==1 or w<=0:out.append(run[0]);continue
   gap=float(np.median([b['rect'][0]-a['rect'][2] for a,b in zip(run,run[1:])]))*(W or 1.)
   own=st.get('mullionFractions',[])
   st.update(ribbon=True,jointWidthM=round(float(np.clip(gap,.05,.12)),3),
             jointFractions=[round(((a['rect'][2]+b['rect'][0])/2-x0)/w,5) for a,b in zip(run,run[1:])],
             mullionFractions=[round((o['rect'][0]+f*(o['rect'][2]-o['rect'][0])-x0)/w,5) for o in run for f in own])
   out.append({**run[0],'rect':[x0,min(o['rect'][1] for o in run),x1,max(o['rect'][3] for o in run)],'style':st,'lights':len(run)})
 return out

def _grid(g,W=None,H=None):
 """Cell centres and pitch of a repeat bank, by count or by pitch in metres."""
 region=g['region'];nx=g.get('columns',1);ny=g.get('rows',1)
 if W is not None and g.get('columnPitchM'):nx=max(1,int(round((region[2]-region[0])*W/g['columnPitchM'])))
 if H is not None and g.get('rowPitchM'):ny=max(1,int(round((region[3]-region[1])*H/g['rowPitchM'])))
 xs=g.get('xCenters',[(i+.5)/nx for i in range(nx)]);ys=g.get('yCenters',[(i+.5)/ny for i in range(ny)]);dx=g.get('spacingX',1/nx);dy=g.get('spacingY',1/ny)
 return region,nx,ny,xs,ys,dx,dy

def head_shape(style,width=1.,height=1.):
 """Outline of an opening in its own unit square: flat, a semicircular arch, or a shallow segmental arch.

 `head` names the aperture shape; `springRatio` (arch/pointed) is the height of the springline as a fraction of the opening, and
 `riseRatio` (segmental) the rise of the shallow arc. Photographs show the aperture head directly; decorative surrounds
 and hoods are independent. The pointed head is a two-sided Gothic curve meeting at the center peak. The grammar
 fills the corners above the springline with wall and runs the frame along the arc.
 """
 head=style.get('head','flat')
 if head=='arch':
  spring=style.get('springRatio',.68);arc=[[.5+.5*math.cos(t),spring+(1-spring)*math.sin(t)] for t in np.linspace(0,math.pi,17)]
  return [[0,0],[1,0],[1,spring]]+arc+[[0,0]]
 if head=='pointed':
  spring=float(np.clip(style.get('springRatio',.68),.2,.95))
  # Two quadratic curves meet with opposing tangents at a pointed crown.
  # Keep the spring endpoints and crown exact; the control point lies at
  # the opposite spring point, producing a simple Gothic/lancet outline.
  ts=np.linspace(0.,1.,10)
  left=[[.5*t, spring+(1.-spring)*t*t] for t in ts]
  right=[[1.-.5*t, spring+(1.-spring)*t*t] for t in ts]
  return [[0,0],[1,0],[1,spring]]+right[1:]+left[-2::-1]+[[0,0]]
 if head=='segmental':
  rise=float(style.get('riseRatio',.14));sag=rise*height;half=width/2
  # A segmental head is part of a circle in metres, not an ellipse in the
  # opening's unit square. An ellipse has vertical tangents at the spring
  # points and makes shallow heads bulge sharply above their jambs.
  if 0<sag<=half and width>0 and height>0:
   radius=(half*half+sag*sag)/(2*sag)
   angle=math.asin(min(1.,half/radius))
   steps=max(12,int(math.ceil(2*angle/max(.01,2*math.acos(max(-1.,1-.002/radius))))))
   steps=min(96,steps+(steps%2))
   arc=[[.5+radius*math.sin(t)/width,1-radius/height+radius*math.cos(t)/height]
        for t in np.linspace(angle,-angle,steps+1)]
  else:
   # Legacy observations occasionally label a tall elliptical head as
   # segmental. Preserve their measured bounds instead of shortening them.
   arc=[[.5+.5*math.cos(t),(1-rise)+rise*math.sin(t)] for t in np.linspace(0,math.pi,17)]
  arc[0]=[1,1-rise];arc[-1]=[0,1-rise]
  return [[0,0],[1,0],[1,1-rise]]+arc+[[0,0]]
 return None


def opening_outline(op,W,H):
 """Return an opening's original outline in facade metres.

 The outline is made before a wall segment clips the opening.  This matters for
 arches and for explicit polygons: rebuilding a head from the clipped bounding
 box changes its centre and curvature at a segment boundary.
 """
 if op.get('polygon') is not None:
  points=np.asarray(op['polygon'],float)
  if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
   raise ValueError('opening polygon must contain finite [x,z] points')
  return Polygon([(float(x)*W,float(z)*H) for x,z in points])
 if 'rect' not in op:raise ValueError('opening needs rect or polygon')
 x1,y1,x2,y2=map(float,op['rect']);w=x2-x1;h=y2-y1
 if w<=0 or h<=0:raise ValueError('opening rect must have positive extent')
 shape=head_shape(op.get('style',{}),w*W,h*H)
 if shape is None:return box(x1*W,y1*H,x2*W,y2*H)
 return Polygon([(x1*W+a*w*W,y1*H+b*h*H) for a,b in shape])


def _component_material(item,fallback):
 """Resolve an evidence-backed component while preserving direct style keys."""
 if item.get('materialEvidence'):
  return resolve_material({**item,'material':item.get('material',fallback)})
 return item.get('material',fallback)


def _module_evidence_panels(profile, panels, width_m, height_m):
 """Return non-overlapping typed wall panels for a measured module zone.

 The source compiler stores zone bounds in facade metres. Existing panels own
 their pixels first; the measured repeat is assigned only to the remaining
 cells inside that zone. Wall cells later union equal material regions, so
 these records do not create a second wall layer.
 """
 evidence=profile.get('moduleEvidence') or {}
 if evidence.get('status')!='paired-semantic-review-accepted':return []
 if evidence.get('pattern')!='horizontal_periodic_boards' or evidence.get('axis')!='horizontal_courses':return []
 if (profile.get('materialEvidence') or {}).get('family')!='timber':return []
 material=str(profile.get('material',''))
 base,_,color=material.partition('#')
 evmat=evidence.get('material') or {}
 color_hex='#'+color
 color_rgb=(profile.get('materialEvidence') or {}).get('colorRgb')
 expected='#'+''.join(f'{int(c):02x}' for c in color_rgb) if color_rgb and len(color_rgb)==3 else None
 if (base!='timber_dark' or not color or color_hex.lower()!=str(evmat.get('hex','')).lower()
     or color_hex.lower()!=str(expected).lower()):return []
 zone=evidence.get('sourceZoneM') or {}
 try:x0,z0,x1,z1=(float(zone[k]) for k in ('x0','z0','x1','z1'))
 except (KeyError,TypeError,ValueError):return []
 try:pitch=float(evidence.get('pitchM'))
 except (TypeError,ValueError):return []
 if (not np.isfinite([x0,z0,x1,z1,width_m,height_m,pitch]).all() or width_m<=0 or height_m<=0
     or pitch<=0 or x1<=x0 or z1<=z0):return []
 x0,x1=max(0.,x0),min(float(width_m),x1)
 z0,z1=max(-.3*height_m,z0),min(1.2*height_m,z1)
 if x1<=x0 or z1<=z0:return []
 # Partition on every existing panel edge. The center test assigns each
 # rectangle to exactly one owner and avoids overwriting any other zone.
 xs={x0,x1};zs={z0,z1}
 existing=[]
 for panel in panels:
  rect=panel.get('rect')
  if not isinstance(rect,(list,tuple)) or len(rect)!=4:continue
  try:a,b,c,d=(float(v) for v in rect)
  except (TypeError,ValueError):continue
  a,c=a*width_m,c*width_m;b,d=b*height_m,d*height_m
  if c<=x0 or a>=x1 or d<=z0 or b>=z1:continue
  existing.append((a,b,c,d));xs.update((max(x0,a),min(x1,c)));zs.update((max(z0,b),min(z1,d)))
 from pipeline import material_recipes
 typed=material_recipes.make_observed_key(base,color_hex.lower(),family='timber',
         pattern='horizontal_courses',moduleM={'z':float(pitch)})
 out=[];xx=sorted(xs);zz=sorted(zs)
 for a,c in zip(xx,xx[1:]):
  if c-a<1e-8:continue
  for b,d in zip(zz,zz[1:]):
   if d-b<1e-8:continue
   mx,mz=(a+c)/2,(b+d)/2
   if any(e0<=mx<=e1 and f0<=mz<=f1 for e0,f0,e1,f1 in existing):continue
   out.append({'rect':[a/width_m,b/height_m,c/width_m,d/height_m],'material':typed})
 return out

def openings(profile,fits,W=None,H=None,wallH=None):
 """Opening rectangles of a profile, as fractions of the frontage W x H.

 A repeat bank may give its pitch in metres (`rowPitchM`, `columnPitchM`) instead of a count: a storey height
 or bay width is what a frontage photo shows, and the number of rows then follows the height the pipeline
 measured for each wall piece, so one profile serves a podium and the tower above it. A bank may also be
 limited to pieces of a height range (`minHeightM`, `maxHeightM`): a ground-floor shopfront bank exists only
 on the pieces that reach the ground floor, a tower window bank only on the tall ones. The range is tested
 against the wall piece's own height (`wallH`), while the metres of pitches and regions refer to the profile's
 frame height H: a frame of 50 m read on a tower and its 24 m wing must not switch the wing's bank off.
 """
 result=[];ribbons=set()
 for gi,group in enumerate(profile.get('groups',[])):
  g=dict(group);fit=next((f for f in fits if f['id']==g.get('fitId')),None)
  gate=wallH if wallH is not None else H
  if gate is not None and not (g.get('minHeightM',0)<=gate<=g.get('maxHeightM',float('inf'))):continue
  if fit:
   region=g.get('region',fit['region']);style=dict(fit['style']);style.update(g.get('style',{}));xs=fit['columns']['centers'];ys=fit['rows']['centers'];dx=fit['columns']['spacing'];dy=fit['rows']['spacing'];cells={(round(c['x'],5),round(c['y'],5)):c['evidence'] for c in fit['cells']}
  else:
   region,nx,ny,xs,ys,dx,dy=_grid(g,W,H);style=g.get('style',{});cells={}
   if ribbon_bank(g,W,region,dx,nx):ribbons.add(gi)
  x0,y0,x1,y1=region
  for ix,x in enumerate(xs):
   for iy,y in enumerate(ys):
    if [ix,iy] in g.get('omit',[]):continue
    ow=dx*(x1-x0)*style.get('openingRatio',.7);oh=dy*(y1-y0)*style.get('openingHeightRatio',.7)
    cx=x0+x*(x1-x0);cy=y0+y*(y1-y0)
    result.append(dict(rect=[cx-ow/2,cy-oh/2,cx+ow/2,cy+oh/2],style=style,evidence=cells.get((round(x,5),round(y,5)),'visual-grammar' if profile.get('sourceIds') else 'inferred'),ix=ix,iy=iy,group=gi,fitId=g.get('fitId')))
 result=[op for op in result if admissible(op,profile)]
 if ribbons:result=[op for op in result if op['group'] not in ribbons]+_ribbons([op for op in result if op['group'] in ribbons],W)
 for opening in profile.get('openings',[]):
  if admissible(opening,profile,explicit=True):result.append(dict(opening,evidence=opening.get('evidence','observed-boundary'),ix=0,iy=0,sourceIndex=opening.get('sourceIndex')))
 return result


def facade_prism(poly, origin, tangent, normal, thickness):
 """Extrude an elevation behind its front plane, preserving outward winding.

 The elevation-to-world frame is a reflection: x maps along the facade,
 y maps up, and extrusion z maps inward. Apply that transform explicitly so
 winding reverses with it. Guessing orientation afterwards with fix_normals
 fails for valid elevations whose aperture touches an exterior edge at one
 point: their extrusion has a shared four-face edge, not a manifold volume.
 """
 import trimesh
 transform=np.eye(4)
 transform[:3,0]=[*np.asarray(tangent,float)[:2],0.]
 transform[:3,1]=[0.,0.,1.]
 transform[:3,2]=[*-np.asarray(normal,float)[:2],0.]
 transform[:3,3]=origin
 mesh=trimesh.creation.extrude_polygon(poly,height=thickness,engine='earcut')
 mesh.apply_transform(transform)
 return mesh


def compile_wall(ctx,c,a,b,z0,H,profile,fits):
 """Anchor one grammar to the whole frontage, then clip it to this wall segment."""
 n=np.asarray(profile['normal'],float);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]])
 g=c['g'];coords=np.concatenate([np.array(p.exterior.coords) for p in ctx['polys'](g)])
 # Use long, parallel frontage edges as the reference plane; ignore minor projections.
 edges=[]
 for p in ctx['polys'](g):
  ring=np.array(p.exterior.coords)
  for pa,pb in zip(ring,ring[1:]):
   v=pb-pa;length=np.linalg.norm(v)
   if length>2 and abs(np.dot(v/length,u))>.96:edges.append((length,float(((pa+pb)/2)@n),pa,pb))
 if not edges:return None
 outward=max(d for _,d,_,_ in edges);selected=[e for e in edges if e[1]>outward-profile.get('planeToleranceM',2.5)]
 allxy=np.concatenate([np.array([e[2],e[3]]) for e in selected]);left=float(np.min(allxy@u));right=float(np.max(allxy@u));W=right-left
 # A profile read from a rectified photograph of one wall carries that image's metric frame along the wall (its
 # u-range); its openings land where the photograph shows them however the outline is partitioned into pieces.
 if profile.get('frameU'):left,right=map(float,profile['frameU']);W=right-left
 mid=(a+b)/2
 # A reading measured on its own plane (planeV, along its own normal) draws the walls standing in that plane, whether or
 # not a wall of the same facing stands further out: a recessed face read square-on was dropped for not being the
 # facing's outermost wall (37 of 171 photographed ground-level readings never reached their wall).
 ref=float(profile['planeV']) if profile.get('planeV') is not None else outward
 if float(mid@n)<ref-profile.get('planeToleranceM',2.5):return None
 low,high=sorted([float(a@u-left),float(b@u-left)])
 if W<3 or high-low<.3:return None
 # A photographed frame says nothing about a piece of the same facing outside it: that piece takes the generic
 # facade (in the building's measured style) instead of an empty wall.
 if profile.get('frameU') and (high<=.5 or low>=W-.5):return None
 plane=float(mid@n);angle=math.atan2(u[1],u[0]);layer='buildings__'+c['id']+'__observed_'+profile['id'];detail='facades__'+c['id']+'__observed_'+profile['id'];cube=ctx['cube'];mat=resolve_material(profile)
 # Vertical frame of the profile. Fractions in a hand-annotated profile refer to the wall the pipeline built
 # (H). A profile read from a rectified photograph carries the height that image was drawn with
 # (`heightRefM`): its fractions are metres above the base, and they must land at those metres whatever the
 # LiDAR wall level came out as. Wall extent stays [0, H]; anything above the wall is clipped away.
 Hs=float(profile.get('heightRefM') or H)
 def block(x,z,w,h,d,material,depth=0,key=detail):
  if min(w,h,d)<=.002:return
  p=u*(left+x)+n*(plane+depth);cube([*p,z0+z],[w,d,h],key,material,angle)
 def point(x,z,depth=0):return np.r_[u*(left+x)+n*(plane+depth),z0+z]
 # A profile's bay band follows the same source plane as the measured bay
 # shell. Wall-edge partitions may put this call on a front/return facet;
 # using that facet's plane here would split the cap into disconnected pieces.
 def source_point(x,z,depth=0):return np.r_[u*(left+x)+n*(ref+depth),z0+z]
 elev=ctx.get('elev')
 def ground(x,depth=0.):
  """The terrain under a point of this wall's frame, in metres above the wall's foot (what stands in front of a wall on a
  hillside stands on its own ground)."""
  if elev is None:return 0.
  p=point(x,0.,depth);return float(elev(p[0],p[1]))-z0
 def points(x,z,depth):
  """point() for arrays of wall-frame coordinates: (n, 3) world points."""
  x,z,d=(np.asarray(v,float) for v in (x,z,depth));return np.c_[np.outer(left+x,u)+np.outer(plane+d,n),z0+z]
 def ground_v(x,depth):
  """ground() for arrays."""
  x=np.asarray(x,float)
  if elev is None:return np.zeros(len(x))
  p=points(x,np.zeros(len(x)),depth);return np.asarray(elev(p[:,0],p[:,1]),float)-z0
 def prism(poly,depth,thickness,material,key=layer):
  """A wall piece of any outline in the facade plane (x right, z up), `thickness` deep behind `depth`."""
  import trimesh
  from shapely.geometry import Polygon as _P
  for q in ([poly] if poly.geom_type=='Polygon' else list(getattr(poly,'geoms',[]))):
   if q.is_empty or q.area<.001:continue
   mesh=facade_prism(q,point(0.,0.,depth),u,n,thickness);ctx['put'](mesh,key,material)
 def sloped_prism(poly,depth,slope,thickness,material,key=layer):
  """Relief on a straight segment of a swept band, using the same affine frame."""
  for q in ([poly] if poly.geom_type=='Polygon' else list(getattr(poly,'geoms',[]))):
   if q.is_empty or q.area<.0001:continue
   mesh=facade_prism(q,point(0.,0.,depth),u+n*slope,n,thickness);ctx['put'](mesh,key,material)
 def beam(a,b,thickness,material,key=detail):
  """A square member between two facade points (x,z,depth); frames follow an arch with these."""
  import trimesh
  aa=point(*a);bb=point(*b);length=np.linalg.norm(bb-aa)
  if length<.002:return
  mesh=trimesh.creation.box([thickness,thickness,length]);T=trimesh.geometry.align_vectors([0,0,1],(bb-aa)/length);T[:3,3]=(aa+bb)/2;mesh.apply_transform(T);ctx['put'](mesh,key,material)
 ops=[]
 # A facade read below its wall's foot (negative z): the street in front lies lower than the building's ground (a garage
 # door on a hill). The wall runs down to the lowest opening read, at most 30 % of the frame below the foot, standing
 # just proud of the foundation there.
 # The facade also runs down to the ground in front of it where that lies below the wall's foot (the downhill end of a
 # hillside wall), in its lowest colour zone, instead of leaving the foundation's plinth showing under it.
 foot=min([0.]+[ground(x,.3) for x in np.linspace(low,high,max(2,int((high-low)/1.5)+1))])-.05
 opening_feet=[]
 for op in openings(profile,fits,W,Hs,H):
  try:
   outline=opening_outline(op,W,Hs)
   if not outline.is_empty and outline.is_valid:opening_feet.append(float(outline.bounds[1]))
  except (TypeError,ValueError):continue
 zlo=max(-max(4.,.4*Hs),min([0.,foot]+opening_feet))
 roof_wall_domain=None
 boundary_roofline_parapets=set()
 roof_domain_unresolved=[]
 shell_chains=(ctx.get('wallTopBoundaryChainsByProfile') or {}).get(profile.get('id'))
 if shell_chains is not None:
  roof_wall_domain=facade_wall_domain.project_profile_chain_domain(
      shell_chains,profile,(low,high),z0,plane,a,b,domain_base_z=zlo,plane_tolerance=.1)
  if roof_wall_domain is not None:
   observed_domain,boundary_roofline_parapets,roof_domain_unresolved=facade_wall_domain.observed_gable_domain(
       profile,(low,high),W,Hs,zlo,roof_wall_domain)
   if observed_domain is not None:roof_wall_domain=observed_domain
   H=max(H,float(roof_wall_domain.bounds[3]))
 variable_roof_top=False
 if roof_wall_domain is not None:
  # The local wall polygon closes to its foot, so ignore those closure vertices
  # when deciding whether a generic horizontal coping belongs at the top. A
  # sloped/gabled shell boundary has no single ridge-height eave to cap.
  top_z=[float(z) for _,z in roof_wall_domain.exterior.coords if z>zlo+.05]
  variable_roof_top=len(top_z)>1 and max(top_z)-min(top_z)>.2
 clip_box=roof_wall_domain if roof_wall_domain is not None else box(low,zlo,high,H)
 for op in openings(profile,fits,W,Hs,H):
  # Build the complete observed outline first.  Recomputing an arch from a
  # segment-clipped bounding box moves its centre and changes the head shape.
  try: original=opening_outline(op,W,Hs)
  except (TypeError,ValueError): continue
  if original.is_empty or not original.is_valid:continue
  outline=original.intersection(clip_box)
  pieces=[outline] if outline.geom_type=='Polygon' else list(getattr(outline,'geoms',[]))
  for piece in pieces:
   if piece.is_empty or piece.area<.001:continue
   x1,y1,x2,y2=piece.bounds
   if x2-x1<.12 or y2-y1<.12:continue
   ops.append({**op,'rect':[x1,y1,x2,y2],'_outline':piece,'_originalOutline':original,
               '_contoured':op.get('polygon') is not None or head_shape(op.get('style',{})) is not None or not original.equals(piece)})
 # A source can establish cladding continuing to the wall outline even when
 # the image crop excludes its apex. Keep that semantic extent distinct from
 # a falsely measured top height; the validated wall domain owns the outline.
 panels=[{**p,'rect':[p['rect'][0],p['rect'][1],p['rect'][2],H/Hs]} if p.get('toWallTop') is True else p
         for p in profile.get('panels',[])]
 panels.extend(_module_evidence_panels(profile,panels,W,Hs))
 def wall_mat(x,z):
  z=max(z,.05)    # a wall below its foot (a street lower than the building's ground) continues the lowest zone read
  found=[p for p in panels if p.get('rect') and p['rect'][0]*W<=x<=p['rect'][2]*W and p['rect'][1]*Hs<=z<=p['rect'][3]*Hs]
  return _component_material(found[-1],mat) if found else mat
 # A measured bay is one assembly in the source wall frame. Footprint edges
 # often already trace its front and returns, leaving no host edge under its
 # centre. Emit it once from any owned host segment, independently of those
 # partitions, and never add the footprint's projection to the measured depth.
 bay_list=[]
 for bay_index,bb in enumerate(profile.get('bays',[])):
  extent=measured_bay_extent(bb,W,Hs);bx0,bz0,bx1,bz1=extent['x0'],extent['z0'],extent['x1'],extent['z1']
  if bx1-bx0>=.8 and bz1-bz0>=1.:bay_list.append({'index':bay_index,'x0':bx0,'x1':bx1,'z0':bz0,'z1':bz1,'hostTopM':H,'hostMaterial':mat,'p':float(np.clip(bb.get('projectionM',.6),.25,1.2)),'shape':bb.get('shape'),'sideAngleDeg':bb.get('sideAngleDeg',45),
                                       'trim':bb.get('trim'),'cladding':bb.get('cladding'),
                                       'claddingZones':[{**zone,'rect':[zone['rect'][0]*W,zone['rect'][1]*Hs,zone['rect'][2]*W,zone['rect'][3]*Hs]} for zone in bb.get('claddingZones',[])],
                                       'storeys':[z*Hs for z in bb.get('storeys',[])],'corbel':bool(bb.get('corbel')),'cap':bb.get('cap','cornice')})
 # A full-width measured band that occupies the decorative bay cornice is
 # the observed trim. Keep the bay's structural lid, but do not add the
 # renderer's inferred decorative layers on top of the source detail.
 for _bay in bay_list:
  if facade_bands.bay_cap_replaced_by_band(_bay,profile.get('bands',[]),W,Hs):
   _bay['cap']='none'
 bay_windows=0
 if bay_list:
  in_bay=lambda r,b:b['x0']-.1<=(r[0]+r[2])/2<=b['x1']+.1 and b['z0']-.1<=(r[1]+r[3])/2<=b['z1']+.1
  ops=[op for op in ops if not any(in_bay(op['_originalOutline'].bounds,b) for b in bay_list)]
  whole=[]
  for op in openings(profile,fits,W,Hs,H):
   try:o=opening_outline(op,W,Hs)
   except (TypeError,ValueError):continue
   if not o.is_empty and o.is_valid:whole.append({**op,'rect':list(o.bounds),'_outline':o})
  emitted=ctx.setdefault('observedBayAssemblies',set())
  bay_plane=ref if profile.get('planeV') is not None else plane
  def bay_point(x,z,depth=0.):return np.r_[u*(left+x)+n*(bay_plane+depth),z0+z]
  for b in bay_list:
   identity=(c.get('sourceId',c['id']),profile['id'],b['index'],round(z0,4))
   if identity in emitted:continue
   mine=[op for op in whole if in_bay(op['rect'],b)]
   trim=b.get('trim') or next((op['style'].get('frameMaterial') for op in mine if op.get('style',{}).get('frameMaterial')),'#ece9e2')
   if ctx.get('_deferObservedBayAssembly'):
    staged=[]
    bay_windows+=bays.draw(bay_point,lambda mesh,key,material:staged.append((mesh,key,material)),layer,detail,b,mine,b.get('cladding') or wall_mat((b['x0']+b['x1'])/2,(b['z0']+b['z1'])/2),trim,claddingZones=b.get('claddingZones'))
    shell=[mesh for mesh,key,_material in staged if key==layer]
    valid_shell=bool(shell) and all(len(mesh.faces)>0 and len(mesh.vertices)>0
                                    and np.isfinite(mesh.vertices).all() and mesh.is_watertight
                                    for mesh in shell)
    valid_all=all(len(mesh.faces)>0 and len(mesh.vertices)>0 and np.isfinite(mesh.vertices).all()
                  for mesh,_key,_material in staged)
    if valid_shell and valid_all:
     from .bay_replacement import replacement_record
     ctx.setdefault('_stagedObservedBayMeshes',[]).extend(staged)
     shape, support=bays.corners(b['x0'],b['x1'],b['p'],b.get('shape'),b.get('sideAngleDeg',45.))
     ctx.setdefault('_observedBayReplacements',[]).append(replacement_record(
      origin=u*left+n*bay_plane,tangent=u,normal=n,plane=bay_plane,zBase=z0,
      xRange=(b['x0'],b['x1']),zRange=(b['z0'],b['z1']+.34),support=support,
      sourceId=identity[0],profileId=profile['id'],bayIndex=b['index']))
    else:
     for mesh,key,material in staged:ctx['put'](mesh,key,material)
   else:
    bay_windows+=bays.draw(bay_point,ctx['put'],layer,detail,b,mine,b.get('cladding') or wall_mat((b['x0']+b['x1'])/2,(b['z0']+b['z1'])/2),trim,claddingZones=b.get('claddingZones'))
   emitted.add(identity)
 # Recesses are explicit negative volumes. A projecting bay owns its own
 # facade and openings, so the host-wall recess builder skips any overlap.
 recess_regions=facade_recesses.profile_regions(profile,W,Hs,H,low,high,clip_box,bay_list)
 if recess_regions:
  # Openings described inside a recessed zone belong on its back plane. The
  # recession's own backOpenings list supplies that geometry; never leave a
  # second pane or frame on the outer facade plane.
  ops=[op for op in ops if not any(box(*r['bounds']).covers(box(*op['rect']).centroid)
                                    for r in recess_regions)]
 # Partition each horizontal slice around its openings and material boundaries,
 # then union equal-material/equal-depth cells before extrusion. Emitting every
 # rectangle as a closed cuboid leaves coplanar internal faces for AO to outline.
 zs=sorted(set([zlo,0.,H]+[v for op in ops for v in [op['rect'][1],op['rect'][3]]]+[min(H,p['rect'][k]*Hs) for p in panels for k in [1,3]]))
 wall_cells={}
 for bottom,top in zip(zs,zs[1:]):
  if top-bottom<.003:continue
  cy=(bottom+top)/2;active=[o['rect'] for o in ops if o['rect'][1]<cy<o['rect'][3]]
  xs=sorted(set([low,high]+[v for r in active for v in [r[0],r[2]]]+[np.clip(p['rect'][k]*W,low,high) for p in panels for k in [0,2]]))
  for x1,x2 in zip(xs,xs[1:]):
   cx=(x1+x2)/2
   if any(r[0]<cx<r[2] for r in active):continue
   material=wall_mat(cx,cy);front=.012 if top<=0 else 0.
   cell=box(x1,bottom,x2,top).intersection(clip_box)
   if not cell.is_empty and cell.area>.001:wall_cells.setdefault((material,front),[]).append(cell)
 for (material,front),cells in wall_cells.items():
  merged=unary_union(cells)
  merged=facade_recesses.remove_host_wall(merged,recess_regions)
  if profile.get('planeV') is not None and plane>ref+.1:
   # The mapped outer face is an approximate bay-front boundary. It cannot
   # remain an opaque sheet over the independently measured bay assembly.
   # Remove only its overlapping elevation, retaining all surrounding wall.
   for b in bay_list:
    merged=merged.difference(box(b['x0'],b['z0'],b['x1'],b['z1']))
  prism(merged,front,.3,material,layer)
 if roof_wall_domain is not None and ctx.get('roofSkirtOwners') is not None:
  # This facade owns the shell's coplanar perimeter skirt over exactly its bounded wall polygon. The generator subtracts
  # these faces after facade compilation so the shaped wall and its real openings remain visible.
  ctx['roofSkirtOwners'].append({'normal':n.tolist(),'tangent':u.tolist(),'plane':plane,'left':left,'zBaseM':z0,
                                 'domain':roof_wall_domain,'sourceId':profile.get('id')})
 # Vertical articulation: piers at the bay boundaries of a bank, and explicit pilasters. Both project from the
 # wall and are clipped to this wall segment.
 def draw_pier(x,zlo,zhi,pw,pp,pm):
  # The surround owns the edge of a recessed opening. A broad pilaster's
  # backing occupies only the solid material between those openings.
  rect=box(x-pw/2,zlo,x+pw/2,zhi)
  cuts=[]
  for op in ops:
   outline=op['_outline']
   if not rect.intersects(outline):continue
   portion=rect.intersection(outline)
   if portion.area>.0001:cuts.append(portion)
  if cuts:
   prism(rect.difference(unary_union(cuts)),pp,pp+.3,pm,layer)
  else:
   block(x,(zlo+zhi)/2,pw,zhi-zlo,pp+.3,pm,(pp-.3)/2,layer)
 for gi,g in enumerate(profile.get('groups',[])):
  if not g.get('pierWidthM') or 'region' not in g or not (g.get('minHeightM',0)<=H<=g.get('maxHeightM',float('inf'))):continue
  region,nx,ny,xs,ys,dx,dy=_grid(g,W,Hs);rx0,ry0,rx1,ry1=region;pw=g['pierWidthM'];pp=g.get('pierProjectionM',.1);pm=g.get('pierMaterial',mat)
  zlo,zhi=ry0*Hs,min(H,ry1*Hs)
  for k in range(nx+1):
   x=(rx0+k*(rx1-rx0)/nx)*W
   if x<low-pw/2 or x>high+pw/2 or zhi-zlo<.05:continue
   x=min(max(x,low+pw/2),high-pw/2);draw_pier(x,zlo,zhi,pw,pp,pm)
 for pier in profile.get('piers',[]):
  x=pier['x']*W;pw=pier.get('widthM',.5);pp=pier.get('projectionM',.12);yy=pier.get('y',[0,1]);zlo,zhi=yy[0]*Hs,min(H,yy[1]*Hs)
  if x<low or x>high or zhi-zlo<.05:continue
  draw_pier(min(max(x,low+pw/2),high-pw/2),zlo,zhi,pw,pp,pier.get('material',mat))
 # Spandrel panels between the rows of one column: a recessed or differently clad strip under each window.
 by_column={}
 for op in ops:
  if op.get('fitId') is None and op.get('group') is not None:by_column.setdefault((op['group'],op['ix']),[]).append(op)
 for (gi,ix),col in by_column.items():
  g=profile['groups'][gi]
  if not g.get('spandrelMaterial'):continue
  col=sorted(col,key=lambda o:o['rect'][1]);sr=g.get('spandrelRecessM',.06)
  for lower,upper in zip(col,col[1:]):
   gap_lo=lower['rect'][3];gap_hi=upper['rect'][1];x1=min(lower['rect'][0],upper['rect'][0]);x2=max(lower['rect'][2],upper['rect'][2])
   if gap_hi-gap_lo>.08:block((x1+x2)/2,(gap_lo+gap_hi)/2,x2-x1,gap_hi-gap_lo-.04,.12,g['spandrelMaterial'],-sr,layer)
 # A deep reveal is wall, not frame: the frame colour holds only a strip at the glass and a lip at the face, and
 # the opening's sides between them are the wall's own material, set back a few mm so the frame reads as a step.
 RF,RL,RS=.08,.04,.008
 def reveal(xc,zc,bw,bh,sx,sz,recess,proj,frame,wm):
  if recess<.18:block(xc,zc,bw,bh,recess+proj,frame,(proj-recess)/2);return
  block(xc,zc,bw,bh,RF,frame,-recess+RF/2)
  block(xc,zc,bw,bh,proj+RL,frame,(proj-RL)/2)
  # sx/sz: which way the opening lies from this piece (-1/0/1); the wall piece pulls back from it by RS.
  block(xc-sx*RS/2,zc-sz*RS/2,bw-abs(sx)*RS,bh-abs(sz)*RS,recess-RF-RL+.02,wm,(RF-RL-recess)/2,layer)
 def reveal_ring(ring,inner,recess,proj,frame,wm):
  if recess<.18:prism(ring,proj,recess+proj,frame,detail);return
  prism(ring,-recess+RF,RF,frame,detail)
  prism(ring,proj,proj+RL,frame,detail)
  prism(ring.difference(inner.buffer(RS,join_style=2)),-RL+.01,recess-RF-RL+.02,wm,layer)
 for j,op in enumerate(ops):
  x1,y1,x2,y2=op['rect'];w=x2-x1;h=y2-y1;cx=(x1+x2)/2;cy=(y1+y2)/2;s=op.get('style',{});kind=s.get('kind','window');fw=s.get('frameWidthM',.06);frame=s.get('frameMaterial','charcoal');proj=s.get('projectionM',.04);recess=s.get('recessM',.2)
  if kind=='projected_bay':recess-=proj
  ra=s.get('reliefArch')
  if ra:
   # A brick or stone arch in the wall above the opening: a semicircle (or a flatter segment) through the two
   # spring points, `bandM` wide, standing a little proud of the wall.
   band=ra.get('bandM',.35);rise_ratio=ra.get('riseRatio',1.);hs_=head_shape(s)
   source_bounds=op.get('_originalOutline',box(x1,y1,x2,y2)).bounds
   ox1,oy1,ox2,oy2=source_bounds;ow=ox2-ox1
   if hs_ is not None:centre_y=oy1+hs_[2][1]*(oy2-oy1);R=ow/2;cx=(ox1+ox2)/2
   else:
    hr=max(.05,(ow/2)*rise_ratio);R=(hr*hr+(ow/2)**2)/(2*hr);centre_y=oy2+hr-R;cx=(ox1+ox2)/2
   ring=Point(cx,centre_y).buffer(R+band,64).difference(Point(cx,centre_y).buffer(R,64)).intersection(box(ox1-band-.01,oy2 if hs_ is None else centre_y,ox2+band+.01,centre_y+R+band+.01))
   prism(ring.intersection(clip_box),ra.get('projectionM',.03),.06,ra.get('material',wall_mat(cx,cy)),layer)
  if kind=='louvre':
   # A louvred panel: a dark backing with horizontal slats, framed like a window.
   block(cx,cy,w,h,.06,s.get('material','charcoal'),-.12)
   for zz in np.arange(y1+.06,y2-.03,s.get('slatPitchM',.1)):block(cx,zz,w-.04,.02,.05,s.get('slatMaterial','silver'),-.06)
   for x in [x1+fw/2,x2-fw/2]:block(x,cy,fw,h,.1,frame,-.05)
   continue
  if kind=='rollup':
   # a vehicle door set back in its opening: the reveal's sides and soffit run back to it (a deep garage recess reads
   # as a cavity in the wall, not a door painted on its face)
   r=max(.25,float(s.get('recessM',.25)))
   if op.get('_contoured') or s.get('observedLeafMarks') or s.get('head') in ('arch','segmental'):
    from .door_contours import draw_observed_garage
    draw_observed_garage(door_leaves.draw_garage_leaf,
        lambda poly,d,t,m:prism(poly,d,t,m,detail),op['_originalOutline'],op['_outline'],
        s,r,wall_mat(cx,cy),ground)
    continue
   door_leaves.draw_garage_leaf(block,(x1,y1,x2,y2),s,depth=-r)
   # The casing is independent of the leaf finish. The rollup branch returns
   # before ordinary opening frames, so it must emit its own two jambs and
   # lintel; a dark surround must not recolour the whole pale door.
   door_outline=op.get('_originalOutline',box(x1,y1,x2,y2))
   casing=door_outline.difference(door_outline.buffer(-fw,join_style=2))
   casing=casing.difference(box(x1-fw,y1-fw,x2+fw,y1+fw))
   prism(casing.intersection(clip_box),-r+.06,.04,frame,detail)
   if r>.35:
    rm=wall_mat(cx,cy)
    for x in (x1+.04,x2-.04):block(x,cy,.08,h,r,rm,-r/2)
    block(cx,y2-.04,w,.08,r,rm,-r/2)
    # A deep ground-level garage passage has a floor return to the recessed
    # leaf. Its far edge stops at the leaf plane, leaving the opening clear.
    floor_z=ground(cx,-r/2)
    if y1<=floor_z+.25:
     block(cx,floor_z+.04,w,.08,max(.05,r-.08),rm,-r/2)
   continue
  # Polygon and arch openings retain the clipped observed contour for every
  # layer.  A negative buffer gives glazing the same shape and the difference
  # supplies a continuous reveal/frame around that shape.
  if op.get('_contoured'):
   outline=op['_outline'];original=op.get('_originalOutline',outline);glass=s.get('glassMaterial','glass')
   inner=original.buffer(-max(.006,fw),join_style=2)
   if inner.is_empty:inner=original
   prism(inner.intersection(clip_box),-recess,.025,glass,detail)
   # Intersect the original frame ring after buffering, so a segment boundary
   # clips the opening without creating a new frame edge at that boundary.
   frame_ring=original.difference(inner).intersection(clip_box)
   if not frame_ring.is_empty:reveal_ring(frame_ring,inner,recess,proj,frame,wall_mat(cx,cy))
   # The segment can cut through an arch or polygon.  Fill only the part of
   # its bounding box outside the actual contour, preserving wall material.
   corners=box(x1,y1,x2,y2).difference(outline.buffer(.002))
   prism(corners,0.,.3,wall_mat(cx,cy),layer)
   ox1,oy1,ox2,oy2=original.bounds;ow=ox2-ox1;oh=oy2-oy1
   for f in s.get('mullionFractions',[]):
    stripe=box(ox1+ow*f-.0185,oy1,ox1+ow*f+.0185,oy2).intersection(original).intersection(clip_box)
    prism(stripe,-recess+.035,.065,frame,detail)
   jw=s.get('jointWidthM',.08)
   for f in s.get('jointFractions',[]):
    # the joint between two lights of a ribbon: a full-depth mullion, like the band's jambs
    stripe=box(ox1+ow*f-jw/2,oy1,ox1+ow*f+jw/2,oy2).intersection(original).intersection(clip_box)
    prism(stripe,proj,recess+proj,frame,detail)
   for f in s.get('transomFractions',[]):
    stripe=box(ox1,oy1+oh*f-.021,ox2,oy1+oh*f+.021).intersection(original).intersection(clip_box)
    prism(stripe,-recess+.035,.065,frame,detail)
   sill=box(ox1-.03,oy1-.07,ox2+.03,oy1).intersection(clip_box)
   prism(sill,SILL,SILL+.275,frame,detail)
   if s.get('shade'):block(cx,y2+.04,w+.25,.06,.55,'charcoal',.25)
   continue
  # Deep interior backing, perimeter reveals and glazing use real, distinct planes.
  block(cx,cy,w,h,.035,'interior',-recess-.18)
  glass=s.get('glassMaterial','glass')
  block(cx,cy,w-.025,h-.025,.025,glass,-recess)
  shape=head_shape(s,w,h)
  if shape is None:
   wm=wall_mat(cx,cy)
   for x,sx in [(x1+fw/2,1),(x2-fw/2,-1)]:reveal(x,cy,fw,h,sx,0,recess,proj,frame,wm)
   for y,sz in [(y1+fw/2,1),(y2-fw/2,-1)]:reveal(cx,y,w,fw,0,sz,recess,proj,frame,wm)
  else:
   # Curved or pointed head: wall fills the corners above the springline, the jambs stop at the springline
   # and the frame follows the arc. Nothing of the frame stands in front of the infill.
   from shapely.geometry import Polygon as _P,box as _box
   outline=_P([(x1+a*w,y1+b*h) for a,b in shape]);spring=shape[2][1]
   corners=_box(x1,y1+spring*h-.01,x2,y2).difference(outline.buffer(.002))
   prism(corners,0.,.3,wall_mat(cx,cy),layer)
   jamb_h=spring*h
   wm=wall_mat(cx,cy)
   for x,sx in [(x1+fw/2,1),(x2-fw/2,-1)]:reveal(x,y1+jamb_h/2,fw,jamb_h,sx,0,recess,proj,frame,wm)
   reveal(cx,y1+fw/2,w,fw,0,1,recess,proj,frame,wm)
   arc=[(x1+a*w,y1+b*h) for a,b in shape[3:-1]]
   for pa,pb in zip(arc,arc[1:]):beam((pa[0],pa[1],(proj-recess)/2),(pb[0],pb[1],(proj-recess)/2),fw,frame)
   # Mullions, transoms and the glazing stay inside the outline: a mullion never rises into the infill.
   for f in s.get('mullionFractions',[]):block(x1+w*f,y1+jamb_h/2,.037,jamb_h,.065,frame,-recess+.035)
   for f in s.get('transomFractions',[]):
    if f<spring:block(cx,y1+h*f,w,.042,.065,frame,-recess+.035)
   block(cx,y1-.035,w+.06,.07,SILL+.12,frame,(SILL-.12)/2)
   if s.get('shade'):block(cx,y2+.04,w+.25,.06,.55,'charcoal',.25)
   if s.get('canopy'):
    d=s.get('canopyDepthM',1.6);block(cx,y2+.14,w+.4,.10,d,frame,d/2-.1);block(cx,y2+.2,w+.32,.025,d-.1,'glass_light',d/2-.1)
    for x in [x1+w*.12,x2-w*.12]:
     p=u*(left+x)+n*plane;ctx['cylinder']([*(p+n*.02),z0+y2+.85],[*(p+n*(d-.15)),z0+y2+.20],.02,detail,'steel',6)
   continue
  if kind=='terrace_frame':
   # Structural outside frame and the glazing at the back of the terrace are separate.
   for x in [x1+fw/2,x2-fw/2]:block(x,cy,fw,h,.35,frame,proj-.12)
   for y in [y1+fw/2,y2-fw/2]:block(cx,y,w,fw,.35,frame,proj-.12)
   block(cx,y1+.045,w,.09,recess+.2,'stone',-recess/2)
   railheight=min(1.1,h*.3)
   for x in [x1+w*.08,x2-w*.08]:block(x,y1+railheight/2,.035,railheight,.04,frame,-.08)
   block(cx,y1+railheight,w,.038,.06,frame,-.08)
   block(cx,y1+railheight*.53,w-.12,railheight*.84,.015,'glass_light',-.10)
  for f in s.get('mullionFractions',[]):block(x1+w*f,cy,.037,h,.065,frame,-recess+.035)
  for f in s.get('jointFractions',[]):block(x1+w*f,cy,s.get('jointWidthM',.08),h,recess+proj,frame,(proj-recess)/2)
  for f in s.get('transomFractions',[]):block(cx,y1+h*f,w,.042,.065,frame,-recess+.035)
  block(cx,y1-.035,w+.06,.07,SILL+.12,frame,(SILL-.12)/2)
  if kind=='projected_bay':
   # External cheeks and inter-floor panel cladding are anchored to the opening bank. A bay that names no cladding
   # is clad like the wall it stands out from (a photographed bay names none; zinc drew a house's bays black).
   clad=s.get('material') or wall_mat(cx,cy)
   for x in [x1-.06,x2+.06]:block(x,cy,.12,h+.65,proj+.2,clad,proj/2)
   block(cx,y2+.24,w+.24,.48,proj+.2,clad,proj/2)
  if s.get('shade'):
   block(cx,y2+.04,w+.25,.06,.55,'charcoal',.25)
   for depth in [.1,.23,.36,.49]:block(cx,y2+.07,w+.32,.026,.025,'charcoal',depth)
  if s.get('canopy'):
   d=s.get('canopyDepthM',1.6);block(cx,y2+.14,w+.4,.10,d,frame,d/2-.1);block(cx,y2+.2,w+.32,.025,d-.1,'glass_light',d/2-.1)
   for x in [x1+w*.12,x2-w*.12]:
    p=u*(left+x)+n*plane;ctx['cylinder']([*(p+n*.02),z0+y2+.85],[*(p+n*(d-.15)),z0+y2+.20],.02,detail,'steel',6)
 # Details read from the photograph that the lines above do not draw (pipeline/facade_detail.py): sashes, surrounds,
 # hoods, doors and steps of the openings; rustication, quoins, fire escapes, balconies and relief panels of the wall.
 # A door or garage reaching below the wall's foot (a street lower than the building's ground) opens into the foundation
 # the building stands on: the generator leaves a pocket behind it out of that foundation (ctx['pockets']).
 if ctx.get('pockets') is not None:
  for op in ops:
   x1,y1,x2,_=op['rect']
   if y1<-.05:
    rec=float((op.get('style') or {}).get('recessM',.2))+.3
    ctx['pockets'].append(Polygon([point(x1-.04,0,.03)[:2],point(x2+.04,0,.03)[:2],point(x2+.04,0,-rec)[:2],point(x1-.04,0,-rec)[:2]]))
 _bands=profile.get('bands',[]);_snap=None
 if _bands and profile.get('legacySnapCornice') is True and not ctx.get('roofShell'):
  _ti=max(range(len(_bands)),key=lambda i:_bands[i]['height']);_zt=_bands[_ti]['height']*Hs
  if H-1.2<=_zt<=H+2.5:_snap=(_zt,(H-_bands[_ti].get('thicknessM',.16)/2)-_zt,_ti)
 # Reliefs may sit on projecting band faces. Pass their actual plan/elevation support
 # and front depth to the relief renderer, which splits relief geometry at each step.
 relief_supports=[]
 for _bi,_band in enumerate(_bands):
  _bz=_band['height']*Hs
  if _snap is not None and _bi==_snap[2]:_bz+=_snap[1]
  if _bz>H+.5:continue
  _bx0,_bx1=_band.get('xRange',[0,1]);_bx0=max(low,_bx0*W);_bx1=min(high,_bx1*W)
  if _bx1-_bx0<.05:continue
  if _band.get('section'):
   for _rise,_hh,_projection in _band['section']:
    _path=facade_bands.support_path(_bx0,_bx1,_bz+_rise,bay_list)
    _plan=facade_bands.band_footprint(_path,_projection/2,_projection+.02)
    relief_supports.append({'rect':box(_bx0,_bz+_rise-_hh/2,_bx1,_bz+_rise+_hh/2),
                            'frontDepth':_projection+.01,'planFootprint':_plan})
  else:
   _thickness=float(_band.get('thicknessM',.16));_depth=float(_band.get('depthM',.2));_projection=float(_band.get('projectionM',.05))
   _path=facade_bands.support_path(_bx0,_bx1,_bz,bay_list)
   _plan=facade_bands.band_footprint(_path,_projection,_depth)
   relief_supports.append({'rect':box(_bx0,_bz-_thickness/2,_bx1,_bz+_thickness/2),
                           'frontDepth':_projection+_depth/2,'planFootprint':_plan})
 dctx=dict(block=block,prism=prism,slopedPrism=sloped_prism,beam=beam,point=point,wall_mat=wall_mat,clip=clip_box,low=low,high=high,W=W,H=H,Hs=Hs,mat=mat,put=ctx['put'],detail=detail,ground=ground,points=points,ground_v=ground_v,pitchedRoof=ctx.get('pitchedRoof') is not None,roofShell=ctx.get('roofShell',False),trees=ctx.get('front_trees'),reliefSupports=relief_supports,accessRoutes=profile.get('accessRoutes',[]))
 dctx['porches']=profile.get('porches',[])
 from .access_constraints import carriageway_limit
 dctx['accessLimit']=lambda center,width,start:carriageway_limit(point,ctx.get('_carriageway'),center,width,start)
 for op in ops:facade_detail.opening(dctx,op)
 facade_detail.wall(dctx,profile,ops)
 if recess_regions:facade_recesses.draw(dctx,recess_regions,layer)
 # Preserve observed band heights. Only an explicitly opted-in legacy profile
 # can use the old cornice snap; a sampled roof boundary never moves the reading.
 for _bi,band in enumerate(_bands):
  z=band['height']*Hs
  if _snap is not None and _bi==_snap[2]:z+=_snap[1]
  if z>H+.5:continue
  bx0,bx1=band.get('xRange',[0,1]);bx0=max(low,bx0*W);bx1=min(high,bx1*W)
  if bx1-bx0<.05:continue
  _range=band.get('xRange',[0,1]);_band_x0=float(_range[0])*W;_band_x1=float(_range[1])*W
  _band_identity=(c.get('sourceId',c['id']),profile['id'],_bi,round(z0,4))
  _band_emitted=ctx.setdefault('observedBayBands',set())
  if band.get('section'):
   # Stacked section sweep (rise, height, projection): the profile of a cornice or string course, not a painted line.
   for _si,(rise,hh,projection) in enumerate(band['section']):
    _identity=(*_band_identity,_si)
    for mesh in facade_bands.sweep_profile_band(
      point,source_point,_band_x0,_band_x1,low,high,z+rise-hh/2,hh,projection+.02,
      projection/2,bay_list,_band_emitted,_identity):ctx['put'](mesh,detail,_component_material(band,'stone'))
  else:
   hh=band.get('thicknessM',.16)
   for mesh in facade_bands.sweep_profile_band(
     point,source_point,_band_x0,_band_x1,low,high,z-hh/2,hh,band.get('depthM',.2),
     band.get('projectionM',.05),bay_list,_band_emitted,_band_identity):ctx['put'](mesh,detail,_component_material(band,'stone'))
 # Parapets: the photographed top outline of the wall, standing on the wall top (or a band following it).
 for _pi,pp in enumerate(profile.get('parapets',[])):
  if _pi in boundary_roofline_parapets and (not pp.get('bandM') or roof_domain_unresolved):continue
  pts=[(x*W,z*Hs) for x,z in pp.get('top',[])]
  if len(pts)<2:continue
  # A parapet outline tops the wall it was photographed on: an outline standing more than 3 m above this piece's
  # wall top, or below it, belongs to another piece of the frontage (a low corner, a set-back wing), not to this one.
  if min(z for _,z in pts)-H>3. or (max(z for _,z in pts)<=H+.02 and not pp.get('bandM')):continue
  px0,px1=max(low,min(x for x,_ in pts)),min(high,max(x for x,_ in pts))
  if px1-px0<.05:continue
  base=H-.02
  if pp.get('bandM'):
   lower=[(x,z-pp['bandM']) for x,z in pts][::-1];poly=Polygon(pts+lower).buffer(0)
  else:poly=Polygon(pts+[(pts[-1][0],base),(pts[0][0],base)]).buffer(0)
  parapet_low=min(z for _,z in pts)-float(pp.get('bandM',0.)) if pp.get('bandM') else base
  poly=poly.intersection(box(px0,parapet_low,px1,max(z for _,z in pts)+.01))
  if roof_wall_domain is not None and not pp.get('bandM'):poly=poly.difference(roof_wall_domain)
  prism(poly,0.,pp.get('thicknessM',.3),pp.get('material',mat),layer)
 # Ornaments: small repeated reliefs (diamonds, dentils, lamps) read from the photograph.
 for orn in profile.get('ornaments',[]):
  kind=orn.get('kind','diamond');z=orn['z']*Hs;size=orn.get('sizeM',.5);material=orn.get('material',mat);proj=orn.get('projectionM',.03)
  if _snap is not None and abs(z-_snap[0])<.6:z+=_snap[1]   # read with the cornice: moves with it
  if 'xs' in orn:xs_=[x*W for x in orn['xs']]
  else:
   a0,a1=orn.get('xRange',[0,1]);a0*=W;a1*=W;pitch=orn.get('pitchM',.6);count=max(1,int((a1-a0)/pitch));xs_=[a0+(i+.5)*(a1-a0)/count for i in range(count)]
  for x in xs_:
   if x<low or x>high or z>H+.5:continue
   if kind=='diamond':
    d=size/2;prism(Polygon([(x,z-d),(x+d,z),(x,z+d),(x-d,z)]),proj,.04,material,detail)
   elif kind=='dentil':block(x,z,size,orn.get('heightM',size),proj+.02,material,proj/2)
   elif kind=='lamp':
    block(x,z,.06,.06,.5,'charcoal',.25);block(x,z-.3,.32,.5,.32,'charcoal',.5)
   else:block(x,z,size,size,proj+.02,material,proj/2)
 # A sign keeps its full source layout when this facade is split into pieces.
 from . import facade_signs
 from .lettering import contours as sign_contours
 sign_support=clip_box
 from . import opening_lettering
 for op in ops:
  for poly,depth,thickness,material in opening_lettering.infill_parts(op):
   prism(poly,depth,thickness,material,detail)
 for sg in profile.get('signs',[]):
  if sg.get('mount') in ('opening-glass','opening-infill'):
   for poly,depth,thickness,material in opening_lettering.parts(sg,W,Hs,ops):
    prism(poly,depth,thickness,material,detail)
   continue
  for poly,depth,thickness,material in facade_signs.parts(sg,W,Hs,sign_support,sign_contours):
   prism(poly,depth,thickness,material,detail)
 # A generic stone plinth is only a fallback for legacy, unsourced grammar
 # profiles. Source-derived walls draw bands only when the reading measured
 # one; an unconditional base strip can cross garage and entry openings.
 if facade_bands.should_emit_default_plinth(profile):
  block((low+high)/2,.16,high-low,.32,.1,'stone',.045)
 if not profile.get('gableEnd') and not variable_roof_top:
  block((low+high)/2,H+.025,high-low,.10,.4,profile.get('copingMaterial','charcoal'),.015)
 return dict(record={'buildingId':c['id'],'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(high-low,2),'heightM':round(H,2),'edge':'grammar','floors':None,'columns':None,'archetype':profile['id'],'material':mat,'sourceIds':profile['sourceIds'],'status':('image-layout-fit' if any(o.get('fitId') for o in ops) else 'visual-grammar'),'grammarId':profile['id'],'openingCount':len(ops),'openingEvidence':{'explicitBoundaries':sum(o.get('evidence')=='observed-boundary' for o in ops),'negativeRegions':len(profile.get('surfaceEvidence',[]))},'fitIds':list(set(o['fitId'] for o in ops if o.get('fitId'))),'unresolved':list(profile.get('unresolved',[]))+roof_domain_unresolved,'streetFacing':True},components=len(ops)+bay_windows)
