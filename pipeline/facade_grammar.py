"""Compile observation-derived wall regions and window topology into metric geometry.
Profiles are data; no geographic IDs or addresses are interpreted here.
"""
import math
import numpy as np
from shapely.geometry import Point,Polygon,box
from .surface_evidence import admissible,resolve_material


RIBBON_PIER_M=.3   # lights of a bank standing at most this far apart are one band of glazing per storey

def ribbon_bank(group,W,region,dx,nx):
 """Whether a repeat bank is ribbon glazing: its lights stand at most RIBBON_PIER_M apart, so what divides them is a
 mullion, not wall. Drawn light by light, each with its own four-sided frame and sill and a sliver of wall between, a
 tile office block's continuous bands of glass came out as a grid of punched windows. A style may say so itself."""
 st=group.get('style') or {}
 if 'ribbon' in st:return bool(st['ribbon'])
 if st.get('kind','window')!='window' or W is None or nx<2:return False
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
   own=st.get('mullionFractions',[.25])
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

def head_shape(style):
 """Outline of an opening in its own unit square: flat, a semicircular arch, or a shallow segmental arch.

 `head` names the shape; `springRatio` (arch) is the height of the springline as a fraction of the opening, and
 `riseRatio` (segmental) the rise of the shallow arc. Photographs show the head shape directly; the grammar
 fills the corners above the springline with wall and runs the frame along the arc.
 """
 head=style.get('head','flat')
 if head=='arch':
  spring=style.get('springRatio',.68);arc=[[.5+.5*math.cos(t),spring+(1-spring)*math.sin(t)] for t in np.linspace(0,math.pi,17)]
  return [[0,0],[1,0],[1,spring]]+arc+[[0,0]]
 if head=='segmental':
  rise=style.get('riseRatio',.14);arc=[[.5+.5*math.cos(t),(1-rise)+rise*math.sin(t)] for t in np.linspace(0,math.pi,13)]
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
 shape=head_shape(op.get('style',{}))
 if shape is None:return box(x1*W,y1*H,x2*W,y2*H)
 return Polygon([(x1*W+a*w*W,y1*H+b*h*H) for a,b in shape])


def _component_material(item,fallback):
 """Resolve an evidence-backed component while preserving direct style keys."""
 if item.get('materialEvidence'):
  return resolve_material({**item,'material':item.get('material',fallback)})
 return item.get('material',fallback)

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
  if admissible(opening,profile,explicit=True):result.append(dict(opening,evidence=opening.get('evidence','observed-boundary'),ix=0,iy=0))
 return result


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
 if float(mid@n)<outward-profile.get('planeToleranceM',2.5):return None
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
 def prism(poly,depth,thickness,material,key=layer):
  """A wall piece of any outline in the facade plane (x right, z up), `thickness` deep behind `depth`."""
  import trimesh
  from shapely.geometry import Polygon as _P
  for q in ([poly] if poly.geom_type=='Polygon' else list(getattr(poly,'geoms',[]))):
   if q.is_empty or q.area<.001:continue
   mesh=trimesh.creation.extrude_polygon(q,height=thickness,engine='earcut');v=mesh.vertices.copy();mesh.vertices=np.array([point(x,z,depth-dd) for x,z,dd in v]);mesh.fix_normals();ctx['put'](mesh,key,material)
 def beam(a,b,thickness,material,key=detail):
  """A square member between two facade points (x,z,depth); frames follow an arch with these."""
  import trimesh
  aa=point(*a);bb=point(*b);length=np.linalg.norm(bb-aa)
  if length<.002:return
  mesh=trimesh.creation.box([thickness,thickness,length]);T=trimesh.geometry.align_vectors([0,0,1],(bb-aa)/length);T[:3,3]=(aa+bb)/2;mesh.apply_transform(T);ctx['put'](mesh,key,material)
 ops=[]
 clip_box=box(low,0,high,H)
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
 panels=profile.get('panels',[])
 def wall_mat(x,z):
  found=[p for p in panels if p.get('rect') and p['rect'][0]*W<=x<=p['rect'][2]*W and p['rect'][1]*Hs<=z<=p['rect'][3]*Hs]
  return _component_material(found[-1],mat) if found else mat
 # Partition each horizontal slice around its openings and material boundaries.
 zs=sorted(set([0,H]+[v for op in ops for v in [op['rect'][1],op['rect'][3]]]+[min(H,p['rect'][k]*Hs) for p in panels for k in [1,3]]))
 for bottom,top in zip(zs,zs[1:]):
  if top-bottom<.003:continue
  cy=(bottom+top)/2;active=[o['rect'] for o in ops if o['rect'][1]<cy<o['rect'][3]]
  xs=sorted(set([low,high]+[v for r in active for v in [r[0],r[2]]]+[np.clip(p['rect'][k]*W,low,high) for p in panels for k in [0,2]]))
  for x1,x2 in zip(xs,xs[1:]):
   cx=(x1+x2)/2
   if any(r[0]<cx<r[2] for r in active):continue
   block(cx,cy,x2-x1,top-bottom,.3,wall_mat(cx,cy),-.15,layer)
 # Vertical articulation: piers at the bay boundaries of a bank, and explicit pilasters. Both project from the
 # wall and are clipped to this wall segment.
 for gi,g in enumerate(profile.get('groups',[])):
  if not g.get('pierWidthM') or 'region' not in g or not (g.get('minHeightM',0)<=H<=g.get('maxHeightM',float('inf'))):continue
  region,nx,ny,xs,ys,dx,dy=_grid(g,W,Hs);rx0,ry0,rx1,ry1=region;pw=g['pierWidthM'];pp=g.get('pierProjectionM',.1);pm=g.get('pierMaterial',mat)
  zlo,zhi=ry0*Hs,min(H,ry1*Hs)
  for k in range(nx+1):
   x=(rx0+k*(rx1-rx0)/nx)*W
   if x<low-pw/2 or x>high+pw/2 or zhi-zlo<.05:continue
   x=min(max(x,low+pw/2),high-pw/2);block(x,(zlo+zhi)/2,pw,zhi-zlo,pp+.3,pm,(pp-.3)/2,layer)
 for pier in profile.get('piers',[]):
  x=pier['x']*W;pw=pier.get('widthM',.5);pp=pier.get('projectionM',.12);yy=pier.get('y',[0,1]);zlo,zhi=yy[0]*Hs,min(H,yy[1]*Hs)
  if x<low or x>high or zhi-zlo<.05:continue
  block(min(max(x,low+pw/2),high-pw/2),(zlo+zhi)/2,pw,zhi-zlo,pp+.3,pier.get('material',mat),(pp-.3)/2,layer)
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
   block(cx,cy,w,h,.08,s.get('material','zinc'),-.25)
   for zz in np.arange(y1,y2,.13):block(cx,zz,w,.017,.035,s.get('slatMaterial','charcoal'),-.19)
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
   if not frame_ring.is_empty:prism(frame_ring,proj,recess+proj,frame,detail)
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
   prism(sill,.035,.31,frame,detail)
   if s.get('shade'):block(cx,y2+.04,w+.25,.06,.55,'charcoal',.25)
   if kind in ['entry','shopfront']:
    for x in [cx-.12,cx+.12]:block(x,y1+h*.48,.025,.4,.05,'silver',-recess+.11)
   continue
  # Deep interior backing, perimeter reveals and glazing use real, distinct planes.
  block(cx,cy,w,h,.035,'interior',-recess-.18)
  glass=s.get('glassMaterial','glass')
  block(cx,cy,w-.025,h-.025,.025,glass,-recess)
  shape=head_shape(s)
  if shape is None:
   for x in [x1+fw/2,x2-fw/2]:block(x,cy,fw,h,recess+proj,frame,(proj-recess)/2)
   for y in [y1+fw/2,y2-fw/2]:block(cx,y,w,fw,recess+proj,frame,(proj-recess)/2)
  else:
   # Arched or segmental head: wall fills the corners above the springline, the jambs stop at the springline
   # and the frame follows the arc. Nothing of the frame stands in front of the infill.
   from shapely.geometry import Polygon as _P,box as _box
   outline=_P([(x1+a*w,y1+b*h) for a,b in shape]);spring=shape[2][1]
   corners=_box(x1,y1+spring*h-.01,x2,y2).difference(outline.buffer(.002))
   prism(corners,0.,.3,wall_mat(cx,cy),layer)
   jamb_h=spring*h
   for x in [x1+fw/2,x2-fw/2]:block(x,y1+jamb_h/2,fw,jamb_h,recess+proj,frame,(proj-recess)/2)
   block(cx,y1+fw/2,w,fw,recess+proj,frame,(proj-recess)/2)
   arc=[(x1+a*w,y1+b*h) for a,b in shape[3:-1]]
   for pa,pb in zip(arc,arc[1:]):beam((pa[0],pa[1],(proj-recess)/2),(pb[0],pb[1],(proj-recess)/2),fw,frame)
   # Mullions, transoms and the glazing stay inside the outline: a mullion never rises into the infill.
   for f in s.get('mullionFractions',[.25]):block(x1+w*f,y1+jamb_h/2,.037,jamb_h,.065,frame,-recess+.035)
   for f in s.get('transomFractions',[.76]):
    if f<spring:block(cx,y1+h*f,w,.042,.065,frame,-recess+.035)
   block(cx,y1-.035,w+.06,.07,.31,frame,.035)
   if s.get('shade'):block(cx,y2+.04,w+.25,.06,.55,'charcoal',.25)
   if kind in ['entry','shopfront']:
    for x in [cx-.12,cx+.12]:block(x,y1+h*.48,.025,.4,.05,'silver',-recess+.11)
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
  for f in s.get('mullionFractions',[.25]):block(x1+w*f,cy,.037,h,.065,frame,-recess+.035)
  for f in s.get('jointFractions',[]):block(x1+w*f,cy,s.get('jointWidthM',.08),h,recess+proj,frame,(proj-recess)/2)
  for f in s.get('transomFractions',[.76]):block(cx,y1+h*f,w,.042,.065,frame,-recess+.035)
  block(cx,y1-.035,w+.06,.07,.31,frame,.035)
  if kind=='projected_bay':
   # External cheeks and inter-floor panel cladding are anchored to the opening bank.
   for x in [x1-.06,x2+.06]:block(x,cy,.12,h+.65,proj+.2,s.get('material','zinc'),proj/2)
   block(cx,y2+.24,w+.24,.48,proj+.2,s.get('material','zinc'),proj/2)
  if s.get('shade'):
   block(cx,y2+.04,w+.25,.06,.55,'charcoal',.25)
   for depth in [.1,.23,.36,.49]:block(cx,y2+.07,w+.32,.026,.025,'charcoal',depth)
  if kind in ['entry','shopfront']:
   for x in [cx-.12,cx+.12]:block(x,y1+h*.48,.025,.4,.05,'silver',-recess+.11)
  if s.get('canopy'):
   d=s.get('canopyDepthM',1.6);block(cx,y2+.14,w+.4,.10,d,frame,d/2-.1);block(cx,y2+.2,w+.32,.025,d-.1,'glass_light',d/2-.1)
   for x in [x1+w*.12,x2-w*.12]:
    p=u*(left+x)+n*plane;ctx['cylinder']([*(p+n*.02),z0+y2+.85],[*(p+n*(d-.15)),z0+y2+.20],.02,detail,'steel',6)
  # Stable blinds, only behind the glazing. Sparse panels reduce geometric cost.
  if kind not in ['entry','shopfront','terrace_frame']:
   cuts=[0.]+list(s.get('jointFractions',[]))+[1.]      # a ribbon's lights take their blinds one by one
   for k,(fa,fb) in enumerate(zip(cuts,cuts[1:])):
    if (j+k)%5 in [1,3]:ch=h*(.22 if (j+k)%5==1 else .47);block(x1+w*(fa+fb)/2,y2-ch/2-fw,w*(fb-fa)-fw*2,ch,.02,'curtain',-recess-.07)
 for band in profile.get('bands',[]):
  z=band['height']*Hs
  if z>H+.5:continue
  bx0,bx1=band.get('xRange',[0,1]);bx0=max(low,bx0*W);bx1=min(high,bx1*W)
  if bx1-bx0<.05:continue
  if band.get('section'):
   # Stacked section sweep (rise, height, projection): the profile of a cornice or string course, not a painted line.
   for rise,hh,projection in band['section']:block((bx0+bx1)/2,z+rise,bx1-bx0,hh,projection+.02,_component_material(band,'stone'),projection/2)
  else:block((bx0+bx1)/2,z,bx1-bx0,band.get('thicknessM',.16),band.get('depthM',.2),_component_material(band,'stone'),band.get('projectionM',.05))
 # Parapets: the photographed top outline of the wall, standing on the wall top (or a band following it).
 for pp in profile.get('parapets',[]):
  pts=[(x*W,z*Hs) for x,z in pp.get('top',[])]
  if len(pts)<2:continue
  # A parapet outline tops the wall it was photographed on: an outline standing more than 3 m above this piece's
  # wall top, or below it, belongs to another piece of the frontage (a low corner, a set-back wing), not to this one.
  if min(z for _,z in pts)-H>3. or max(z for _,z in pts)<=H+.02:continue
  px0,px1=max(low,min(x for x,_ in pts)),min(high,max(x for x,_ in pts))
  if px1-px0<.05:continue
  base=H-.02
  if pp.get('bandM'):
   lower=[(x,z-pp['bandM']) for x,z in pts][::-1];poly=Polygon(pts+lower).buffer(0)
  else:poly=Polygon(pts+[(pts[-1][0],base),(pts[0][0],base)]).buffer(0)
  poly=poly.intersection(box(px0,base,px1,max(z for _,z in pts)+.01))
  prism(poly,0.,pp.get('thicknessM',.3),pp.get('material',mat),layer)
 # Ornaments: small repeated reliefs (diamonds, dentils, lamps) read from the photograph.
 for orn in profile.get('ornaments',[]):
  kind=orn.get('kind','diamond');z=orn['z']*Hs;size=orn.get('sizeM',.5);material=orn.get('material',mat);proj=orn.get('projectionM',.03)
  if 'xs' in orn:xs_=[x*W for x in orn['xs']]
  else:
   a0,a1=orn.get('xRange',[0,1]);a0*=W;a1*=W;pitch=orn.get('pitchM',.6);count=max(1,int((a1-a0)/pitch));xs_=[a0+(i+.5)*(a1-a0)/count for i in range(count)]
  for x in xs_:
   if x<low or x>high or z>H+.6:continue
   if kind=='diamond':
    d=size/2;prism(Polygon([(x,z-d),(x+d,z),(x,z+d),(x-d,z)]),proj,.04,material,detail)
   elif kind=='dentil':block(x,z,size,orn.get('heightM',size),proj+.02,material,proj/2)
   elif kind=='lamp':
    block(x,z,.06,.06,.5,'charcoal',.25);block(x,z-.3,.32,.5,.32,'charcoal',.5)
   else:block(x,z,size,size,proj+.02,material,proj/2)
 # Signs: a board and literal photographed text, placed where the photograph shows them.
 for sg in profile.get('signs',[]):
  sx1,sy1,sx2,sy2=sg['rect'];sx1*=W;sx2*=W;sy1*=Hs;sy2*=Hs
  if sx2<low or sx1>high or sy1>H+1.:continue
  sx1,sx2=max(low,sx1),min(high,sx2)
  if sx2-sx1<.2 or sy2-sy1<.1:continue
  dep=sg.get('depthM',.05)
  if sg.get('boardMaterial'):block((sx1+sx2)/2,(sy1+sy2)/2,sx2-sx1,sy2-sy1,dep,sg['boardMaterial'],dep/2)
  text=sg.get('text')
  if text:
   import trimesh
   from .lettering import contours
   geometry,ww,hh=contours(text,sg.get('font','Arial Narrow Bold.ttf'));scale=min((sx2-sx1)*.92/ww,(sy2-sy1)*.78/hh);gh=hh*scale;cx_=(sx1+sx2)/2;cy_=(sy1+sy2)/2
   for poly in ([geometry] if geometry.geom_type=='Polygon' else list(getattr(geometry,'geoms',[]))):
    if poly.is_empty or poly.area<1e-6:continue
    mesh=trimesh.creation.extrude_polygon(poly,height=.03,engine='earcut');v=mesh.vertices.copy()
    xx=cx_+(v[:,0]-ww/2)*scale;yy=cy_-gh/2+(hh-v[:,1])*scale
    mesh.vertices=np.array([point(x,y,dep+d) for x,y,d in zip(xx,yy,v[:,2])]);mesh.fix_normals();ctx['put'](mesh,detail,sg.get('textMaterial','white'))
 block((low+high)/2,.16,high-low,.32,.1,'stone',.045)
 if not profile.get('gableEnd'):block((low+high)/2,H+.025,high-low,.10,.4,profile.get('copingMaterial','charcoal'),.015)
 return dict(record={'buildingId':c['id'],'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(high-low,2),'heightM':round(H,2),'edge':'grammar','floors':None,'columns':None,'archetype':profile['id'],'material':mat,'sourceIds':profile['sourceIds'],'status':('image-layout-fit' if any(o.get('fitId') for o in ops) else 'visual-grammar'),'grammarId':profile['id'],'openingCount':len(ops),'openingEvidence':{'explicitBoundaries':sum(o.get('evidence')=='observed-boundary' for o in ops),'negativeRegions':len(profile.get('surfaceEvidence',[]))},'fitIds':list(set(o['fitId'] for o in ops if o.get('fitId'))),'unresolved':profile.get('unresolved',[]),'streetFacing':True},components=len(ops))
