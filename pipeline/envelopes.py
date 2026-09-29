"""Evidence-driven envelope graph -> articulated surfaces, returns and assemblies.
Geographic matching and observations live in JSON. Operators work in facade meters.
No address-, landmark- or architectural-style-specific code is permitted here.
"""
import math,re,copy
import numpy as np
import trimesh
from shapely.geometry import Polygon,Point,box,LineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter1d
from .surface_evidence import admissible,resolve_material

SKIN=.24   # thickness of every emitted skin slab (prism); returns span only the depth the slabs leave open
ROOF_TOLERANCE_M=.3


def _sheet(ctx,quad,outward,key,mat):
 """One opaque sheet whose front faces `outward`. The viewer culls back faces (glTF doubleSided is
 false), so a sheet wound toward its own building is invisible from the street: winding is evidence."""
 v=np.asarray(quad,float);nrm=np.cross(v[1]-v[0],v[2]-v[0])
 if np.linalg.norm(nrm)<1e-9:return
 faces=[[0,1,2],[0,2,3]] if np.dot(nrm,outward)>0 else [[0,2,1],[0,3,2]]
 ctx['put'](trimesh.Trimesh(v,faces,process=False),key,mat)

def _twosided(ctx,quad,key,mat):
 v=np.asarray(quad,float);ctx['put'](trimesh.Trimesh(v,[[0,1,2],[0,2,3],[0,2,1],[0,3,2]],process=False),key,mat)


def aperture_shape(style):
 if style.get('shapeKind')=='arch':
  spring=style.get('springRatio',.68)
  return [[0,0],[1,0],[1,spring]]+[[.5+.5*math.cos(a),spring+(1-spring)*math.sin(a)] for a in np.linspace(0,math.pi,17)]+[[0,0]]
 return style.get('shape',[[0,0],[1,0],[1,1],[0,1]])

def return_edges(node,region,nodes,shapes):
 """Adjacent patches own only their exposed depth difference, not two buried walls."""
 ring=list(orient(region,1.0).exterior.coords)   # CCW in facade (x right, z up) as seen from the street
 for aa,bb in zip(ring,ring[1:]):
  edge=LineString([aa,bb]);remaining=edge
  for other in nodes:
   if other['id']==node['id'] or other['parentId']!=node['parentId']:continue
   shared=edge.intersection(shapes[other['id']].boundary)
   if shared.length<.001:continue
   remaining=remaining.difference(shared)
   if node['depthM']>other['depthM']+.002:
    for line in _lines(shared):yield line.coords[0],line.coords[-1],other['depthM']
  for line in _lines(remaining):yield line.coords[0],line.coords[-1],node['parentDepthM']

def _lines(g):
 if g.is_empty:return []
 if g.geom_type=='LineString':return [g]
 return [p for q in getattr(g,'geoms',[]) for p in _lines(q)]

def of_source(c,profile):
 m=profile['match']
 return bool((m.get('featureId')==c.get('sourceId',c['id'])) or (c.get('osm') and str(m.get('osmWayId'))==str(c['osm']['id'])))


def matches(c,profile):
 if not of_source(c,profile):return False
 m=profile['match']
 if m.get('partId') is not None:return m['partId']==c['id']
 return owns_frontage(c,profile)


def owns_frontage(c,profile):
 """A part of a split outline takes a profile of its source only where the photographed frontage is its own. Every
 part used to match by the source identity and frame the profile on its own outermost edge facing that way, so a
 lower wing whose edge ran behind a tall block's street face carried that face's loft roofs over its own roof."""
 whole=c.get('sourceG')
 if whole is None or profile.get('normal') is None:return True
 if profile.get('referenceLine'):
  from shapely.geometry import LineString
  return c['g'].distance(LineString([tuple(q) for q in profile['referenceLine']]))<=profile.get('planeToleranceM',3)
 try:return part_span(c,profile,_frame_of(whole,profile)) is not None
 except ValueError:return False


def _frontage_edges(g,profile):
 n=np.array(profile['normal'],float);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]]);edges=[]
 for p in _polys(g):
  a=np.array(p.exterior.coords)
  for pa,pb in zip(a,a[1:]):
   v=pb-pa;le=np.linalg.norm(v)
   if le>2 and abs(v/le@u)>.96:edges.append((le,float((pa+pb)@n/2),pa,pb))
 return n,u,edges


def _frame_of(g,profile):
 n,u,edges=_frontage_edges(g,profile)
 if profile.get('referenceLine'):
  a,b=np.array(profile['referenceLine'],float);left,right=sorted([a@u,b@u]);plane=float(((a+b)/2)@n)
 else:
  if not edges:raise ValueError('No compatible footprint frontage: '+profile['id'])
  plane=max(e[1] for e in edges);es=[e for e in edges if e[1]>plane-profile.get('planeToleranceM',3)]
  points=np.concatenate([np.array([e[2],e[3]]) for e in es]);left=float(min(points@u));right=float(max(points@u))
 return {'normal':n,'tangent':u,'left':left,'width':right-left,'plane':plane}


def part_span(c,profile,f):
 """A part's own stretch (tangent coordinates) of a frontage framed on the whole outline: its edges within the plane
 tolerance of the frame's plane, or None when it has none there."""
 n,u,edges=_frontage_edges(c['g'],profile);es=[e for e in edges if e[1]>f['plane']-profile.get('planeToleranceM',3)]
 if not es:return None
 pts=np.concatenate([np.array([e[2],e[3]]) for e in es]);return float(min(pts@u)),float(max(pts@u))


def frame(c,profile):
 """The frontage frame a profile's fractions refer to. A part of a split outline frames it on the whole outline
 (`span` is the part's own stretch, which is all the part draws): framed on each part's own edges, one frontage design
 was squeezed into every part that shared the frontage, and a split that moved put its curtain wall on the wrong bays."""
 whole=c.get('sourceG')
 f=_frame_of(whole if whole is not None else c['g'],profile)
 if whole is not None:f['span']=part_span(c,profile,f)
 return f


def span_clip(f,pad=.02):
 """Half-spaces (normal, offset: keep normal.p <= offset) holding what a part draws of a frontage to its own stretch."""
 if not f.get('span'):return None
 u=np.asarray(f['tangent'],float);a,b=f['span']
 return [((-u[0],-u[1],0.),-(a-pad)),((u[0],u[1],0.),b+pad)]


def _polys(g):
 if g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 return [p for part in getattr(g,'geoms',[]) for p in _polys(part)]


def fit_levels(profile,f,samples,z0):
 """Bounded, robust roof-band fitting. Image labels select the relevant roof family.
 A high roof behind the street is not allowed to set the entire street-wall height.
 """
 levels={};evidence=[];x=(samples[:,:2]@f['tangent']-f['left'])/f['width'];depth=samples[:,:2]@f['normal']-f['plane'];z=samples[:,2]-z0
 for name,a in profile.get('levels',{}).items():
  prior=a['priorM'];radius=a.get('searchRadiusM',2.);span=a.get('span',[0,1]);dr=a.get('depthRangeM',[-8,.5])
  select=(x>=span[0])&(x<=span[1])&(depth>=dr[0])&(depth<=dr[1])&(z>=prior-radius)&(z<=prior+radius)
  values=z[select];value=prior;status='image-estimate'
  if radius==0:status=a.get('status','fixed-image-estimate')
  elif len(values)>=30:
   edges=np.arange(prior-radius,prior+radius+.125,.125);hist,_=np.histogram(values,edges);centers=(edges[:-1]+edges[1:])/2
   score=gaussian_filter1d(hist.astype(float),1.4)*np.exp(-.5*((centers-prior)/max(radius,.2))**2)
   peak=centers[np.argmax(score)];cluster=values[np.abs(values-peak)<.35];value=float(np.quantile(cluster,.75));status='bounded-lidar-fit'
  levels[name]=value;evidence.append({'level':name,'heightM':round(value,3),'priorM':prior,'sampleCount':len(values),'status':status,'searchRangeM':[prior-radius,prior+radius],'span':span,'depthRangeM':dr})
 return levels,evidence


def value(v,levels):
 if isinstance(v,(int,float)):return float(v)
 m=re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)([+-][0-9.]+)?',v)
 if not m or m[1] not in levels:raise ValueError('Invalid level reference '+v)
 return levels[m[1]]+float(m[2] or 0)


def resolved_nodes(profile,levels):
 nodes=[]
 def visit(node,parent=None):
  d=copy.deepcopy(node)
  if parent:
   x0,z0,x1,z1=parent['rect'];a,b,c,e=d.get('rect',[0,0,1,1]);rect=[x0+a*(x1-x0),z0+b*(z1-z0),x0+c*(x1-x0),z0+e*(z1-z0)]
   d['depthM']=parent.get('depthM',0)+d.get('offsetM',0)
  else:
   x0,x1=d.get('span',[0,1]);z0,z1=[value(x,levels) for x in d['z']];rect=[x0,z0,x1,z1]
  if rect[2]<=rect[0] or rect[3]<=rect[1]:raise ValueError('Degenerate envelope node '+d['id'])
  d['rect']=rect;d['parentId']=parent['id'] if parent else None;d['parentDepthM']=parent.get('depthM',0) if parent else -profile.get('replaceDepthM',10)
  d['material']=d.get('material',parent.get('material','stucco') if parent else 'stucco');d['material']=resolve_material(d);d.setdefault('depthM',0)
  nodes.append(d)
  for ch in d.get('children',[]):visit(ch,d)
 for node in profile['nodes']:visit(node)
 return nodes


def prepare(ctx,c,z0,samples,profiles):
 # Every profile of the source outline is resolved, so the corner joins and shared levels between frontages come out
 # the same in every part of a split outline; each part keeps (draws, masks) only the frontages it owns.
 resolved=[]
 for p in profiles:
  if not (of_source(c,p) if c.get('sourceG') is not None else matches(c,p)):continue
  try:f=frame(c,p)
  except ValueError:
   if matches(c,p):raise
   continue
  levels,ev=fit_levels(p,f,samples,z0)
  resolved.append({'profile':p,'frame':f,'levels':levels,'levelEvidence':ev,'joins':[]})
 # Explicit observed corner correspondences are solved jointly in the ground plane.
 byid={r['profile']['id']:r for r in resolved}
 for r in resolved:
  for link in r['profile'].get('joins',[]):
   other=byid.get(link['profile'])
   if other is None:continue
   f1,f2=r['frame'],other['frame'];A=np.array([f1['normal'],f2['normal']])
   if abs(np.linalg.det(A))<.1:continue
   q=np.linalg.solve(A,[f1['plane'],f2['plane']]);updates=[]
   for fr,end in [(f1,link['end']),(f2,link['otherEnd'])]:
    xx=float(q@fr['tangent']);current=fr['left']+(fr['width'] if end=='end' else 0);updates.append((fr,end,xx,current))
   delta=max(abs(x-old) for fr,end,x,old in updates);accepted=delta<=link.get('maxAdjustmentM',5)
   if accepted:
    for fr,end,x,old in updates:
     if end=='end':fr['width']=x-fr['left']
     else:fr['width']+=fr['left']-x;fr['left']=x
   event={'otherProfile':other['profile']['id'],'maxAdjustmentM':round(delta,3),'accepted':bool(accepted),'corner':q.tolist()}
   r['joins'].append(event);other['joins'].append({**event,'otherProfile':r['profile']['id']})
 # Share only explicitly corresponding height datums, never every equal-looking row.
 groups={}
 for r in resolved:
  for name in r['profile'].get('shareLevels',[]):
   groups.setdefault(name,[]).append(r)
 for name,rs in groups.items():
  if len(rs)<2:continue
  vals=[rr['levels'][name] for rr in rs];a=max(rr['profile']['levels'][name]['priorM']-rr['profile']['levels'][name].get('searchRadiusM',2) for rr in rs);b=min(rr['profile']['levels'][name]['priorM']+rr['profile']['levels'][name].get('searchRadiusM',2) for rr in rs)
  if a>b or max(vals)-min(vals)>3:continue
  joint=float(np.clip(np.median(vals),a,b))
  for rr in rs:
   rr['levels'][name]=joint
   ev=next(e for e in rr['levelEvidence'] if e['level']==name);ev['beforeJointFitM']=ev['heightM'];ev['heightM']=round(joint,3);ev['sharedAcrossFrontages']=len(rs)
 for r in resolved:r['nodes']=resolved_nodes(r['profile'],r['levels'])
 return [r for r in resolved if matches(c,r['profile'])]


def root_slots(resolved):
 """Plan slot of every root skin: its x-extent on the frontage by the replacement depth. Root returns
 stand exactly on these x-ends, so the host may be cut here and nowhere else."""
 for r in resolved:
  f=r['frame'];p=r['profile'];u=f['tangent'];n=f['normal'];W,left,plane=f['width'],f['left'],f['plane'];depth=p.get('replaceDepthM',10)
  for d in r['nodes']:
   if d['parentId'] is not None:continue
   x0,z0,x1,z1=d['rect'];a=left+x0*W;b=left+x1*W
   yield d,Polygon([u*x+n*y for x,y in [(a,plane+3),(b,plane+3),(b,plane-depth),(a,plane-depth)]]),float(z0),float(z1)

def replacement_mask(resolved):
 """Host walls are removed only where an emitted, closed skin box replaces them (cut domain = emitted
 skin, the rule surface_coverage.replacement_transaction states). Frame padding produced open slivers."""
 return unary_union([slot for _,slot,_,_ in root_slots(resolved)])

def roof_mask(resolved,host_height,tolerance=ROOF_TOLERANCE_M):
 """The host cap is removed only under skins that rise above it (their box gets its own lid, see `lids`). A skin
 whose top meets the cap keeps the cap as its lid; a lower skin keeps the host roof above it and the host wall
 band closes the front between skin top and cap. Cutting the cap under a skin that merely reached it left the
 replacement box open to the sky along every observed frontage."""
 return unary_union([slot for _,slot,_,z1 in root_slots(resolved) if z1>host_height+tolerance])

def lids(resolved,host_height,tolerance=ROOF_TOLERANCE_M):
 """Plan slot and top height (above the building base) of every root skin that rises above the host cap: the
 box that replaces the host there is closed by a lid at that height, so the roof is continuous."""
 return [(slot,z1) for _,slot,_,z1 in root_slots(resolved) if z1>host_height+tolerance]

def coverage_spans(resolved,z0,host_height,tolerance=ROOF_TOLERANCE_M):
 """For the facade compiler: per root skin, its plan slot and the absolute height band its closed box
 replaces. A skin whose top is the roof owns the host height up to the cap, so no sliver band appears."""
 spans=[]
 for _,slot,lo,hi in root_slots(resolved):
  top=max(hi,host_height) if hi>=host_height-tolerance else hi
  spans.append({'slot':slot,'zlo':z0+lo,'zhi':z0+top})
 return spans


AGREE_DE=15.


def _face_colour(cells,face):
 """Area-weighted median colour of one photograph's 2 m cells over a part's visible face, when they cover 30 % of it."""
 ws,cols=[],[]
 for x0,z0,x1,z1,rgb,cov in cells:
  a=face.intersection(box(x0,z0,x1,z1)).area*cov
  if a>0:ws.append(a);cols.append(rgb)
 if not ws or sum(ws)<.3*face.area:return None
 cols=np.asarray(cols,float);ws=np.asarray(ws);med=[]
 for k in range(3):
  o=np.argsort(cols[:,k]);cw=np.cumsum(ws[o]);med.append(cols[o[np.searchsorted(cw,cw[-1]/2)],k])
 return med


def _lab(rgb):
 c=np.asarray(rgb,float)/255.;c=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
 X=c@np.array([[.4124,.2126,.0193],[.3576,.7152,.1192],[.1805,.0722,.9505]])/np.array([.9505,1.,1.089])
 f=np.where(X>.008856,np.cbrt(X),7.787*X+16/116);return np.array([116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])])


def agreed_colour(cols):
 """The colour at least two photographs agree on: the largest set of photographs whose colours all agree pairwise
 within dE 15 (lightness at half weight), and its median; None when no two agree (one photograph's shade,
 foliage or pose error is not believed on its own)."""
 # Photographs under similar light only (dE 15, lightness at half weight): a part takes their median colour, and a
 # median of sunlit and shaded photographs (which colour_agreement.same_material would group) gave one storey of a
 # brick wall a lighter colour than the next, a band the wall does not have.
 from itertools import combinations
 labs=[_lab(c) for c in cols]
 def de(i,j):return math.sqrt(((labs[i][0]-labs[j][0])*.5)**2+(labs[i][1]-labs[j][1])**2+(labs[i][2]-labs[j][2])**2)
 for size in range(len(cols),1,-1):
  for g in combinations(range(len(cols)),size):
   if all(de(i,j)<AGREE_DE for i,j in combinations(g,2)):return np.median(np.asarray([cols[i] for i in g],float),0)
 return None


def observed_colour(mat,obs,face):
 """The photographed colour of an envelope part (read_envelope_colours_v32.py): each photograph's area-weighted median
 of its 2 m cells over the part's visible face (when they cover 30 % of it), and of those the colour two or more
 photographs agree on. The part keeps its designed shape and material family (a recipe takes the colour as a
 variant); designed flat colours (glass, metal, dark panels) stay, and so does the designed colour when the
 photographs do not agree. A single photograph recoloured a brick wall's upper storeys olive from shaded foliage."""
 from . import material_recipes
 base=mat.split('#')[0]
 if not obs or face.is_empty or face.area<1. or base not in material_recipes.RECIPES:return mat
 photos=obs.get('photos') or ([{'cells':obs['cells']}] if obs.get('cells') else [])
 cols=[c for c in (_face_colour(ph.get('cells',[]),face) for ph in photos) if c is not None]
 med=agreed_colour(cols) if cols else None
 if med is None:return mat
 return base+'#%02x%02x%02x'%tuple(int(max(0,min(255,round(v)))) for v in med)


def compile(ctx,c,z0,resolved):
 records=[];opening_count=0
 for r in resolved:
  p=r['profile'];f=r['frame'];u,n=f['tangent'],f['normal'];W,left,plane=f['width'],f['left'],f['plane'];angle=math.atan2(u[1],u[0]);nodes=r['nodes'];prefix='__'+c['id']+'__observed_'+p['id'];wallkey='buildings'+prefix;detail='facades'+prefix
  clip=span_clip(f)
  if clip and ctx.get('set_clip'):ctx['set_clip'](clip)
  # The frontage frame is bound when the helpers are made, not read when they are called: a component's own loop
  # variable (a clock's tick angle) must never re-aim every box drawn after it on the same frontage.
  def point(x,z,depth,u=u,n=n,left=left,plane=plane,z0=z0):return np.r_[u*(left+x)+n*(plane+depth),z0+z]
  def block(x,z,w,h,d,mat,depth=0,key=detail,angle=angle):
   if min(w,h,d)>.002:ctx['cube'](point(x,z,depth),[w,d,h],key,mat,angle)
  def beam(a,b,thickness,mat='verdigris',key=detail):
   # Rectangular members retain the silhouette of beams, rather than round poles.
   aa=point(*a);bb=point(*b);length=np.linalg.norm(bb-aa)
   if length<.002:return
   mesh=trimesh.creation.box([thickness,thickness,length]);T=trimesh.geometry.align_vectors([0,0,1],(bb-aa)/length);T[:3,3]=(aa+bb)/2;mesh.apply_transform(T);ctx['put'](mesh,key,mat)
  def prism(poly,depth,thickness,mat,key=wallkey):
   for q in _polys(poly):
    if q.area<.001:continue
    mesh=trimesh.creation.extrude_polygon(q,height=thickness,engine='earcut');v=mesh.vertices.copy();mesh.vertices=np.array([point(x,z,depth-dd) for x,z,dd in v]);mesh.fix_normals();ctx['put'](mesh,key,mat)
  def patchshape(node):
   x0,z0,x1,z1=node['rect'];x0*=W;x1*=W
   return Polygon([(x0+a*(x1-x0),z0+b*(z1-z0)) for a,b in node['shape']]) if node.get('shape') else box(x0,z0,x1,z1)
  node_shapes={d['id']:patchshape(d) for d in nodes}
  photo=ctx.get('ENVELOPE_COLOURS',{}).get(p['id']+'|'+c['id']);recoloured=0
  for node in nodes:
   region=node_shapes[node['id']];x0,zmin,x1,zmax=region.bounds;wid=x1-x0;H=zmax-zmin;depth=node['depthM'];mat=node['material'];children=[node_shapes[d['id']] for d in nodes if d['parentId']==node['id']]
   visible=region.difference(unary_union(children));ops=[]
   if photo:
    new_mat=observed_colour(mat,photo,visible)
    if new_mat!=mat:mat=new_mat;recoloured+=1
   for group in node.get('grids',[]):
    a,b,cc,dd=group.get('rect',[0,0,1,1]);nx,ny=group.get('columns',1),group.get('rows',1);s=group.get('style',{});xs=group.get('xCenters',[(j+.5)/nx for j in range(nx)]);ys=group.get('yCenters',[(j+.5)/ny for j in range(ny)])
    for ix,x in enumerate(xs):
     for iy,y in enumerate(ys):
      if [ix,iy] in group.get('omit',[]):continue
      ow=wid*(cc-a)/nx*s.get('widthRatio',.75);oh=H*(dd-b)/ny*s.get('heightRatio',.78);cx=x0+(a+x*(cc-a))*wid;cz=zmin+(b+y*(dd-b))*H
      shape=aperture_shape(s)
      uv={'polygon':[((cx+(xx-.5)*ow-x0)/wid,(cz+(zz-.5)*oh-zmin)/H) for xx,zz in shape]}
      if not admissible(uv,node):continue
      aperture=Polygon([(cx+(xx-.5)*ow,cz+(zz-.5)*oh) for xx,zz in shape]).intersection(visible)
      if aperture.area<.05:continue
      ops.extend((part,s,ix,iy) for part in _polys(aperture) if part.area>.05)
   for op in node.get('openings',[]):
    if not admissible(op,node,explicit=True):continue
    a,b,cc,dd=op['rect'];shape=aperture_shape(op.get('style',{}));aperture=Polygon([(x0+(a+xx*(cc-a))*wid,zmin+(b+zz*(dd-b))*H) for xx,zz in shape]).intersection(visible)
    ops.extend((part,op.get('style',{}),0,0) for part in _polys(aperture) if part.area>.05)
   skin=visible.difference(unary_union([o[0] for o in ops]));prism(skin,depth,.24,mat)
   # Returns at every step/outcrop are true perpendicular surfaces. Parent skin was cut away.
   return_depth=depth-node['parentDepthM']
   if abs(return_depth)>.02:
    for a,b,back_depth in return_edges(node,region,nodes,node_shapes):
     aa=np.array(a);bb=np.array(b);v=bb-aa;le=np.linalg.norm(v)
     if le<.002:continue
     # Both skins are closed .24 m slabs; the return spans only the depth they leave open.
     lo,hi=min(depth,back_depth),max(depth,back_depth)-SKIN
     if hi-lo<.002:continue
     out2=np.array([v[1],-v[0]])/le          # outward from this patch along a CCW ring
     if depth<back_depth:out2=-out2           # a recess: its side walls are seen from inside the niche
     outward=np.array([u[0]*out2[0],u[1]*out2[0],out2[1]])
     _sheet(ctx,[point(*aa,lo),point(*bb,lo),point(*bb,hi),point(*aa,hi)],outward,wallkey,mat)
   if node['parentId'] is None:
    # Close the back of the replacement box; above the host roof it is seen from behind.
    x0r,z0r,x1r,z1r=region.bounds;back=node['parentDepthM']
    _sheet(ctx,[point(x0r,z0r,back),point(x1r,z0r,back),point(x1r,z1r,back),point(x0r,z1r,back)],np.array([-n[0],-n[1],0.]),wallkey,mat)
   for op,s,ix,iy in ops:
    xa,za,xb,zb=op.bounds;ow=xb-xa;oh=zb-za;cx=(xa+xb)/2;cz=(za+zb)/2;recess=s.get('recessM',.16);fw=s.get('frameWidthM',.055);frame_mat=s.get('frameMaterial','silver');gd=depth-recess
    prism(op,gd-.35,.025,'interior',detail);prism(op,gd,.025,s.get('glassMaterial','glass'),detail)
    ring=list(orient(op,1.0).exterior.coords)
    for a,b in zip(ring,ring[1:]):
     beam([a[0],a[1],gd+.02],[b[0],b[1],gd+.02],fw,frame_mat)
     # The skin slab's own hole faces line the first .24 m; a deeper recess gets a reveal facing into the opening.
     if depth-SKIN-gd>.002:
      e=np.array(b)-np.array(a);le=np.linalg.norm(e)
      if le>.002:
       inward=-np.array([e[1],-e[0]])/le
       _sheet(ctx,[point(*a,gd),point(*b,gd),point(*b,depth-SKIN),point(*a,depth-SKIN)],np.array([u[0]*inward[0],u[1]*inward[0],inward[1]]),detail,s.get('revealMaterial',mat))
    for frac in s.get('mullions',[.5]):
     for ln in _lines(LineString([(xa+ow*frac,za),(xa+ow*frac,zb)]).intersection(op)):beam([*ln.coords[0],gd+.025],[*ln.coords[-1],gd+.025],.043,frame_mat)
    for frac in s.get('transoms',[.76]):
     for ln in _lines(LineString([(xa,za+oh*frac),(xb,za+oh*frac)]).intersection(op)):beam([*ln.coords[0],gd+.025],[*ln.coords[-1],gd+.025],.043,frame_mat)
    if s.get('awning') and (ix+2*iy)%4==1:
     aw=ow*.75;ah=min(.66,oh*.23);bot=za+oh*.47;top=bot+ah;proj=.19
     for xx in [cx-aw/2,cx+aw/2]:beam([xx,top,gd+.025],[xx,bot,gd+proj],.035,frame_mat)
     for zz,dep in [(top,gd+.025),(bot,gd+proj)]:block(cx,zz,aw,.035,.04,frame_mat,dep)
     vv=[point(cx-aw/2,top,gd+.025),point(cx+aw/2,top,gd+.025),point(cx+aw/2,bot,gd+proj),point(cx-aw/2,bot,gd+proj)];_twosided(ctx,vv,detail,'glass_light')
    if s.get('interior'):
     # Stable depth cues; these are priors and never evidence for real interiors.
     for frac in s.get('interiorSlabs',[.38]):block(cx,za+oh*frac,ow,.10,1.15,'stone',gd-.9)
    opening_count+=1
   # Metric cladding joints run within this patch and stop at every opening/child.
   joints=node.get('jointsM')
   if joints:
    cuts=[]
    for z in np.arange(zmin+joints[1],zmax,joints[1]):cuts.append(box(x0,z-.009,x1,z+.009))
    for x in np.arange(x0+joints[0],x1,joints[0]):cuts.append(box(x-.008,zmin,x+.008,zmax))
    if cuts:prism(unary_union(cuts).intersection(skin),depth+.009,.01,'joint',detail)
   for component in node.get('components',[]):
    typ=component['op'];a,b=component.get('span',[0,1]);xa=x0+a*wid;xb=x0+b*wid;zz=zmin+component.get('at',1)*H;dep=depth+component.get('depthM',0);cm=component.get('material','silver')
    if typ=='cornice':
     # Stacked section sweep. Width/depth changes produce the profile, not a painted band.
     for rise,hh,projection in component.get('section',[[0,.16,.35]]):block((xa+xb)/2,zz+rise,xb-xa,hh,projection,cm,dep+projection/2)
    elif typ=='canopy':
     projection=component.get('projectionM',1);thickness=component.get('thicknessM',.12)
     block((xa+xb)/2,zz,xb-xa,thickness,projection,cm,dep+projection/2)
     if component.get('braces'):
      for xx in [xa+.15,xb-.15]:beam([xx,zz+.75,dep],[xx,zz,dep+projection*.9],.035,cm)
    elif typ=='pilasters':
     for xx in np.linspace(xa,xb,component.get('count',6)):
      block(xx,(zmin+zmax)/2,component.get('widthM',.25),H,component.get('projectionM',.25),cm,dep+.10)
    elif typ=='dentils':
     for xx in np.arange(xa,xb,component.get('pitchM',.32)):block(xx,zz,.14,.22,.24,cm,dep+.1)
    elif typ=='fireEscape':
     # Observed platforms and alternating open stair flights are anchored to floor datums.
     zs=component['levelsM'];projection=component.get('projectionM',1.1)
     for hh in zs:
      block((xa+xb)/2,hh,xb-xa,.1,projection,cm,dep+projection/2)
      for xx in np.arange(xa,xb,.15):beam([xx,hh,dep+projection],[xx,hh+1.0,dep+projection],.022,cm)
      beam([xa,hh+1,dep+projection],[xb,hh+1,dep+projection],.035,cm)
     for i,(lo,hi) in enumerate(zip(zs,zs[1:])):
      xx1,xx2=(xa,xb) if i%2==0 else (xb,xa);steps=max(5,round((hi-lo)/.19))
      for j in range(steps):block(xx1+(xx2-xx1)*(j+.5)/steps,lo+(hi-lo)*(j+1)/steps,abs(xx2-xx1)/steps+.02,.045,projection*.75,cm,dep+projection/2)
      for d in [dep+.1,dep+projection-.1]:beam([xx1,lo+1,d],[xx2,hi+1,d],.035,cm)
    elif typ=='entrySteps':
     # A photographed raised entrance supplies rise. Ground supplies the lower support.
     top=component.get('topM',.65);run=component.get('runM',1.4);steps=component.get('count',4)
     for j in range(steps):
      h=top*(j+1)/steps;d=run*(1-(j+.5)/steps);block((xa+xb)/2,h/2,xb-xa,h+.05,run/steps+.02,cm,dep+d)
     if component.get('handrail'):
      for xx in [xa,xb]:beam([xx,1,dep+run],[xx,top+1,dep],.035,'charcoal')
    elif typ=='blade':
     projection=component.get('projectionM',1.5);raise_m=component.get('raiseM',1.3);post_pitch=component.get('postPitchM',3.5)
     block((xa+xb)/2,zz+raise_m,xb-xa,component.get('thicknessM',.13),projection,cm,dep+projection/2)
     for xx in np.linspace(xa,xb,max(2,round((xb-xa)/post_pitch)+1)):
      beam([xx,zz,dep],[xx,zz+raise_m,dep],.09,cm)
      beam([xx,zz+raise_m,dep],[xx,zz+raise_m,dep+projection*.82],.07,cm)
    elif typ=='fins':
     for hh in np.linspace(0,H,component.get('count',4)+1):block((xa+xb)/2,zmin+hh,xb-xa,.055,component.get('projectionM',.5),cm,dep+.2)
    elif typ=='frame':
     t=component.get('thicknessM',.18);d=component.get('projectionM',.3)
     for xx in [xa,xb]:block(xx,(zmin+zmax)/2,t,H,d,cm,dep+d/2)
     for hh in [zmin,zmax]:block((xa+xb)/2,hh,xb-xa,t,d,cm,dep+d/2)
    elif typ=='clock':
     radius=component.get('radiusM',2.1);cx=(xa+xb)/2
     ctx['cylinder'](point(cx,zz,dep+.06),point(cx,zz,dep+.18),radius,detail,'stone',64)
     for j in range(60):
      tick=j*math.pi/30;ro=radius*.88;ri=radius*(.72 if j%5==0 else .82)
      beam([cx+ri*math.sin(tick),zz+ri*math.cos(tick),dep+.20],[cx+ro*math.sin(tick),zz+ro*math.cos(tick),dep+.20],.06 if j%5==0 else .025,'charcoal')
     for hand,le in [(1.2,radius*.56),(-.4,radius*.75)]:beam([cx,zz,dep+.23],[cx+math.sin(hand)*le,zz+math.cos(hand)*le,dep+.23],.09,'charcoal')
    elif typ=='pyramid':
     dd=component.get('roofDepthM',wid);rise=component.get('riseM',4.5);front=dep+.35;back=dep-dd;cx=(xa+xb)/2
     vv=[point(xa-.35,zz,front),point(xb+.35,zz,front),point(xb+.35,zz,back),point(xa-.35,zz,back),point(cx,zz+rise,(front+back)/2)]
     ctx['put'](trimesh.Trimesh(vv,[[0,1,4],[1,2,4],[2,3,4],[3,0,4]],process=False),detail,cm)
     # Standing seams follow each roof face toward the apex.
     apex=[cx,zz+rise,(front+back)/2]
     for xx in np.linspace(xa,xb,max(3,round((xb-xa)/1.2))):
      beam([xx,zz+.025,front],apex,.028,cm);beam([xx,zz+.025,back],apex,.028,cm)
     ctx['cylinder'](point(cx,zz+rise,(front+back)/2),point(cx,zz+rise+component.get('poleM',5),(front+back)/2),.06,detail,'silver',8)
  # Open structures have no opaque wall behind them: braced frames, decks and lights.
  decks=[]
  for assembly in p.get('assemblies',[]):
   zlo,zhi=[value(z,r['levels']) for z in assembly['z']];xa,xb=[x*W for x in assembly.get('span',[0,1])];dep=assembly.get('depthM',-2);pitch=assembly.get('pitchM',7);xs=np.linspace(xa,xb,max(2,round((xb-xa)/pitch)+1));cm=assembly.get('material','verdigris');typ=assembly['op']
   if typ=='bracedFrame':
    for x in xs:beam([x,zlo,dep],[x,zhi,dep],.28,cm)
    for a,b in zip(xs,xs[1:]):
     beam([a,zlo,dep],[b,zhi,dep],.18,cm);beam([b,zlo,dep],[a,zhi,dep],.18,cm)
    for z in [zlo,zhi,zhi-.9]:beam([xa,z,dep],[xb,z,dep],.30,cm)
    if assembly.get('rearDepthM'):
     # Raked seating faces the interior: the back row is the frame top on the frontage and the deck steps down away
     # from it to the frame foot, carried by rakers just under the treads. A raked deck is one stepped surface: each
     # tread gets the riser up to the next, so no sky shows between treads (separate thin treads read as a floating
     # striped sheet from any distance).
     rear=assembly['rearDepthM'];back=dep-rear;count=max(1,round((zhi-zlo)/1.1));rise=(zhi-zlo)/count;run=rear/count
     for x in xs:beam([x,zhi-.6,dep],[x,zlo-.6,back],.24,cm)
     for k in range(count+1):
      z=zlo+k*rise;d=back+k*run;block((xa+xb)/2,z,xb-xa,.16,max(1.4,run+.1),'concrete',d)
      if k<count:block((xa+xb)/2,z+rise/2,xb-xa,rise+.16,.16,'concrete',d+run/2)
     decks.append((xa,xb,dep,back,zhi,zlo))
   elif typ=='lightBank':
    for fraction in assembly.get('centers',[.3,.65]):
     x=xa+fraction*(xb-xa);wid=assembly.get('widthM',14)
     for dx in [-wid*.38,wid*.38]:
      # A light mast stands on the deck it serves, not on the level datum above it.
      foot=min([zlo]+[lo+(hi-lo)*(dep-bk)/(fr-bk) for a0,b0,fr,bk,hi,lo in decks if a0<=x+dx<=b0 and bk<=dep<=fr])
      beam([x+dx,foot,dep],[x+dx,zhi,dep],.36,cm)
     for z in [zhi-2,zhi]:beam([x-wid/2,z,dep],[x+wid/2,z,dep],.22,cm)
     for dx in np.linspace(-wid/2,wid/2,14):
      for z in [zhi-1.65,zhi-.45]:block(x+dx,z,.68,.58,.3,'lamp',dep+.05)
  record={'buildingId':c['id'],'grammarId':p['id'],'status':'envelope-graph','photoColouredParts':recoloured,'sourceIds':p.get('sourceIds',[])+(['mapillary:'+ph['camera'] for ph in (photo.get('photos') or [photo]) if ph.get('camera')] if recoloured and photo else []),'midpoint':(u*(left+W/2)+n*plane).tolist(),'normal':n.tolist(),'lengthM':round(W,2),'heightM':round(max(d['rect'][3] for d in nodes),2),'streetFacing':True,'nodeCount':len(nodes),'depthPlanesM':sorted(set(d['depthM'] for d in nodes)),'levels':r['levelEvidence'],'cornerJoins':r['joins'],'nodes':[{'id':d['id'],'parent':d['parentId'],'rect':d['rect'],'depthM':d['depthM'],'material':d['material'],'surfaceEvidence':d.get('surfaceEvidence',[]),'explicitOpenings':len(d.get('openings',[]))} for d in nodes],'unresolved':p.get('unresolved',[])}
  if clip:
   # the record describes the stretch this part drew
   a_,b_=f['span'];record.update(midpoint=(u*((a_+b_)/2)+n*plane).tolist(),lengthM=round(b_-a_,2),frameLengthM=round(W,2))
   if ctx.get('set_clip'):ctx['set_clip'](None)
  records.append(record)
 return records,opening_count
