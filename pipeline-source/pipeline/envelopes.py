"""Evidence-driven envelope graph -> articulated surfaces, returns and assemblies.
Geographic matching and observations live in JSON. Operators work in facade meters.
No address-, landmark- or architectural-style-specific code is permitted here.
"""
import math,re,copy
import numpy as np
import trimesh
from shapely.geometry import Polygon,Point,box
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter1d
from .surface_evidence import admissible


def matches(c,profile):
 m=profile['match']
 return (m.get('featureId')==c['id']) or (c.get('osm') and str(m.get('osmWayId'))==str(c['osm']['id']))


def frame(c,profile):
 n=np.array(profile['normal'],float);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]])
 if profile.get('referenceLine'):
  a,b=np.array(profile['referenceLine'],float);left,right=sorted([a@u,b@u]);plane=float(((a+b)/2)@n)
 else:
  edges=[]
  for p in _polys(c['g']):
   a=np.array(p.exterior.coords)
   for pa,pb in zip(a,a[1:]):
    v=pb-pa;le=np.linalg.norm(v)
    if le>2 and abs(v/le@u)>.96:edges.append((le,float((pa+pb)@n/2),pa,pb))
  if not edges:raise ValueError('No compatible footprint frontage: '+profile['id'])
  plane=max(e[1] for e in edges);es=[e for e in edges if e[1]>plane-profile.get('planeToleranceM',3)]
  points=np.concatenate([np.array([e[2],e[3]]) for e in es]);left=float(min(points@u));right=float(max(points@u))
 return {'normal':n,'tangent':u,'left':left,'width':right-left,'plane':plane}


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
  d['material']=d.get('material',parent.get('material','stucco') if parent else 'stucco');d.setdefault('depthM',0)
  nodes.append(d)
  for ch in d.get('children',[]):visit(ch,d)
 for node in profile['nodes']:visit(node)
 return nodes


def prepare(ctx,c,z0,samples,profiles):
 resolved=[]
 for p in profiles:
  if not matches(c,p):continue
  f=frame(c,p);levels,ev=fit_levels(p,f,samples,z0)
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
 return resolved


def replacement_mask(resolved):
 masks=[]
 for r in resolved:
  f=r['frame'];p=r['profile'];u=f['tangent'];n=f['normal'];left=f['left'];right=left+f['width'];plane=f['plane'];depth=p.get('replaceDepthM',10)
  masks.append(Polygon([u*x+n*y for x,y in [(left-.05,plane+3),(right+.05,plane+3),(right+.05,plane-depth),(left-.05,plane-depth)]]))
 return unary_union(masks)


def compile(ctx,c,z0,resolved):
 records=[];opening_count=0
 for r in resolved:
  p=r['profile'];f=r['frame'];u,n=f['tangent'],f['normal'];W,left,plane=f['width'],f['left'],f['plane'];angle=math.atan2(u[1],u[0]);nodes=r['nodes'];prefix='__'+c['id']+'__observed_'+p['id'];wallkey='buildings'+prefix;detail='facades'+prefix
  def point(x,z,depth):return np.r_[u*(left+x)+n*(plane+depth),z0+z]
  def block(x,z,w,h,d,mat,depth=0,key=detail):
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
  for node in nodes:
   region=node_shapes[node['id']];x0,zmin,x1,zmax=region.bounds;wid=x1-x0;H=zmax-zmin;depth=node['depthM'];mat=node['material'];children=[node_shapes[d['id']] for d in nodes if d['parentId']==node['id']]
   visible=region.difference(unary_union(children));ops=[]
   for group in node.get('grids',[]):
    a,b,cc,dd=group.get('rect',[0,0,1,1]);nx,ny=group.get('columns',1),group.get('rows',1);s=group.get('style',{});xs=group.get('xCenters',[(j+.5)/nx for j in range(nx)]);ys=group.get('yCenters',[(j+.5)/ny for j in range(ny)])
    for ix,x in enumerate(xs):
     for iy,y in enumerate(ys):
      if [ix,iy] in group.get('omit',[]):continue
      ow=wid*(cc-a)/nx*s.get('widthRatio',.75);oh=H*(dd-b)/ny*s.get('heightRatio',.78);cx=x0+(a+x*(cc-a))*wid;cz=zmin+(b+y*(dd-b))*H
      shape=s.get('shape',[[0,0],[1,0],[1,1],[0,1]])
      uv={'polygon':[((cx+(xx-.5)*ow-x0)/wid,(cz+(zz-.5)*oh-zmin)/H) for xx,zz in shape]}
      if not admissible(uv,node):continue
      aperture=Polygon([(cx+(xx-.5)*ow,cz+(zz-.5)*oh) for xx,zz in shape]).intersection(visible)
      if aperture.area<.05:continue
      ops.extend((part,s,ix,iy) for part in _polys(aperture) if part.area>.05)
   for op in node.get('openings',[]):
    if not admissible(op,node,explicit=True):continue
    a,b,cc,dd=op['rect'];aperture=box(x0+a*wid,zmin+b*H,x0+cc*wid,zmin+dd*H).intersection(visible)
    ops.extend((part,op.get('style',{}),0,0) for part in _polys(aperture) if part.area>.05)
   skin=visible.difference(unary_union([o[0] for o in ops]));prism(skin,depth,.24,mat)
   # Returns at every step/outcrop are true perpendicular surfaces. Parent skin was cut away.
   return_depth=depth-node['parentDepthM']
   if abs(return_depth)>.02:
    for a,b in zip(region.exterior.coords,list(region.exterior.coords)[1:]):
     aa=np.array(a);bb=np.array(b);v=bb-aa
     if np.linalg.norm(v)<.002:continue
     verts=[point(*aa,depth),point(*bb,depth),point(*bb,node['parentDepthM']),point(*aa,node['parentDepthM'])]
     mesh=trimesh.Trimesh(verts,[[0,1,2],[0,2,3]],process=False);ctx['put'](mesh,wallkey,mat)
   for op,s,ix,iy in ops:
    xa,za,xb,zb=op.bounds;ow=xb-xa;oh=zb-za;cx=(xa+xb)/2;cz=(za+zb)/2;recess=s.get('recessM',.16);fw=s.get('frameWidthM',.055);frame_mat=s.get('frameMaterial','silver');gd=depth-recess
    prism(op,gd-.35,.025,'interior',detail);prism(op,gd,.025,s.get('glassMaterial','glass'),detail)
    for a,b in zip(op.exterior.coords,list(op.exterior.coords)[1:]):
     beam([a[0],a[1],gd+.02],[b[0],b[1],gd+.02],fw,frame_mat)
     verts=[point(*a,depth),point(*b,depth),point(*b,gd),point(*a,gd)];ctx['put'](trimesh.Trimesh(verts,[[0,1,2],[0,2,3]],process=False),detail,s.get('revealMaterial',mat))
    for frac in s.get('mullions',[.5]):block(xa+ow*frac,cz,.043,oh,.055,frame_mat,gd+.025)
    for frac in s.get('transoms',[.76]):block(cx,za+oh*frac,ow,.043,.055,frame_mat,gd+.025)
    if s.get('awning') and (ix+2*iy)%4==1:
     aw=ow*.75;ah=min(.66,oh*.23);bot=za+oh*.47;top=bot+ah;proj=.19
     for xx in [cx-aw/2,cx+aw/2]:beam([xx,top,gd+.025],[xx,bot,gd+proj],.035,frame_mat)
     for zz,dep in [(top,gd+.025),(bot,gd+proj)]:block(cx,zz,aw,.035,.04,frame_mat,dep)
     vv=[point(cx-aw/2,top,gd+.025),point(cx+aw/2,top,gd+.025),point(cx+aw/2,bot,gd+proj),point(cx-aw/2,bot,gd+proj)];ctx['put'](trimesh.Trimesh(vv,[[0,1,2],[0,2,3]],process=False),detail,'glass_light')
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
      angle=j*math.pi/30;ro=radius*.88;ri=radius*(.72 if j%5==0 else .82)
      beam([cx+ri*math.sin(angle),zz+ri*math.cos(angle),dep+.20],[cx+ro*math.sin(angle),zz+ro*math.cos(angle),dep+.20],.06 if j%5==0 else .025,'charcoal')
     for angle,le in [(1.2,radius*.56),(-.4,radius*.75)]:beam([cx,zz,dep+.23],[cx+math.sin(angle)*le,zz+math.cos(angle)*le,dep+.23],.09,'charcoal')
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
  for assembly in p.get('assemblies',[]):
   zlo,zhi=[value(z,r['levels']) for z in assembly['z']];xa,xb=[x*W for x in assembly.get('span',[0,1])];dep=assembly.get('depthM',-2);pitch=assembly.get('pitchM',7);xs=np.linspace(xa,xb,max(2,round((xb-xa)/pitch)+1));cm=assembly.get('material','verdigris');typ=assembly['op']
   if typ=='bracedFrame':
    for x in xs:beam([x,zlo,dep],[x,zhi,dep],.28,cm)
    for a,b in zip(xs,xs[1:]):
     beam([a,zlo,dep],[b,zhi,dep],.18,cm);beam([b,zlo,dep],[a,zhi,dep],.18,cm)
    for z in [zlo,zhi,zhi-.9]:beam([xa,z,dep],[xb,z,dep],.30,cm)
    if assembly.get('rearDepthM'):
     back=dep-assembly['rearDepthM']
     for x in xs:beam([x,zlo,dep],[x,zhi-1,back],.24,cm)
     for z in np.arange(zlo,zhi,1.1):
      d=dep-(z-zlo)/(zhi-zlo)*assembly['rearDepthM'];block((xa+xb)/2,z,xb-xa,.16,1.4,'concrete',d)
   elif typ=='lightBank':
    for fraction in assembly.get('centers',[.3,.65]):
     x=xa+fraction*(xb-xa);wid=assembly.get('widthM',14)
     for dx in [-wid*.38,wid*.38]:beam([x+dx,zlo,dep],[x+dx,zhi,dep],.36,cm)
     for z in [zhi-2,zhi]:beam([x-wid/2,z,dep],[x+wid/2,z,dep],.22,cm)
     for dx in np.linspace(-wid/2,wid/2,14):
      for z in [zhi-1.65,zhi-.45]:block(x+dx,z,.68,.58,.3,'lamp',dep+.05)
  record={'buildingId':c['id'],'grammarId':p['id'],'status':'envelope-graph','sourceIds':p.get('sourceIds',[]),'midpoint':(u*(left+W/2)+n*plane).tolist(),'normal':n.tolist(),'lengthM':round(W,2),'heightM':round(max(d['rect'][3] for d in nodes),2),'streetFacing':True,'nodeCount':len(nodes),'depthPlanesM':sorted(set(d['depthM'] for d in nodes)),'levels':r['levelEvidence'],'cornerJoins':r['joins'],'nodes':[{'id':d['id'],'parent':d['parentId'],'rect':d['rect'],'depthM':d['depthM'],'material':d['material'],'surfaceEvidence':d.get('surfaceEvidence',[]),'explicitOpenings':len(d.get('openings',[]))} for d in nodes],'unresolved':p.get('unresolved',[])}
  records.append(record)
 return records,opening_count
