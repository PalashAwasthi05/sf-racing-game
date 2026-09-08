"""Compile observation-derived wall regions and window topology into metric geometry.
Profiles are data; no geographic IDs or addresses are interpreted here.
"""
import math
import numpy as np
from shapely.geometry import Point
from .surface_evidence import admissible,resolve_material


def openings(profile,fits):
 result=[]
 for group in profile.get('groups',[]):
  g=dict(group);fit=next((f for f in fits if f['id']==g.get('fitId')),None)
  if fit:
   region=g.get('region',fit['region']);style=dict(fit['style']);style.update(g.get('style',{}));xs=fit['columns']['centers'];ys=fit['rows']['centers'];dx=fit['columns']['spacing'];dy=fit['rows']['spacing'];cells={(round(c['x'],5),round(c['y'],5)):c['evidence'] for c in fit['cells']}
  else:
   region=g['region'];style=g.get('style',{});nx=g.get('columns',1);ny=g.get('rows',1);xs=g.get('xCenters',[(i+.5)/nx for i in range(nx)]);ys=g.get('yCenters',[(i+.5)/ny for i in range(ny)]);dx=g.get('spacingX',1/nx);dy=g.get('spacingY',1/ny);cells={}
  x0,y0,x1,y1=region
  for ix,x in enumerate(xs):
   for iy,y in enumerate(ys):
    if [ix,iy] in g.get('omit',[]):continue
    ow=dx*(x1-x0)*style.get('openingRatio',.7);oh=dy*(y1-y0)*style.get('openingHeightRatio',.7)
    cx=x0+x*(x1-x0);cy=y0+y*(y1-y0)
    result.append(dict(rect=[cx-ow/2,cy-oh/2,cx+ow/2,cy+oh/2],style=style,evidence=cells.get((round(x,5),round(y,5)),'visual-grammar' if profile.get('sourceIds') else 'inferred'),ix=ix,iy=iy,fitId=g.get('fitId')))
 result=[op for op in result if admissible(op,profile)]
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
 mid=(a+b)/2
 if float(mid@n)<outward-profile.get('planeToleranceM',2.5):return None
 low,high=sorted([float(a@u-left),float(b@u-left)])
 if W<3 or high-low<.3:return None
 plane=float(mid@n);angle=math.atan2(u[1],u[0]);layer='buildings__'+c['id']+'__observed_'+profile['id'];detail='facades__'+c['id']+'__observed_'+profile['id'];cube=ctx['cube'];mat=resolve_material(profile)
 def block(x,z,w,h,d,material,depth=0,key=detail):
  if min(w,h,d)<=.002:return
  p=u*(left+x)+n*(plane+depth);cube([*p,z0+z],[w,d,h],key,material,angle)
 ops=[]
 for op in openings(profile,fits):
  x1,y1,x2,y2=op['rect'];x1*=W;x2*=W;y1*=H;y2*=H
  if x2<low or x1>high or y2<0 or y1>H:continue
  x1=max(low,x1);x2=min(high,x2);y1=max(0,y1);y2=min(H,y2)
  if x2-x1<.12 or y2-y1<.12:continue
  ops.append({**op,'rect':[x1,y1,x2,y2]})
 panels=profile.get('panels',[])
 def wall_mat(x,z):
  found=[p for p in panels if p['rect'][0]*W<=x<=p['rect'][2]*W and p['rect'][1]*H<=z<=p['rect'][3]*H]
  return found[-1]['material'] if found else mat
 # Partition each horizontal slice around its openings and material boundaries.
 zs=sorted(set([0,H]+[v for op in ops for v in [op['rect'][1],op['rect'][3]]]+[p['rect'][k]*H for p in panels for k in [1,3]]))
 for bottom,top in zip(zs,zs[1:]):
  if top-bottom<.003:continue
  cy=(bottom+top)/2;active=[o['rect'] for o in ops if o['rect'][1]<cy<o['rect'][3]]
  xs=sorted(set([low,high]+[v for r in active for v in [r[0],r[2]]]+[np.clip(p['rect'][k]*W,low,high) for p in panels for k in [0,2]]))
  for x1,x2 in zip(xs,xs[1:]):
   cx=(x1+x2)/2
   if any(r[0]<cx<r[2] for r in active):continue
   block(cx,cy,x2-x1,top-bottom,.3,wall_mat(cx,cy),-.15,layer)
 for j,op in enumerate(ops):
  x1,y1,x2,y2=op['rect'];w=x2-x1;h=y2-y1;cx=(x1+x2)/2;cy=(y1+y2)/2;s=op['style'];kind=s.get('kind','window');fw=s.get('frameWidthM',.06);frame=s.get('frameMaterial','charcoal');proj=s.get('projectionM',.04);recess=s.get('recessM',.2)
  if kind=='projected_bay':recess-=proj
  if kind=='rollup':
   block(cx,cy,w,h,.08,'zinc',-.25)
   for zz in np.arange(y1,y2,.13):block(cx,zz,w,.017,.035,'charcoal',-.19)
   continue
  # Deep interior backing, perimeter reveals and glazing use real, distinct planes.
  block(cx,cy,w,h,.035,'interior',-recess-.18)
  glass=s.get('glassMaterial','glass')
  block(cx,cy,w-.025,h-.025,.025,glass,-recess)
  for x in [x1+fw/2,x2-fw/2]:block(x,cy,fw,h,recess+proj,frame,(proj-recess)/2)
  for y in [y1+fw/2,y2-fw/2]:block(cx,y,w,fw,recess+proj,frame,(proj-recess)/2)
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
  if kind not in ['entry','shopfront','terrace_frame'] and j%5 in [1,3]:
   ch=h*(.22 if j%5==1 else .47);block(cx,y2-ch/2-fw,w-fw*2,ch,.02,'curtain',-recess-.07)
 for band in profile.get('bands',[]):
  z=band['height']*H;block((low+high)/2,z,high-low,band.get('thicknessM',.16),band.get('depthM',.2),band.get('material','stone'),band.get('projectionM',.05))
 block((low+high)/2,.16,high-low,.32,.1,'stone',.045)
 block((low+high)/2,H+.025,high-low,.10,.4,profile.get('copingMaterial','charcoal'),.015)
 return dict(record={'buildingId':c['id'],'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(high-low,2),'heightM':round(H,2),'edge':'grammar','floors':None,'columns':None,'archetype':profile['id'],'material':mat,'sourceIds':profile['sourceIds'],'status':('image-layout-fit' if any(o.get('fitId') for o in ops) else 'visual-grammar'),'grammarId':profile['id'],'openingCount':len(ops),'openingEvidence':{'explicitBoundaries':sum(o.get('evidence')=='observed-boundary' for o in ops),'negativeRegions':len(profile.get('surfaceEvidence',[]))},'fitIds':list(set(o['fitId'] for o in ops if o.get('fitId'))),'unresolved':profile.get('unresolved',[]),'streetFacing':True},components=len(ops))
