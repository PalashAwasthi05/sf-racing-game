"""Street detail from topology, ROW and inventories; inferred positions remain explicit."""
import math,re
import numpy as np
import trimesh
from shapely.geometry import Point,LineString,box
from shapely.ops import unary_union,nearest_points

def compile(ctx):
 surface,cube,cyl=ctx['surface'],ctx['cube'],ctx['cylinder'];boundary=ctx['boundary'];roads=ctx['roads'];elev=ctx['elev'];records=[]
 road=ctx['road_area']
 cycle=unary_union([w['g'].buffer(1.7) for w in ctx['ways'] if w['tags'].get('highway')=='cycleway' and w['g'].geom_type=='LineString']).intersection(boundary)
 crossings=unary_union([w['g'].buffer(1.8) for w in ctx['footpaths'] if w['tags'].get('footway')=='crossing'])
 rows=unary_union([ctx['geom'](r['the_geom']) for r in ctx['json'].load(open(ctx['DATA']/'row.json'))])
 sidewalks=road.buffer(3.2).intersection(rows.buffer(.5)).difference(road.buffer(.02)).difference(ctx['all_buildings'].buffer(.02)).difference(cycle).intersection(boundary)
 pits=[]
 for t in ctx['json'].load(open(ctx['DATA']/'trees.json')):
  try:x,y=ctx['local'](float(t['longitude']),float(t['latitude']))
  except:continue
  if Point(x,y).within(sidewalks):pits.append(box(x-.65,y-.65,x+.65,y+.65))
 if pits:sidewalks=sidewalks.difference(unary_union(pits))
 surface(sidewalks,'sidewalks','concrete',.26)
 for p in pits:surface(p,'landscape','soil',.095)
 curb_lines=road.boundary.intersection(boundary.buffer(-1)).difference(crossings)
 for line in ctx['lines'](curb_lines):
  # Short terrain-conforming strips avoid metre-long floating curb slabs.
  for ss in np.arange(0,line.length,.65):
   seg=LineString([line.interpolate(ss),line.interpolate(min(ss+.65,line.length))])
   surface(seg.buffer(.09,cap_style=2),'sidewalks','concrete',.26)
   for side in [-1,1]:
    ln=seg.offset_curve(side*.09);a,b=np.array(ln.coords[0]),np.array(ln.coords[-1]);za,zb=float(elev(*a)),float(elev(*b))
    vv=[[a[0],a[1],za+.09],[b[0],b[1],zb+.09],[b[0],b[1],zb+.26],[a[0],a[1],za+.26]]
    ctx['put'](trimesh.Trimesh(vv,[[0,1,2],[0,2,3]],process=False),'sidewalks','concrete')
 # Score joints in the paving; geometry stays local and follows each street orientation.
 for w,width,lanes in roads:
  g=w['g']
  for side in [-1,1]:
   for line in ctx['lines'](g.offset_curve(side*(width/2+1.6)).intersection(boundary)):
    for ss in np.arange(1.2,line.length,1.8):
     p=line.interpolate(ss);p1=line.interpolate(max(0,ss-.1));p2=line.interpolate(min(line.length,ss+.1));v=np.array([p2.x-p1.x,p2.y-p1.y]);le=np.linalg.norm(v)
     if le<.01:continue
     u=v/le;n=np.array([u[1],-u[0]]);a=np.array([p.x,p.y])-n*1.6;b=np.array([p.x,p.y])+n*1.6
     for joint in ctx['lines'](LineString([a,b]).intersection(sidewalks)):
      surface(joint.buffer(.0075,cap_style=2),'sidewalks','joint',.269)
  if w['tags'].get('busway:right')=='lane':
   surface(g.offset_curve(-(width/2-1.6)).buffer(1.5).intersection(road),'markings','redpaint',.103)
  # Render direction arrows only when lane-turn semantics exist.
  turn=w['tags'].get('turn:lanes','')
  if turn and g.length>22:
   parts=turn.split('|')
   for k,value in enumerate(parts):
    if not value:continue
    ln=g.offset_curve((lanes/2-k-.5)*3.15);p=ln.interpolate(max(5,ln.length-13));q=ln.interpolate(max(4,ln.length-14));v=np.array([p.x-q.x,p.y-q.y]);u=v/max(.01,np.linalg.norm(v));n=np.array([-u[1],u[0]]);center=np.array([p.x,p.y]);z=float(elev(*center))+.15
    if 'through' in value or 'right' in value or 'left' in value:
     a=center-u*1.5;b=center+u*.8
     cube([*(a+b)/2,z],[2.3,.17,.018],'markings','white',math.atan2(u[1],u[0]))
     tip=center+u*1.8
     if 'right' in value:tip=center-n*1.3
     if 'left' in value:tip=center+n*1.3
     for side in [-1,1]:
      tail=tip-u*.7+n*.5*side;vec=tip-tail;cube([*(tip+tail)/2,z],[np.linalg.norm(vec),.19,.018],'markings','white',math.atan2(vec[1],vec[0]))
 # Separately mapped cycleways are owned by mobility.compile.
 # Tactile panels at mapped crossing endpoints, with detectable raised studs.
 for w in ctx['footpaths']:
  if w['tags'].get('footway')!='crossing':continue
  for ln in ctx['lines'](w['g'].intersection(boundary.buffer(-2))):
   if ln.length>45:continue
   for p,q in [(np.array(ln.coords[0]),np.array(ln.coords[1])),(np.array(ln.coords[-1]),np.array(ln.coords[-2]))]:
    u=(q-p)/max(.01,np.linalg.norm(q-p));n=np.array([-u[1],u[0]]);angle=math.atan2(n[1],n[0]);z=float(elev(*p))+.275
    cube([*p,z],[1.6,.6,.035],'sidewalks','yellow',angle)
    for xx in np.arange(-.7,.75,.1):
     for yy in [-.2,-.1,0,.1,.2]:
      s=p+n*xx+u*yy;cyl([*s,z+.02],[*s,z+.029],.019,'sidewalks','yellow',6)
 # Inventory contains many co-located records. Deduplicate types at each candidate location,
 # snap cautiously to a nearby inferred curb, retain displacement and uncertainty.
 used=set();signs=ctx['json'].load(open(ctx['DATA']/'signs.json'));signcount=0
 for item in signs:
  legend=item.get('legend','').upper()
  if not any(s in legend for s in ['NO PARKING','NO STOPPING','SPEED LIMIT','ONE WAY','DO NOT ENTER','STOP','STREET NAME']):continue
  try:p=ctx['geom'](item['shape'])
  except:continue
  if p.distance(Point(0,0))>ctx['CONFIG']['region']['studyRadiusM']:continue
  family='STOP' if legend=='STOP' else 'ONE WAY' if 'ONE WAY' in legend else 'NO PARKING' if 'PARKING' in legend else 'NO STOPPING' if 'STOPPING' in legend else 'DO NOT ENTER' if 'DO NOT ENTER' in legend else 'SPEED LIMIT' if 'SPEED LIMIT' in legend else 'STREET'
  key=(round(p.x/7),round(p.y/7),family)
  if key in used:continue
  used.add(key);curb=nearest_points(p,curb_lines)[1]
  if p.distance(curb)>12:continue
  nearest=min(roads,key=lambda r:r[0]['g'].distance(curb));line=nearest[0]['g'];s=line.project(curb);a=line.interpolate(max(0,s-.2));b=line.interpolate(min(line.length,s+.2));u=np.array([b.x-a.x,b.y-a.y]);u=u/max(np.linalg.norm(u),.01);n=np.array([u[1],-u[0]]);pos=np.array([curb.x,curb.y]);toward=np.array([p.x,p.y])-pos
  if np.dot(toward,n)<0:n=-n
  pos+=n*.3
  facing=item.get('corner_side_facing','').upper();dirs={'NORTHBOUND':np.array([0.,-1.]),'SOUTHBOUND':np.array([0.,1.]),'EASTBOUND':np.array([-1.,0.]),'WESTBOUND':np.array([1.,0.])};normal=next((v for k,v in dirs.items() if k in facing),-n);angle=math.atan2(normal[0],-normal[1]);z=float(elev(*pos))+.26
  cyl([*pos,z],[*pos,z+2.9],.032,'signs','steel',8)
  signid='sign_'+str(item.get('asset_id'));ctx['make_sign'](pos,z+2.5,angle,family,legend,signid)
  for h in [.35,1.4]:cube([*pos,z+h],[.085,.055,.025],'signs','steel',angle)
  records.append({'id':signid,'legend':legend,'family':family,'position':pos.tolist(),'sourceLocation':[p.x,p.y],'snapDistanceM':round(p.distance(curb),2),'orientationText':facing,'positionStatus':'inferred-curb-snap','source':'DataSF sign inventory','observedAt':item.get('last_edited_date'),'imageVerified':False});signcount+=1
 ctx['counts']['signPanels']=signcount;ctx['counts']['sidewalkAreaM2']=round(sidewalks.area)
 return records,sidewalks
