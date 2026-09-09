"""Mapped mobility topology -> lanes, steps and barriers, with one shared surface.
No rail adjacency implies a railing. No building identity implies street furniture.
"""
import math
import numpy as np
from shapely.geometry import Point,LineString,Polygon

def number(value,default):
 try:return float(str(value).split()[0])
 except (ValueError,TypeError):return default

def spec(tags):
 lanes=max(1,int(number(tags.get('lanes'),2)));vehicle=lanes*3.15
 if tags.get('highway')=='service':vehicle=4.5
 bike={};parking={}
 for side in ['left','right']:
  b=tags.get('cycleway:'+side,tags.get('cycleway:both',tags.get('cycleway','no')))
  if tags.get('oneway')=='yes' and side=='left' and 'cycleway:'+side not in tags and tags.get('cycleway')=='lane':b='no'
  bike[side]=1.8 if b in ['lane','opposite_lane'] else 0
  parked=tags.get('parking:lane:'+side,tags.get('parking:lane:both',tags.get('parking:'+side,tags.get('parking:both','no'))))
  parking[side]=4.5 if parked=='diagonal' else 2.1 if parked in ['parallel','lane'] else 0
 return {'lanes':lanes,'vehicleWidthM':vehicle,'leftM':vehicle/2+bike['left']+parking['left'],'rightM':vehicle/2+bike['right']+parking['right'],'bike':bike,'parking':parking}

def road_polygon(g,s):return g.buffer(s['leftM'],single_sided=True,join_style=2).union(g.buffer(-s['rightM'],single_sided=True,join_style=2))

def compile(ctx):
 surface,cube,cyl,elev=ctx['surface'],ctx['cube'],ctx['cylinder'],ctx['elev'];boundary=ctx['boundary'];buildings=ctx['all_buildings'];records=[]
 def paint(line,width,color='white',offset=.145):
  surface(line.buffer(width/2,cap_style=2,join_style=2).intersection(boundary).difference(buildings),'markings',color,offset)
 def bike_symbol(line,dist):
  p=line.interpolate(dist);q=line.interpolate(min(dist+.2,line.length));u=np.array(q.coords[0])-p.coords[0];u/=max(np.linalg.norm(u),.01);n=np.array([-u[1],u[0]]);p=np.array(p.coords[0])
  def pt(x,y):return p+n*x+u*y
  for x in [-.38,.38]:paint(LineString([pt(x+.27*math.cos(a),-.15+.27*math.sin(a)) for a in np.linspace(0,2*math.pi,25)]),.065)
  for a,b in [((-.38,-.15),(0,.55)),((0,.55),(.38,-.15)),((-.38,-.15),(.38,-.15)),((0,.55),(-.12,.77)),((-.3,.77),(.1,.77)),((.38,-.15),(.28,.70)),((.28,.70),(.47,.70))]:paint(LineString([pt(*a),pt(*b)]),.06)
 # Separately mapped tracks are rendered once. Shared promenades retain their surface,
 # without inventing protected-lane symbols or barriers.
 for w in ctx['ways']:
  t,g=w['tags'],w['g']
  if g.geom_type!='LineString':continue
  if t.get('highway')=='cycleway':
   width=number(t.get('width'),3.2 if t.get('oneway')=='no' else 1.8);width=min(6,max(1.4,width));area=g.buffer(width/2,cap_style=2).intersection(boundary).difference(buildings)
   mat='concrete' if t.get('surface') in ['concrete','paving_stones','paved'] else 'asphalt'
   surface(area,'cycleways',mat,.115)
   shared=t.get('foot') in ['yes','designated'] and t.get('segregated')=='no'
   crossing=t.get('cycleway')=='crossing'
   if crossing and not shared:surface(area,'markings','greenpaint',.128)
   if not shared:
    for side in [-1,1]:
     for line in ctx['lines'](g.offset_curve(side*(width/2-.10))):paint(line,.10)
    if t.get('oneway')=='no':
     for s in np.arange(2,g.length-2,5):paint(LineString([g.interpolate(s),g.interpolate(min(s+2,g.length))]),.09,'yellow')
    for s in np.arange(min(9,g.length/2),g.length-1,32):bike_symbol(g,s)
   records.append({'rule':'mapped-cycleway','id':w['id'],'lengthM':round(g.intersection(boundary).length,2),'widthM':width,'shared':shared,'source':'OSM geometry; width default unless tagged'})
  elif t.get('highway') in ['footway','path','pedestrian'] and t.get('footway')!='crossing':
   width=number(t.get('width'),1.8);surface(g.buffer(width/2,cap_style=2).intersection(boundary).difference(buildings),'paths','concrete',.24)
  elif t.get('highway')=='steps':
   if g.length<.3:continue
   a=np.array(g.coords[0]);b=np.array(g.coords[-1]);za=float(elev(*a));zb=float(elev(*b));count=max(2,round(number(t.get('step_count'),g.length/.32)));width=number(t.get('width'),2.0)
   if abs(zb-za)<.08:
    records.append({'rule':'stairs-unresolved-rise','id':w['id'],'stepCount':count,'reason':'Ground samples cannot establish rise; no unsupported elevation invented'});continue
   if za>zb:g=LineString(list(g.coords)[::-1]);za,zb=zb,za
   for i in range(count):
    p=g.interpolate((i+.5)/count*g.length);aa=g.interpolate(i/count*g.length);bb=g.interpolate((i+1)/count*g.length);u=np.array(bb.coords[0])-aa.coords[0];h=(zb-za)*(i+1)/count;zbottom=za-.2
    cube([p.x,p.y,zbottom+(h+.2)/2],[max(.05,np.linalg.norm(u))+.01,width,h+.2],'steps','stone',math.atan2(u[1],u[0]))
   if t.get('handrail')=='yes':
    for side in [-1,1]:
     line=g.offset_curve(side*(width/2-.08));coords=list(line.coords)
     for aa,bb in zip(coords,coords[1:]):cyl([*aa,float(elev(*aa))+1.0],[*bb,float(elev(*bb))+1.0],.035,'street_objects','steel',8)
     for s in np.linspace(0,line.length,max(2,round(line.length/1.5)+1)):
      p=line.interpolate(s);z=float(elev(p.x,p.y));cyl([p.x,p.y,z],[p.x,p.y,z+1],.035,'street_objects','steel',8)
   records.append({'rule':'mapped-stairs','id':w['id'],'stepCount':count,'riseM':round(zb-za,3),'handrail':t.get('handrail')=='yes','riseSource':'terrain; individual risers inferred'})
  if t.get('barrier') not in ['fence','wall','retaining_wall','bollard']:continue
  if not g.intersects(boundary):continue
  typ=t.get('barrier');height=number(t.get('height'),1.1 if t.get('fence_type')=='railing' else 1.6);key='street_objects'
  for ln in ctx['lines'](g.intersection(boundary).difference(buildings.buffer(.01))):
   if typ in ['wall','retaining_wall']:
    for s in np.arange(0,ln.length,1):
     a=ln.interpolate(s);b=ln.interpolate(min(s+1,ln.length));v=np.array(b.coords[0])-a.coords[0];p=(np.array(a.coords[0])+b.coords[0])/2;cube([*p,float(elev(*p))+height/2],[np.linalg.norm(v),.22,height],key,'stone',math.atan2(v[1],v[0]))
   else:
    pitch=2.5 if typ=='bollard' else 1.8
    for s in np.linspace(0,ln.length,max(2,round(ln.length/pitch)+1)):
     p=ln.interpolate(s);z=float(elev(p.x,p.y));cyl([p.x,p.y,z+.1],[p.x,p.y,z+height],.09 if typ=='bollard' else .035,key,'steel',8)
    if typ=='fence':
     for hh in [.22,height]:
      for aa,bb in zip(ln.coords,list(ln.coords)[1:]):cyl([*aa,float(elev(*aa))+hh],[*bb,float(elev(*bb))+hh],.028,key,'steel',8)
   records.append({'rule':'mapped-barrier','id':w['id'],'kind':typ,'lengthM':round(ln.length,2),'heightM':height,'source':'OSM mapped feature; spacing/height inferred unless tagged'})
 for w,width,lanes in ctx['roads']:
  t,g=w['tags'],w['g'];s=ctx['road_specs'][w['id']]
  if t.get('lane_markings')!='no':
   for k in range(1,lanes):
    line=g.offset_curve((k-lanes/2)*3.15)
    for ln in ctx['lines'](line):
     for d in np.arange(6,ln.length-4,9):paint(LineString([ln.interpolate(d),ln.interpolate(min(d+3,ln.length))]),.12)
   if t.get('oneway')!='yes' and lanes>=2:paint(g,.14,'yellow')
  for side,sign in [('left',1),('right',-1)]:
   if s['bike'][side]<=0:continue
   offset=sign*(s['vehicleWidthM']/2+s['bike'][side]/2);line=g.offset_curve(offset)
   for ln in ctx['lines'](line):
    for edge in [-1,1]:
     for ee in ctx['lines'](ln.offset_curve(edge*(s['bike'][side]/2-.04))):paint(ee,.12)
    for d in np.arange(12,ln.length-4,35):bike_symbol(ln,d)
   records.append({'rule':'tagged-cycle-lane','id':w['id'],'side':side,'widthM':s['bike'][side],'source':'OSM lane side; width/paint pattern inferred'})
 ctx['counts']['mappedStairways']=sum(r['rule']=='mapped-stairs' for r in records);ctx['counts']['cyclewayFeatures']=sum(r['rule'] in ['mapped-cycleway','tagged-cycle-lane'] for r in records);ctx['counts']['mappedBarrierSections']=sum(r['rule']=='mapped-barrier' for r in records)
 return records
