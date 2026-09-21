"""Mapped mobility topology -> lanes, steps and barriers, with one shared surface.
No rail adjacency implies a railing. No building identity implies street furniture.
"""
import math
import re
import numpy as np
from shapely.geometry import Point,LineString,Polygon


_LENGTH_UNITS={'m':1.,'meter':1.,'meters':1.,'metre':1.,'metres':1.,
               'cm':.01,'mm':.001,'ft':.3048,'foot':.3048,'feet':.3048,
               'in':.0254,'inch':.0254,'inches':.0254}


def length_m(value,default=None):
 """Parse an OSM length, including feet/inches, into metres.

 OSM values without a unit are metres.  A bare feet/inches notation such as
 ``12' 6\"`` is accepted because it occurs in imported survey tags.
 """
 if value is None:return default
 if isinstance(value,(int,float)):
  return float(value) if math.isfinite(float(value)) else default
 text=str(value).strip().lower().replace(',', '.')
 if not text:return default
 # Feet/inches notation is unambiguous and needs to be handled before the
 # generic unit parser (the inch quote is otherwise easy to lose).
 foot=re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\"?)?\s*",text)
 if foot:
  result=float(foot.group(1))*.3048+float(foot.group(2) or 0)*.0254
  return result if math.isfinite(result) else default
 compound=re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*(?:ft|feet|foot)\s*(?:(\d+(?:\.\d+)?)\s*(?:in|inch|inches))?\s*',text)
 if compound:
  result=float(compound.group(1))*.3048+float(compound.group(2) or 0)*.0254
  return result if math.isfinite(result) else default
 match=re.fullmatch(r'\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*([a-z\"\']*)\s*',text)
 if not match:return default
 try:value=float(match.group(1))
 except ValueError:return default
 if not math.isfinite(value):return default
 raw_unit=match.group(2).strip().lower()
 if raw_unit in ('"','inches','inch'):unit='in'
 else:unit=raw_unit.strip('"\'') or 'm'
 if unit not in _LENGTH_UNITS:return default
 result=value*_LENGTH_UNITS[unit]
 return result if math.isfinite(result) else default


def number(value,default):
 parsed=length_m(value,None)
 return parsed if parsed is not None else default


def _tag_value(tags,keys):
 for key in keys:
  value=tags.get(key)
  if value is not None and str(value).strip() not in ('','no','none'):
   return str(value).strip()
 return None


def _colour_name(value,default=None):
 """Map explicit OSM colour names to materials present in the scene.

 The renderer's material catalogue deliberately has a small vocabulary.  An
 unknown colour is therefore ignored rather than becoming a material name
 that would fail at export.  In particular, green is never selected without
 an explicit colour tag.
 """
 if value is None:return default
 value=str(value).strip().lower()
 if not value:return default
 if value.startswith('#') and len(value) in (4,7):
  try:
   h=value[1:]
   if len(h)==3:h=''.join(c*2 for c in h)
   tuple(int(h[i:i+2],16) for i in (0,2,4))
  except ValueError:return default
  # ``stone`` is a supported recipe and safely carries an exact colour
  # variant through Materials.get. Do not turn blue paint into water.
  return f'stone#{h}'
 if 'green' in value:return 'greenpaint'
 if 'red' in value or 'maroon' in value:return 'redpaint'
 if 'yellow' in value or 'amber' in value:return 'yellow'
 if 'white' in value:return 'white'
 if value in ('black','charcoal','dark gray','dark grey'):return 'charcoal'
 if value in ('blue','navy'):return 'stone#356aaf'
 if value == 'cyan':return 'stone#35aab5'
 return default


def surface_colour(tags,side=None,default=None):
 """Return a safe material for an explicitly tagged pavement colour."""
 keys=[]
 if side:
  keys.extend([f'cycleway:{side}:surface:colour',f'cycleway:{side}:surface:color',
               f'cycleway:{side}:colour',f'cycleway:{side}:color'])
 keys.extend(['cycleway:surface:colour','cycleway:surface:color',
              'surface:colour','surface:color','colour','color'])
 return _colour_name(_tag_value(tags,keys),default)


def _cycleway_value(tags,side):
 explicit=tags.get(f'cycleway:{side}')
 if explicit is not None:return explicit
 both=tags.get('cycleway:both')
 if both is not None:return both
 value=tags.get('cycleway','no')
 # A generic lane on a one-way way is conventionally the direction-of-travel
 # side; reverse one-way ways invert that side.  Explicit side tags always win.
 if tags.get('oneway')=='yes' and side=='left':return 'no'
 if tags.get('oneway')=='-1' and side=='right':return 'no'
 return value


def _parking_value(tags,side):
 return tags.get(f'parking:lane:{side}',tags.get('parking:lane:both',
        tags.get(f'parking:{side}',tags.get('parking:both','no'))))


def _parking_width(tags,side,value):
 keys=[f'parking:lane:{side}:width',f'parking:{side}:width',
       f'parking:lane:{side}:{value}:width']
 tagged=length_m(_tag_value(tags,keys),None)
 if tagged is not None and tagged>0:return tagged
 value=str(value or 'no').lower()
 if value in ('diagonal','perpendicular','angle'):return 4.5
 if value in ('parallel','lane','marked','yes'):return 2.1
 return 0.


def _cycleway_width(tags,side,value):
 tagged=length_m(_tag_value(tags,[f'cycleway:{side}:width',
                                  'cycleway:width']),None)
 if tagged is not None and tagged>0:return tagged
 return 1.8 if value in ('lane','opposite_lane') else 0.

def spec(tags):
 lanes=max(1,int(number(tags.get('lanes'),1 if tags.get('highway') in ('service','living_street') else 2)))
 lane_width=length_m(_tag_value(tags,['lanes:width','lane_width']),None)
 vehicle=lanes*(lane_width if lane_width and lane_width>0 else 3.15)
 if tags.get('highway')=='service' and lane_width is None:vehicle=4.5
 bike={};parking={}
 for side in ['left','right']:
  b=_cycleway_value(tags,side)
  bike[side]=_cycleway_width(tags,side,b)
  parking[side]=_parking_width(tags,side,_parking_value(tags,side))
 explicit=length_m(_tag_value(tags,['width']),None)
 ancillary=sum(bike.values())+sum(parking.values())
 if explicit is not None and explicit>0:
  # OSM width describes the complete mapped way.  Keep the side features
  # inside that envelope instead of widening a way that carries an explicit
  # survey width.
  vehicle=max(.1,explicit-ancillary)
 return {'lanes':lanes,'vehicleWidthM':vehicle,'leftM':vehicle/2+bike['left']+parking['left'],'rightM':vehicle/2+bike['right']+parking['right'],'bike':bike,'parking':parking,
         'widthM':vehicle+ancillary,'laneWidthM':vehicle/lanes}

def road_polygon(g,s):
 """Build a repaired two-sided carriageway following curved centerlines."""
 if g.is_empty:return g
 area=g.buffer(s['leftM'],single_sided=True,join_style=2).union(g.buffer(-s['rightM'],single_sided=True,join_style=2))
 return area.buffer(0)

def compile(ctx):
 surface,cube,cyl,elev=ctx['surface'],ctx['cube'],ctx['cylinder'],ctx['elev'];boundary=ctx['boundary'];buildings=ctx['all_buildings'];records=[]
 clear=boundary.difference(buildings)
 street_plan=ctx.get('street_plan') or {}
 road_mask=street_plan.get('road')
 if road_mask is None or road_mask.is_empty:road_mask=ctx.get('road_area')
 cycle_mask=street_plan.get('cycle')
 sidewalk_mask=street_plan.get('sidewalk')
 path_excluded=[m for m in (road_mask,cycle_mask,sidewalk_mask) if m is not None and not m.is_empty]
 def paint(line,width,color='white',offset=.145,mask=None):
  if line is None or line.is_empty:return
  area=line.buffer(width/2,cap_style=2,join_style=2).intersection(clear)
  if mask is not None:area=area.intersection(mask)
  if not area.is_empty:surface(area,'markings',color,offset)
 def bike_symbol(line,dist,mask=None):
  p=line.interpolate(dist);q=line.interpolate(min(dist+.2,line.length));u=np.array(q.coords[0])-p.coords[0];u/=max(np.linalg.norm(u),.01);n=np.array([-u[1],u[0]]);p=np.array(p.coords[0])
  def pt(x,y):return p+n*x+u*y
  for x in [-.38,.38]:paint(LineString([pt(x+.27*math.cos(a),-.15+.27*math.sin(a)) for a in np.linspace(0,2*math.pi,25)]),.065,mask=mask)
  for a,b in [((-.38,-.15),(0,.55)),((0,.55),(.38,-.15)),((-.38,-.15),(.38,-.15)),((0,.55),(-.12,.77)),((-.3,.77),(.1,.77)),((.38,-.15),(.28,.70)),((.28,.70),(.47,.70))]:paint(LineString([pt(*a),pt(*b)]),.06,mask=mask)
 # Separately mapped tracks are rendered once. Shared promenades retain their surface,
 # without inventing protected-lane symbols or barriers.
 for w in ctx['ways']:
  t,g=w['tags'],w['g']
  if g.geom_type!='LineString':continue
  if t.get('highway')=='cycleway':
   width=length_m(t.get('width'),3.2 if t.get('oneway')=='no' else 1.8);width=min(12,max(.5,width));area=g.buffer(width/2,cap_style=2).intersection(clear)
   if cycle_mask is not None and not cycle_mask.is_empty:area=area.intersection(cycle_mask)
   # "Paved" establishes a hard surface, not its substrate. A crossing or
   # generic paved track must not become a bright concrete ribbon in asphalt.
   mat='concrete' if t.get('surface') in ['concrete','concrete:plates'] else 'stone' if t.get('surface')=='paving_stones' else 'asphalt'
   colour=surface_colour(t,default=None)
   if not area.is_empty:surface(area,'cycleways',colour or mat,.115)
   shared=t.get('foot') in ['yes','designated'] and t.get('segregated')=='no'
   crossing=t.get('cycleway')=='crossing'
   # Crossings and cycle lanes are green only when their source tags say so.
   if crossing and not shared and colour and not area.is_empty:surface(area,'markings',colour,.128)
   if not shared and not crossing:
    for side in [-1,1]:
     for line in ctx['lines'](g.offset_curve(side*(width/2-.10))):paint(line,.10,mask=area)
    if t.get('oneway')=='no':
     for s in np.arange(2,g.length-2,5):paint(LineString([g.interpolate(s),g.interpolate(min(s+2,g.length))]),.09,'yellow',mask=area)
    for s in np.arange(min(9,g.length/2),g.length-1,32):bike_symbol(g,s,mask=area)
   records.append({'rule':'mapped-cycleway','id':w['id'],'lengthM':round(g.intersection(boundary).length,2),'widthM':width,'shared':shared,'surfaceColour':colour,'source':'OSM geometry; width default unless tagged'})
  elif t.get('highway') in ['footway','path','pedestrian'] and t.get('footway')!='crossing':
   width=length_m(t.get('width'),1.8);width=min(12,max(.3,width));path_area=g.buffer(width/2,cap_style=2).intersection(clear)
   for owned in path_excluded:path_area=path_area.difference(owned)
   if not path_area.is_empty:surface(path_area,'paths',surface_colour(t,default='concrete'),.24)
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
  typ=t.get('barrier');height=length_m(t.get('height'),.95 if typ=='bollard' else 1.1 if t.get('fence_type')=='railing' else 1.6);key='street_objects'
  placement=None
  if typ=='bollard' and street_plan:
   space=ctx.get('street_space')
   if space is None:
    try:
     from . import street_space as space
    except ImportError:
     space=None
   placement=(getattr(space,'place',None) if space is not None and
              all(k in street_plan for k in ('forbidden','occupied','records')) else None)
  barrier_line=g.intersection(boundary).difference(buildings.buffer(.01))
  if street_plan:
   for name in ['road','cycle','crossings']:
    if name in street_plan:barrier_line=barrier_line.difference(street_plan[name].buffer(.08))
  for ln in ctx['lines'](barrier_line):
   if ln.is_empty or ln.length<.01:continue
   if typ in ['wall','retaining_wall']:
    for s in np.arange(0,ln.length,1):
     a=ln.interpolate(s);b=ln.interpolate(min(s+1,ln.length));v=np.array(b.coords[0])-a.coords[0];p=(np.array(a.coords[0])+b.coords[0])/2;cube([*p,float(elev(*p))+height/2],[np.linalg.norm(v),.22,height],key,'stone',math.atan2(v[1],v[0]))
   else:
    pitch=2.5 if typ=='bollard' else 1.8
    placed=0
    for index,s in enumerate(np.linspace(0,ln.length,max(2,round(ln.length/pitch)+1))):
     p=ln.interpolate(s)
     if typ=='bollard' and placement is not None:
      pos=placement(street_plan,(p.x,p.y),.09,boundary,kind='mapped-bollard',source_id=f"{w['id']}:{index}",max_shift=.5)
      if pos is None:continue
      x,y=pos
     else:x,y=p.x,p.y
     z=float(elev(x,y));cyl([x,y,z+.1],[x,y,z+height],.09 if typ=='bollard' else .035,key,'charcoal' if typ=='bollard' else 'steel',8);placed+=1
     if typ=='bollard':ctx['counts']['bollards']=ctx['counts'].get('bollards',0)+1
    if typ=='fence':
     for hh in [.22,height]:
      for aa,bb in zip(ln.coords,list(ln.coords)[1:]):cyl([*aa,float(elev(*aa))+hh],[*bb,float(elev(*bb))+hh],.028,key,'steel',8)
   records.append({'rule':'mapped-barrier','id':w['id'],'kind':typ,'lengthM':round(ln.length,2),'heightM':height,'source':'OSM mapped feature; spacing/height inferred unless tagged'})
 for w,width,lanes in ctx['roads']:
  t,g=w['tags'],w['g'];s=ctx['road_specs'][w['id']];lane_width=s.get('laneWidthM',s['vehicleWidthM']/max(1,lanes))
  road_shape=road_polygon(g,s).intersection(clear)
  mark_mask=road_mask if road_mask is not None else road_shape
  if road_mask is not None:road_shape=road_shape.intersection(road_mask)
  road_colour=surface_colour(t,default=None)
  if road_colour and not road_shape.is_empty:surface(road_shape,'roads',road_colour,.092)
  if t.get('lane_markings')!='no':
   for k in range(1,lanes):
    line=g.offset_curve((k-lanes/2)*lane_width)
    for ln in ctx['lines'](line):
     for d in np.arange(6,ln.length-4,9):paint(LineString([ln.interpolate(d),ln.interpolate(min(d+3,ln.length))]),.12,mask=mark_mask)
   if t.get('oneway') not in ('yes','-1') and lanes>=2:paint(g,.14,'yellow',mask=mark_mask)
  for side,sign in [('left',1),('right',-1)]:
   if s['bike'][side]<=0:continue
   bike_width=s['bike'][side];offset=sign*(s['vehicleWidthM']/2+bike_width/2);line=g.offset_curve(offset)
   lane_colour=surface_colour(t,side,default=None)
   if lane_colour:
    lane_area=line.buffer(bike_width/2,cap_style=2,join_style=2).intersection(clear)
    lane_area=lane_area.intersection(mark_mask)
    if not lane_area.is_empty:surface(lane_area,'cycleways',lane_colour,.11)
   for ln in ctx['lines'](line):
    lane_mask=ln.buffer(bike_width/2,cap_style=2,join_style=2).intersection(mark_mask)
    for edge in [-1,1]:
     for ee in ctx['lines'](ln.offset_curve(edge*(bike_width/2-.04))):paint(ee,.12,mask=lane_mask)
    for d in np.arange(12,ln.length-4,35):bike_symbol(ln,d,mask=lane_mask)
   records.append({'rule':'tagged-cycle-lane','id':w['id'],'side':side,'widthM':bike_width,'surfaceColour':lane_colour,'source':'OSM lane side; width/paint pattern inferred'})
 ctx['counts']['mappedStairways']=sum(r['rule']=='mapped-stairs' for r in records);ctx['counts']['cyclewayFeatures']=sum(r['rule'] in ['mapped-cycleway','tagged-cycle-lane'] for r in records);ctx['counts']['mappedBarrierSections']=sum(r['rule']=='mapped-barrier' for r in records)
 return records
