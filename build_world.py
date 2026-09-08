"""King Street pilot: geospatial inputs -> metric scene, evidence manifest and GLB."""
import json, math, pathlib, collections, xml.etree.ElementTree as ET
import numpy as np
import trimesh
from shapely.geometry import shape, Polygon, Point, LineString, box
from shapely.ops import transform, unary_union
from shapely import contains_xy, make_valid
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, gaussian_filter, distance_transform_edt
from rasterio.features import shapes
from rasterio.transform import from_origin

ROOT=pathlib.Path(__file__).parent;DATA=ROOT/'data';PUBLIC=ROOT/'king-street'/'public';PUBLIC.mkdir(exist_ok=True)
OUT=ROOT.parent/'outputs';OUT.mkdir(exist_ok=True)
ll=Transformer.from_crs(4326,32610,always_xy=True);inv=Transformer.from_crs(32610,4326,always_xy=True)
web=Transformer.from_crs(3857,32610,always_xy=True)
osm=ET.parse(DATA/'osm.xml').getroot()
nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in osm.findall('node')}
raw_ways=[]
for w in osm.findall('way'):
 tags={a.attrib['k']:a.attrib['v'] for a in w.findall('tag')}
 pts=[nodes[n.attrib['ref']] for n in w.findall('nd') if n.attrib['ref'] in nodes]
 if len(pts)>1:raw_ways.append((w.attrib['id'],tags,pts))
target=next(w for w in raw_ways if w[0]=='149167949')
target_poly=transform(ll.transform,Polygon(target[2]));cx,cy=target_poly.centroid.coords[0];lon,lat=inv.transform(cx,cy)
def local(x,y,z=None):
 a,b=ll.transform(x,y);return np.asarray(a)-cx,np.asarray(b)-cy
def geom(g):return transform(local,shape(g))
def polys(g):
 if g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 if hasattr(g,'geoms'):return [p for a in g.geoms for p in polys(a)]
 return []
def lines(g):
 if g.is_empty:return []
 if g.geom_type=='LineString':return [g]
 if hasattr(g,'geoms'):return [p for a in g.geoms for p in lines(a)]
 return []
boundary=Point(0,0).buffer(265,quad_segs=96);study=Point(0,0).buffer(200)
ways=[]
for id,t,p in raw_ways:
 x,y=local(*zip(*p));coords=list(zip(x,y))
 g=Polygon(coords) if coords[0]==coords[-1] and len(coords)>=4 else LineString(coords)
 if not g.is_valid:g=make_valid(g)
 if g.intersects(boundary):ways.append({'id':id,'tags':t,'g':g})

pts=np.load(DATA/'points.npz')['points'];px,py=web.transform(pts[:,0],pts[:,1]);pts[:,0]=px-cx;pts[:,1]=py-cy
ground=pts[pts[:,3]==2];ground=ground[(ground[:,2]>-5)&(ground[:,2]<20)]
base=float(np.median(ground[np.linalg.norm(ground[:,:2],axis=1)<40,2]));pts[:,2]-=base;ground[:,2]-=base
tree=cKDTree(ground[:,:2]);N=179;STEP=3.;MIN=-267.
gx,gy=np.meshgrid(np.arange(N)*STEP+MIN,np.arange(N)*STEP+MIN)
dist,idx=tree.query(np.c_[gx.ravel(),gy.ravel()],k=8)
grid=np.median(ground[idx,2],axis=1).reshape(N,N);grid=gaussian_filter(grid,.8)
def elev(x,y):
 ix=np.clip(np.rint((np.asarray(x)-MIN)/STEP).astype(int),0,N-1);iy=np.clip(np.rint((np.asarray(y)-MIN)/STEP).astype(int),0,N-1)
 return grid[iy,ix]

scene=trimesh.Scene();features=[];batches=collections.defaultdict(lambda:[[],[],[]]);counts=collections.Counter();materials={}
palette={'asphalt':'#41484b','sidewalk':'#b4b5ac','brick':'#97604b','brick2':'#b4795d','stone':'#c5beb0','dark':'#283c42','glass':'#4b7686','frame':'#c4d0d1','roof':'#737b79','grass':'#566e4b','trunk':'#75644d','leaf':'#425e48','metal':'#546661','paint':'#e4e5d8','yellow':'#d7b351','rail':'#87918e','water':'#386c7a'}
def rgba(c):
 c=palette.get(c,c).lstrip('#');return [int(c[i:i+2],16) for i in (0,2,4)]+[255]
def put(mesh,key,color):
 if len(mesh.vertices)==0:return
 vertices,faces,colors=batches[key];off=sum(len(v) for v in vertices)
 vertices.append(np.asarray(mesh.vertices));faces.append(np.asarray(mesh.faces)+off);colors.append(np.tile(rgba(color),(len(mesh.vertices),1)))
def cube(center,size,key,color,angle=0):
 sx,sy,sz=size
 if min(sx,sy,sz)<=0:return
 v=np.array([[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]])*size
 if angle:
  c,s=math.cos(angle),math.sin(angle);v[:,:2]=v[:,:2]@np.array([[c,s],[-s,c]])
 v+=center
 f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
 put(trimesh.Trimesh(v,f,process=False),key,color)
def extrude(p,z,h,key,color):
 for a in polys(p):
  if a.area<.05:continue
  try:
   m=trimesh.creation.extrude_polygon(a,height=max(h,.02),engine='earcut');m.apply_translation([0,0,z]);put(m,key,color)
  except Exception as e:print('polygon failed',key,str(e)[:70])
def surface(p,key,color,offset=.03):
 for a in polys(p):
  if a.area<.02:continue
  try:
   v,f=trimesh.creation.triangulate_polygon(a,engine='earcut');vv=np.c_[v,elev(v[:,0],v[:,1])+offset];put(trimesh.Trimesh(vv,f,process=False),key,color)
  except:pass
def cylinder(a,b,r,key,color,sections=8):
 v=np.asarray(b)-a;l=np.linalg.norm(v)
 if l<.001:return
 m=trimesh.creation.cylinder(radius=r,height=l,sections=sections);T=trimesh.geometry.align_vectors([0,0,1],v/l);T[:3,3]=(np.asarray(a)+b)/2;m.apply_transform(T);put(m,key,color)

# Ground surface uses LiDAR elevations, with an explicit circular model boundary.
vv=np.c_[gx.ravel(),gy.ravel(),grid.ravel()];ff=[]
for y in range(N-1):
 for x in range(N-1):
  i=y*N+x
  if math.hypot(gx[y,x],gy[y,x])<265:ff.extend([[i,i+1,i+N],[i+1,i+N+1,i+N]])
put(trimesh.Trimesh(vv,ff,process=False),'terrain','sidewalk')
waterways=[w['g'] for w in ways if w['tags'].get('natural')=='water' and w['g'].geom_type in ['Polygon','MultiPolygon']]
for p in waterways:extrude(p.intersection(boundary),-base,.04,'water','water')

# Municipal footprints are reconciled with newer OSM buildings.
osm_b=[w for w in ways if 'building' in w['tags'] and w['g'].area>40]
candidates=[]
for i,b in enumerate(json.load(open(DATA/'buildings.json'))):
 g=geom(b['shape'])
 if g.area<65 or not g.intersects(study.buffer(25)):continue
 overlaps=[(g.intersection(w['g']).area/max(g.area,1),w) for w in osm_b if g.intersects(w['g'])]
 match=max(overlaps,key=lambda a:a[0])[1] if overlaps else None
 if match and match['tags'].get('name','').startswith('Hyatt'):continue
 candidates.append({'id':'sf_'+b['sf16_bldgid'].replace('.','_'),'g':g,'old':b,'osm':match,'fp':'DataSF footprint'})
for w in osm_b:
 if not w['g'].intersects(study.buffer(25)):continue
 overlap=unary_union([c['g'].intersection(w['g']) for c in candidates if c['g'].intersects(w['g'])]).area
 if overlap/w['g'].area<.55:
  candidates.append({'id':'osm_'+w['id'],'g':w['g'],'old':{},'osm':w,'fp':'OSM footprint'})

all_buildings=unary_union([c['g'] for c in candidates]);ground_kd=cKDTree(ground[:,:2])
def get_points(g):
 a,b,c,d=g.bounds;m=(pts[:,0]>=a)&(pts[:,0]<=c)&(pts[:,1]>=b)&(pts[:,1]<=d)&(pts[:,3]!=7)&(pts[:,3]!=18)
 a=pts[m];return a[contains_xy(g,a[:,0],a[:,1])]

for c in candidates:
 id,g=c['id'],c['g'];og=c['old'];tags=c['osm']['tags'] if c['osm'] else {};name=tags.get('name') or (tags.get('addr:housenumber','').replace(';',' / ')+' '+tags.get('addr:street','')).strip() or 'Building '+og.get('sf16_bldgid',id)
 hero=bool(c['osm'] and c['osm']['id']=='149167949');stadium=g.area>20000
 if hero:name='188 King Street'
 if stadium:name='Oracle Park'
 sample=get_points(g.buffer(-.65));z0=float(elev(g.centroid.x,g.centroid.y));roof=sample[sample[:,2]>z0+3]
 if len(roof)>50:
  lo,top=np.quantile(roof[:,2],[.55,.97]);height=max(3.,float(top-z0));source='USGS LiDAR 2023'
 else:
  try:height=float(tags.get('height',og.get('hgt_median_m','12')).split()[0])
  except:height=12.
  lo=z0+height;top=lo;source='Catalog estimate'
 color='brick' if hero or stadium or height<20 else ['stone','brick2','stone','brick'][sum(map(ord,id))%4]
 roof_step=2.0 if stadium else 1.5
 xmin,ymin,xmax,ymax=g.bounds;nx=max(2,math.ceil((xmax-xmin)/roof_step));ny=max(2,math.ceil((ymax-ymin)/roof_step))
 roofparts=0
 if len(roof)>50:
  rr=np.full((ny,nx),np.nan);ix=np.clip(((roof[:,0]-xmin)/roof_step).astype(int),0,nx-1);iy=np.clip(((ymax-roof[:,1])/roof_step).astype(int),0,ny-1)
  groups=collections.defaultdict(list)
  for k,z in zip(iy*nx+ix,roof[:,2]):groups[int(k)].append(z)
  for k,vs in groups.items():rr[k//nx,k%nx]=np.quantile(vs,.8)
  missing=~np.isfinite(rr)
  if (~missing).any():rr[missing]=rr[tuple(distance_transform_edt(missing,return_distances=False,return_indices=True)[:,missing])]
  rr=median_filter(rr,3);base_roof=float(np.quantile(rr[np.isfinite(rr)],[.15])[0])
  if stadium:base_roof=max(z0+6,min(base_roof,z0+12))
  else:base_roof=max(z0+4,min(base_roof,lo))
 else:base_roof=z0+height;rr=None
 extrude(g,z0,base_roof-z0,'buildings__'+id,color)
 extrude(g,base_roof,.15,'roofs__'+id,'roof')
 if rr is not None:
  # Planar roof tiers come from the measured height raster, regularized to 1 m levels.
  quant=(np.round(rr-base_roof)).astype(np.int16);quant[quant<2]=0;quant[quant>max(height+3,5)]=0
  for geo,val in shapes(quant.astype('int16'),mask=quant>0,transform=from_origin(xmin,ymax,roof_step,roof_step)):
   part=shape(geo).intersection(g)
   if part.area<12:continue
   extrude(part,base_roof,float(val),'buildings__'+id,color);extrude(part,base_roof+float(val),.13,'roofs__'+id,'roof');roofparts+=1
 # Facade dimensions are procedural approximations, kept separate from measured geometry.
 main_height=min(height,base_roof-z0+1);floor=5.4 if hero else (4.2 if stadium else 3.25)
 for p in polys(g):
  rings=[p.exterior]+list(p.interiors)
  for ring in rings:
   coords=list(ring.coords)
   for a,b in zip(coords,coords[1:]):
    a,b=np.asarray(a),np.asarray(b);v=b-a;length=np.linalg.norm(v)
    if length<4:continue
    u=v/length;n=np.array([u[1],-u[0]]);mid=(a+b)/2
    if g.contains(Point(*(mid+n*.4))):n=-n
    # Hide shared-wall windows where another building is immediately adjacent.
    if all_buildings.contains(Point(*(mid+n*1.0))):continue
    angle=math.atan2(u[1],u[0]);front=hero and n[0]>.2 and n[1]<-.2
    spacing=5.0 if front else (7 if stadium else 3.1);cols=max(1,int(length/spacing));step=length/cols
    rows=max(1,int((main_height-4)/floor))
    for j in range(cols):
     pos=a+u*(j+.5)*step+n*.12
     if length<7 and cols==1:ww=min(step*.65,2.4)
     else:ww=step*(.76 if front else .57)
     for row in range(rows):
      zz=z0+4.6+row*floor;hh=min(floor*.7,main_height-(zz-z0)-.5)
      if hh<.6:continue
      if front:
       cube([*pos,zz+hh/2],[ww+.24,.3,hh+.24],'facades__'+id,'frame',angle)
       cube([*(pos+n*.18),zz+hh/2],[ww,.12,hh],'facades__'+id,'glass',angle)
       for dx in [-ww/6,ww/6]:cube([*(pos+u*dx+n*.25),zz+hh/2],[.045,.07,hh],'facades__'+id,'frame',angle)
      else:
       cube([*pos,zz+hh/2],[ww,.13,hh],'facades__'+id,'glass',angle)
       cube([*(pos+n*.07),zz+hh/2],[.07,.13,hh],'facades__'+id,'frame',angle)
       cube([*pos,zz-.07],[ww+.14,.23,.13],'facades__'+id,'stone',angle)
     cube([*pos,z0+1.65],[min(step*.8,3.5),.14,2.8],'facades__'+id,'dark',angle)
    cylinder([*a,base_roof+.3],[*b,base_roof+.3],.1,'facades__'+id,'frame')
    if front:
     # Distinctive aluminum-framed penthouse and terrace, approximate dimensions.
     ppos=mid-n*1.7;zh=z0+height-3
     cube([*ppos,zh],[length*.92,2.8,4.5],'facades__'+id,'glass',angle)
     cube([*ppos,zh+2.3],[length*.98,3.2,.25],'facades__'+id,'frame',angle)
     cylinder([*(a+n*.15),z0+height-6],[*(b+n*.15),z0+height-6],.065,'facades__'+id,'metal')
 features.append({'id':id,'name':name,'height':round(height,2),'heightSource':source,'footprintSource':c['fp'],'roofParts':roofparts,'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'pointCount':len(roof),'facade':'Brick and projecting aluminum window frames; dimensions remain approximate.' if hero else 'Measured building envelope with an inferred facade treatment.','sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id']})
 print(name,round(height,1),len(roof),roofparts,flush=True)

# Paved carriageways use OSM centerlines and lane counts; widths are inferred.
roads=[];rails=[];footpaths=[]
for w in ways:
 t,g=w['tags'],w['g'];h=t.get('highway')
 if h in ['primary','secondary','tertiary','unclassified','residential','living_street','service'] and g.geom_type=='LineString':
  try:lanes=int(t.get('lanes','2'))
  except:lanes=2
  width=lanes*3.15+(2.2 if 'parking:lane:right' in t or 'parking:lane:both' in t else 0)+(1.5 if t.get('cycleway:right')=='lane' else 0)
  if h=='service':width=4.5
  width=min(width,23)
  road=g.buffer(width/2,cap_style=2,join_style=2).intersection(boundary).difference(all_buildings)
  surface(road,'roads','asphalt',.09);roads.append((w,width,lanes))
 elif h in ['footway','pedestrian','path','cycleway']:
  footpaths.append(w)
 if t.get('railway') in ['light_rail','tram','rail'] and g.geom_type=='LineString':rails.append(w)

road_area=unary_union([r[0]['g'].buffer(r[1]/2,cap_style=2) for r in roads]);counts['roadSegments']=len(roads)
for w in footpaths:
 g=w['g'];t=w['tags']
 if t.get('footway')=='crossing':
  for line in lines(g.intersection(boundary)):
   if line.length>45:continue
   for s in np.arange(.3,line.length,.95):
    pt=line.interpolate(s);p2=line.interpolate(min(s+.1,line.length));angle=math.atan2(p2.y-pt.y,p2.x-pt.x)
    cube([pt.x,pt.y,float(elev(pt.x,pt.y))+.14],[.45,2.6,.025],'markings','paint',angle)
  counts['crosswalks']+=1
 elif g.geom_type in ['Polygon','MultiPolygon']:surface(g.intersection(boundary).difference(all_buildings),'paths','sidewalk',.12)

for w,width,lanes in roads:
 g=w['g'];t=w['tags']
 for k in range(1,lanes):
  offset=(k-lanes/2)*3.15
  for l in lines(g.offset_curve(offset).intersection(boundary)):
   for s in np.arange(7,l.length-5,9):
    a=l.interpolate(s);b=l.interpolate(min(s+3,l.length-3));v=np.array([b.x-a.x,b.y-a.y]);le=np.linalg.norm(v)
    if le>.5:cube([(a.x+b.x)/2,(a.y+b.y)/2,float(elev(a.x,a.y))+.14],[le,.12,.02],'markings','paint',math.atan2(v[1],v[0]))
 # A continuous divider is only synthesized for mapped two-way roads.
 if t.get('oneway')!='yes' and lanes>=2:
  for l in lines(g.intersection(boundary)):
   for a,b in zip(l.coords,list(l.coords)[1:]):
    a,b=np.array(a),np.array(b);v=b-a
    cube([*(a+b)/2,float(elev(*(a+b)/2))+.13],[np.linalg.norm(v),.15,.02],'markings','yellow',math.atan2(v[1],v[0]))

for w in rails:
 g=w['g'];surface(g.buffer(1.5,cap_style=2).intersection(boundary),'transit','stone',.11)
 for off in [-.7175,.7175]:
  for line in lines(g.offset_curve(off).intersection(boundary)):
   coords=list(line.coords)
   for a,b in zip(coords,coords[1:]):cylinder([*a,float(elev(*a))+.2],[*b,float(elev(*b))+.2],.045,'rails','rail',6)
 # Overhead contact wire uses mapped track position; height is an inferred standard.
 for line in lines(g.intersection(boundary)):
  for a,b in zip(line.coords,list(line.coords)[1:]):cylinder([*a,float(elev(*a))+5.7],[*b,float(elev(*b))+5.7],.025,'street_objects','metal',5)
 counts['railSegments']+=1

# Tree inventory placement; species guides reusable shape, not an exact tree scan.
tree_positions=[]
for t in json.load(open(DATA/'trees.json')):
 try:x,y=local(float(t['longitude']),float(t['latitude']))
 except:continue
 if math.hypot(x,y)>230 or all_buildings.contains(Point(x,y)):continue
 if any((x-a)**2+(y-b)**2<1 for a,b in tree_positions):continue
 tree_positions.append((x,y));z=float(elev(x,y));palm='palm' in t.get('qspecies','').lower() or 'phoenix' in t.get('qspecies','').lower()
 seed=int(float(t.get('treeid',0)));rng=np.random.default_rng(seed)
 h=9+rng.random()*3 if palm else 5+rng.random()*3
 cylinder([x,y,z],[x,y,z+h],.22 if palm else .14,'vegetation','trunk',8)
 if palm:
  for j in range(10):
   ang=j*math.tau/10;end=[x+3*math.cos(ang),y+3*math.sin(ang),z+h-.9]
   mid=[x+1.6*math.cos(ang),y+1.6*math.sin(ang),z+h+.45]
   cylinder([x,y,z+h],mid,.15,'vegetation','leaf',5);cylinder(mid,end,.11,'vegetation','leaf',5)
   m=trimesh.creation.icosphere(subdivisions=1,radius=1);m.apply_scale([1.25,.4,.15]);T=trimesh.transformations.rotation_matrix(ang,[0,0,1]);m.apply_transform(T);m.apply_translation(mid);put(m,'vegetation','leaf')
 else:
  for j in range(3):
   m=trimesh.creation.icosphere(subdivisions=1,radius=1);m.apply_scale([1.6+rng.random(),1.6+rng.random(),2.2]);m.apply_translation([x+rng.uniform(-.6,.6),y+rng.uniform(-.6,.6),z+h-1+j*.7]);put(m,'vegetation',['#405f46','#56724e','#4c6b4b'][j])
counts['trees']=len(tree_positions)

# Render independently mapped signal nodes; retain coarse municipal sign records separately.
for node in osm.findall('node'):
 tags={t.attrib['k']:t.attrib['v'] for t in node.findall('tag')}
 if tags.get('highway') not in ['traffic_signals','street_lamp']:continue
 x,y=local(float(node.attrib['lon']),float(node.attrib['lat']))
 if math.hypot(x,y)>230:continue
 z=float(elev(x,y));h=5.5 if tags['highway']=='traffic_signals' else 8
 cylinder([x,y,z],[x,y,z+h],.09,'street_objects','metal')
 if tags['highway']=='traffic_signals':
  cube([x,y,z+h-.5],[.3,.3,1.1],'street_objects','dark')
  for j,col in enumerate(['#b95246','#c2a457','#6a915f']):cube([x,y-.17,z+h-.15-j*.3],[.17,.05,.17],'street_objects',col)
 else:
  cylinder([x,y,z+h],[x+1.6,y,z+h],.06,'street_objects','metal');cube([x+1.6,y,z+h],[.9,.4,.14],'street_objects','frame')
 counts['signals' if tags['highway']=='traffic_signals' else 'lamps']+=1

# King Street median barriers, repeated along mapped tracks; inferred dimensions.
for w in rails:
 if w['g'].distance(Point(35,-30))>100:continue
 for l in lines(w['g'].offset_curve(1.7).intersection(study)):
  if l.length<15:continue
  for s in np.arange(5,l.length-5,3):
   a=l.interpolate(s);b=l.interpolate(min(s+3,l.length-5));z=float(elev(a.x,a.y))
   cylinder([a.x,a.y,z+.1],[a.x,a.y,z+1.05],.035,'street_objects','metal',6)
   for hh in [.55,1.0]:cylinder([a.x,a.y,z+hh],[b.x,b.y,float(elev(b.x,b.y))+hh],.025,'street_objects','metal',5)

for w in ways:
 t=w['tags'];g=w['g']
 if t.get('leisure')=='pitch' and g.geom_type in ['Polygon','MultiPolygon']:surface(g.intersection(boundary),'landscape','grass',.2)

# Finish batches: one mesh per semantic group/building keeps draw-call cost bounded.
for key,(v,f,c) in batches.items():
 if not v:continue
 mesh=trimesh.Trimesh(np.concatenate(v),np.concatenate(f),vertex_colors=np.concatenate(c),process=False)
 layer=key.split('__')[0];fid=key.split('__')[1] if '__' in key else None
 mesh.metadata={'layer':layer,'featureId':fid,'evidence':'measured' if layer in ['terrain','buildings','roofs'] else 'inferred'}
 scene.add_geometry(mesh,node_name=key,geom_name=key,metadata=mesh.metadata)
scene.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float))
counts.update(buildings=len(features),lidarPoints=len(pts),roofParts=sum(f['roofParts'] for f in features))
coarse_signs=[]
for s in json.load(open(DATA/'signs.json')):
 try:x,y=geom(s['shape']).coords[0]
 except:continue
 if math.hypot(x,y)<200:coarse_signs.append({'id':s.get('asset_id'),'legend':s.get('legend'),'x':round(x,2),'y':round(y,2),'precision':'municipal approximate location; not rendered as surveyed placement'})
counts['unresolvedSigns']=len(coarse_signs)
world={'center':{'lat':lat,'lon':lon},'coordinateSystem':'Local UTM zone 10N meters; GLB X east, Y up, Z south','verticalOrigin':base,'buildings':features,'counts':dict(counts),'ground':np.round(grid,3).ravel().tolist(),'grid':{'min':MIN,'step':STEP,'n':N},'sources':[
 {'name':'USGS LiDAR','date':'April 2023','detail':'4.6 million sampled points; ground elevations and fitted roof tiers. Absolute accuracy has not been independently tested.','url':'https://www.fisheries.noaa.gov/inport/item/73386/full-list'},
 {'name':'DataSF building footprints','date':'2010-derived geometry','detail':'Building outlines reconciled against current OpenStreetMap identities and newer LiDAR heights.','url':'https://data.sfgov.org/d/ynuv-fyni'},
 {'name':'OpenStreetMap','date':'Retrieved Sep 2026','detail':'Address identity, newer buildings, road topology, lane counts, crossings and transit tracks. Capture age varies.','url':'https://www.openstreetmap.org/copyright'},
 {'name':'DataSF street trees','date':'Inventory; dates vary','detail':'Mapped tree positions and species. Canopies and branches are procedural.','url':'https://data.sfgov.org/d/tkzw-k3nq'},
 {'name':'Street-level reference','date':'January 2025','detail':'Targeted Google Street View inspection of the King Street frontage. No Google image textures are embedded.','url':'https://www.google.com/maps/@37.7786899,-122.3913452,3a,75y,315h,110t/data=!3m4!1e1!3m2!1sS4hs-Kyh3P14aRv9CWeTvg!2e0'}]}
glb=scene.export(file_type='glb');(PUBLIC/'king-street.glb').write_bytes(glb);(OUT/'king-street-pilot.glb').write_bytes(glb)
(PUBLIC/'world.json').write_text(json.dumps(world,separators=(',',':')));(OUT/'king-street-evidence.json').write_text(json.dumps({'world':world,'unresolvedSigns':coarse_signs},indent=2))
print('OUTPUT',len(glb),'bytes',dict(counts),'center',lat,lon,flush=True)
