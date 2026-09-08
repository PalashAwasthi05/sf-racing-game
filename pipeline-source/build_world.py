"""King Street pilot: geospatial inputs -> metric scene, evidence manifest and GLB."""
import json, math, pathlib, collections, xml.etree.ElementTree as ET
import time
START=time.perf_counter()
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
from pipeline.materials import Materials
from pipeline import facades,streets,vegetation,signs
from trimesh.visual.texture import TextureVisuals

ROOT=pathlib.Path(__file__).parent;DATA=ROOT/'data';PUBLIC=ROOT/'king-street'/'public';PUBLIC.mkdir(parents=True,exist_ok=True)
OUT=ROOT.parent/'outputs';OUT.mkdir(exist_ok=True)
CONFIG=json.load(open(ROOT/'pipeline/config.json'));OBS=json.load(open(ROOT/'pipeline/observations.json'));facade_records=[];sign_records=[]
ll=Transformer.from_crs(4326,32610,always_xy=True);inv=Transformer.from_crs(32610,4326,always_xy=True)
web=Transformer.from_crs(3857,32610,always_xy=True)
osm=ET.parse(DATA/'osm.xml').getroot()
nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in osm.findall('node')}
raw_ways=[]
for w in osm.findall('way'):
 tags={a.attrib['k']:a.attrib['v'] for a in w.findall('tag')}
 pts=[nodes[n.attrib['ref']] for n in w.findall('nd') if n.attrib['ref'] in nodes]
 if len(pts)>1:raw_ways.append((w.attrib['id'],tags,pts))
target=next(w for w in raw_ways if w[0]==CONFIG['region']['addressWayId'])
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
boundary=Point(0,0).buffer(CONFIG['region']['contextRadiusM'],quad_segs=96);study=Point(0,0).buffer(CONFIG['region']['studyRadiusM'])
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
material_bank=Materials(ROOT/'material-assets',palette)
batches=collections.defaultdict(lambda:[[],[],[],[]])
def put(mesh,key,color,uv=None):
 if len(mesh.vertices)==0:return
 material,scale=material_bank.get(color);batchkey=key+'___'+material.name
 vertices,faces,uvs,normals=batches[batchkey];off=sum(len(v) for v in vertices)
 # Split face corners to preserve hard normals and per-face, metric texture coordinates.
 inds=mesh.faces.ravel();v=np.asarray(mesh.vertices)[inds];norm=np.repeat(mesh.face_normals,3,axis=0)
 if uv is None:
  axis=np.argmax(np.abs(norm),axis=1);coord=np.zeros((len(v),2));horizontal=axis==2
  coord[horizontal]=v[horizontal,:2]/scale
  mask=~horizontal;tangent=np.c_[-norm[mask,1],norm[mask,0]];tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
  coord[mask,0]=np.sum(v[mask,:2]*tangent,axis=1)/scale[0];coord[mask,1]=v[mask,2]/scale[1]
 else:coord=np.asarray(uv)[inds]
 vertices.append(v);faces.append(np.arange(len(v)).reshape(-1,3)+off);uvs.append(coord);normals.append(norm)
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
 if match and any(d['osmWayId']==match['id'] and d['selectedSource']=='OSM' for d in OBS.get('footprintDecisions',[])):continue
 candidates.append({'id':'sf_'+b['sf16_bldgid'].replace('.','_'),'g':g,'old':b,'osm':match,'fp':'DataSF footprint'})
for w in osm_b:
 if not w['g'].intersects(study.buffer(25)):continue
 overlap=unary_union([c['g'].intersection(w['g']) for c in candidates if c['g'].intersects(w['g'])]).area
 if overlap/w['g'].area<.55:
  candidates.append({'id':'osm_'+w['id'],'g':w['g'],'old':{},'osm':w,'fp':'OSM footprint'})

for c in candidates:
 c['full_area']=c['g'].area;c['clipped']=not boundary.covers(c['g']);c['g']=c['g'].intersection(boundary)
all_buildings=unary_union([c['g'] for c in candidates]);ground_kd=cKDTree(ground[:,:2])
def get_points(g):
 a,b,c,d=g.bounds;m=(pts[:,0]>=a)&(pts[:,0]<=c)&(pts[:,1]>=b)&(pts[:,1]<=d)&(pts[:,3]!=7)&(pts[:,3]!=18)
 a=pts[m];return a[contains_xy(g,a[:,0],a[:,1])]

for c in candidates:
 id,g=c['id'],c['g'];og=c['old'];tags=c['osm']['tags'] if c['osm'] else {};name=tags.get('name') or (tags.get('addr:housenumber','').replace(';',' / ')+' '+tags.get('addr:street','')).strip() or 'Building '+og.get('sf16_bldgid',id)
 stadium=c['full_area']>18000
 sample=get_points(g.buffer(-.65));z0=float(elev(g.centroid.x,g.centroid.y));roof=sample[sample[:,2]>z0+3]
 if len(roof)>50:
  lo,top=np.quantile(roof[:,2],[.55,.97]);height=max(3.,float(top-z0));source='USGS LiDAR 2023'
 else:
  try:height=float(tags.get('height',og.get('hgt_median_m','12')).split()[0])
  except:height=12.
  lo=z0+height;top=lo;source='Catalog estimate'
 style=facades.prior(tags,height,c['full_area']);color=style['material']
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
  rr=median_filter(rr,5);base_roof=float(np.quantile(rr[np.isfinite(rr)],[.15])[0])
  if stadium:base_roof=max(z0+6,min(base_roof,z0+12))
  else:base_roof=max(z0+4,min(base_roof,lo))
 else:base_roof=z0+height;rr=None
 # Facade compiler builds walls with real openings; no opaque block behind glazing.
 extrude(g,base_roof,.15,'roofs__'+id,'roof')
 if rr is not None:
  # Planar roof tiers come from the measured height raster, regularized to 1 m levels.
  step=CONFIG['defaults']['roofRegularizationM'];quant=(np.round((rr-base_roof)/step)*step).astype(np.int16);quant[quant<3]=0;quant[quant>max(height+3,5)]=0
  for geo,val in shapes(quant.astype('int16'),mask=quant>0,transform=from_origin(xmin,ymax,roof_step,roof_step)):
   part=shape(geo).buffer(.8).buffer(-.8).simplify(.45).intersection(g)
   if part.area<max(25,g.area*.012):continue
   tier=dict(c);tier['g']=part;tier['elevationOffset']=base_roof-z0;tier['outerBoundary']=g.boundary;tier_ctx=dict(globals());tier_ctx['all_buildings']=all_buildings.difference(g)
   rr_records,rr_components=facades.compile(tier_ctx,tier,base_roof,float(val),height,tags,OBS['facades']);facade_records.extend(rr_records);counts['windowAssemblies']+=rr_components
   extrude(part,base_roof+float(val),.13,'roofs__'+id,'roof');roofparts+=1
 outer_ctx=dict(globals());outer_ctx['roofSamples']=roof
 records,component_count=facades.compile(outer_ctx,c,z0,base_roof-z0,height,tags,OBS['facades']);facade_records.extend(records);counts['windowAssemblies']+=component_count
 observed_count=sum(r['status']=='visual-fit' for r in records)
 features.append({'id':id,'name':name,'height':round(height,2),'heightSource':source,'footprintSource':c['fp'],'roofParts':roofparts,'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'clippedAtBoundary':c['clipped'],'pointCount':len(roof),'facade':'Image-informed facade components; remaining walls use unverified priors.' if observed_count else 'Detailed facade generated from an unverified architectural prior.','facadeEvidence':'Partial visual fit' if observed_count else 'Unverified prior','fittedWalls':observed_count,'totalWalls':len(records),'sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id']})
 print(name,round(height,1),len(roof),roofparts,flush=True)

# Paved carriageways use OSM centerlines and lane counts; widths are inferred.
roads=[];rails=[];footpaths=[]
for w in ways:
 t,g=w['tags'],w['g'];h=t.get('highway')
 if h in ['primary','secondary','tertiary','unclassified','residential','living_street','service'] and g.geom_type=='LineString':
  try:lanes=int(t.get('lanes','2'))
  except:lanes=2
  width=lanes*CONFIG['defaults']['laneWidthM']+(2.2 if 'parking:lane:right' in t or 'parking:lane:both' in t else 0)+(1.5 if t.get('cycleway:right')=='lane' else 0)
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

# Shared species templates replace low-detail canopy primitives.
vegetation.compile(globals())

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

# Median barriers along mapped tram/light-rail tracks; inferred dimensions.
for w in rails:
 if w['tags'].get('railway') not in ['tram','light_rail']:continue
 for l in lines(w['g'].offset_curve(1.7).intersection(study)):
  if l.length<15:continue
  for s in np.arange(5,l.length-5,3):
   a=l.interpolate(s);b=l.interpolate(min(s+3,l.length-5));z=float(elev(a.x,a.y))
   cylinder([a.x,a.y,z+.1],[a.x,a.y,z+1.05],.035,'street_objects','metal',6)
   for hh in [.55,1.0]:cylinder([a.x,a.y,z+hh],[b.x,b.y,float(elev(b.x,b.y))+hh],.025,'street_objects','metal',5)

for w in ways:
 t=w['tags'];g=w['g']
 if t.get('leisure')=='pitch' and g.geom_type in ['Polygon','MultiPolygon']:surface(g.intersection(boundary),'landscape','grass',.2)

# Street compiler uses source topology and ROW; every snap remains an uncertain record.
make_sign=signs.install(globals());sign_records,sidewalk_geometry=streets.compile(globals())
# One mesh per material and feature; image payloads are shared by glTF.
for key,(v,f,uv,normals) in batches.items():
 if not v:continue
 semantic,material_name=key.split('___',1);mat,_=material_bank.get(material_name)
 mesh=trimesh.Trimesh(np.concatenate(v),np.concatenate(f),vertex_normals=np.concatenate(normals),visual=TextureVisuals(uv=np.concatenate(uv),material=mat),process=False)
 mesh.merge_vertices(merge_tex=False,merge_norm=False)
 layer=semantic.split('__')[0];fid=semantic.split('__')[1] if '__' in semantic else None
 status='measured-fit' if layer in ['terrain','roofs'] else 'visual-fit' if fid and any(r['buildingId']==fid and r['status']=='visual-fit' for r in facade_records) else 'inferred'
 mesh.metadata={'layer':layer,'featureId':fid,'evidence':status,'material':material_name}
 scene.add_geometry(mesh,node_name=key,geom_name=key,metadata=mesh.metadata)
scene.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float))
counts.update(buildings=len(features),lidarPoints=len(pts),roofParts=sum(f['roofParts'] for f in features))
coarse_signs=[]
for s in json.load(open(DATA/'signs.json')):
 try:x,y=geom(s['shape']).coords[0]
 except:continue
 if math.hypot(x,y)<200:coarse_signs.append({'id':s.get('asset_id'),'legend':s.get('legend'),'x':round(x,2),'y':round(y,2),'precision':'municipal approximate location; not rendered as surveyed placement'})
counts['unresolvedSigns']=len(coarse_signs)
counts['facadeWalls']=len(facade_records);counts['imageFittedWalls']=sum(r['status']=='visual-fit' for r in facade_records);counts['pbrMaterials']=len(material_bank.cache)
counts['generationSeconds']=round(time.perf_counter()-START,2)
world={'pipelineVersion':CONFIG['version'],'facadeCoverage':{'walls':len(facade_records),'visuallyFittedWalls':counts['imageFittedWalls'],'independentlyVerifiedWalls':0,'accuracyGate':'NOT PASSED'},'studyRadiusM':200,'contextRadiusM':265,'center':{'lat':lat,'lon':lon},'coordinateSystem':'Local UTM zone 10N meters; GLB X east, Y up, Z south','verticalOrigin':base,'buildings':features,'counts':dict(counts),'ground':np.round(grid,3).ravel().tolist(),'grid':{'min':MIN,'step':STEP,'n':N},'sources':[
 {'name':'USGS LiDAR','date':'April 2023','detail':'4.6 million sampled points; ground elevations and fitted roof tiers. Absolute accuracy has not been independently tested.','url':'https://www.fisheries.noaa.gov/inport/item/73386/full-list'},
 {'name':'DataSF building footprints','date':'2010-derived geometry','detail':'Building outlines reconciled against current OpenStreetMap identities and newer LiDAR heights.','url':'https://data.sfgov.org/d/ynuv-fyni'},
 {'name':'OpenStreetMap','date':'Retrieved Sep 2026','detail':'Address identity, newer buildings, road topology, lane counts, crossings and transit tracks. Capture age varies.','url':'https://www.openstreetmap.org/copyright'},
 {'name':'DataSF street trees','date':'Inventory; dates vary','detail':'Mapped tree positions and species. Canopies and branches are procedural.','url':'https://data.sfgov.org/d/tkzw-k3nq'},
 {'name':'Reusable PBR surfaces','date':'CC0 scans','detail':'Scanned brick, asphalt and concrete at physical texture scale; material identity remains inferred on unobserved walls.','url':'https://ambientcg.com/'},
 {'name':'Mapillary coverage','date':'Coverage inspected Sep 2026','detail':'Dense capture coverage is visible around King and Third. Individual images and their capture dates have not been ingested in this pass.','url':'https://www.mapillary.com/app/?lat=37.7786&lng=-122.3918&z=18'},
 {'name':'Street-level reference','date':'January 2025','detail':'Dated frontage observations fit reusable window/frame/canopy parameters. No Google image textures are embedded; most facades remain unobserved.','url':'https://www.google.com/maps/@37.7786899,-122.3913452,3a,75y,315h,110t/data=!3m4!1e1!3m2!1sS4hs-Kyh3P14aRv9CWeTvg!2e0'}]}
glb=scene.export(file_type='glb');(PUBLIC/'king-street.glb').write_bytes(glb);(OUT/'king-street-pilot.glb').write_bytes(glb)
(PUBLIC/'world.json').write_text(json.dumps(world,separators=(',',':')));(OUT/'king-street-evidence.json').write_text(json.dumps({'world':world,'unresolvedSigns':coarse_signs,'facadeWalls':facade_records,'signPlacements':sign_records,'observationSources':OBS['sources'],'materials':material_bank.manifest},indent=2))
print('OUTPUT',len(glb),'bytes',dict(counts),'center',lat,lon,flush=True)

(PUBLIC/'facade-evidence.json').write_text(json.dumps({'walls':facade_records,'signs':sign_records,'sources':OBS['sources']},separators=(',',':')))
print('GENERATION_SECONDS',round(time.perf_counter()-START,2),flush=True)

import gzip
with gzip.GzipFile(PUBLIC/'king-street.glb.gz','wb',compresslevel=7,mtime=0) as stream:stream.write(glb)
(PUBLIC/'king-street.glb').unlink()
# Bounded transfer chunks; raw GLB is retained only as a local interchange artifact.
packed=(PUBLIC/'king-street.glb.gz').read_bytes();chunks=[]
for i in range(0,len(packed),4*1024*1024):
 name=f'king-street-{i//(4*1024*1024):02d}.bin';(PUBLIC/name).write_bytes(packed[i:i+4*1024*1024]);chunks.append('/'+name)
(PUBLIC/'model.json').write_text(json.dumps({'compression':'gzip','chunks':chunks,'compressedBytes':len(packed),'uncompressedBytes':len(glb)}))
(PUBLIC/'king-street.glb.gz').unlink()
