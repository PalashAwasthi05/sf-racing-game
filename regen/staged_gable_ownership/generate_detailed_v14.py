"""Area generator: geospatial inputs -> metric scene, evidence manifest and GLB. Area-specific text and file names
come from pipeline/config.json (`areaSources`, `outputPrefix`); the rules carry no location.
v14 driver: same generator as history/v7/build_world.py; output directory is a CLI argument."""
import json, math, pathlib, collections, xml.etree.ElementTree as ET
import time
START=time.perf_counter()
import numpy as np
import trimesh
from shapely.geometry import shape, Polygon, Point, LineString, box
from shapely.ops import transform, unary_union
from shapely import contains_xy, make_valid, segmentize, polygons as polygon_array, covered_by
from pyproj import Transformer
from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import median_filter, gaussian_filter, distance_transform_edt
from rasterio.features import shapes
from rasterio.transform import from_origin
from pipeline.materials import Materials
from pipeline import facades,streets,vegetation,signs,envelopes,lettering,surface_evidence,spatial_rules,mobility,freshness,street_space,street_ground,ground_paint,roof_envelope,landmarks,roof_forms,roof_features,facade_lint,roof_observations,roof_shell,roof_assemblies,roof_source_parts,facade_wall_domain,roof_front_ownership,roof_gable_ownership
from general_world.roof_planes import roof_solid,evaluate as roof_evaluate
from trimesh.visual.texture import TextureVisuals

import argparse
_args=argparse.ArgumentParser(description='Detailed pilot generator; writes a candidate (raw GLB, chunks, world.json, evidence) into --output instead of the live preview.');_args.add_argument('--output',required=True);_args.add_argument('--regen-profiles',default=None);_args.add_argument('--area',default=None);_args.add_argument('--streets-only',action='store_true',help='Emit street layers only for validated layer replacement; never publish this partial scene directly.');_args.add_argument('--massing',default=None,help='JSON {building id: {"drop": true} | {"outline": [[x, y], ...]}}: outlines the street photographs and LiDAR contradict (areas/ring/massing)');_args.add_argument('--only',default=None,help='JSON list of building ids: build only those buildings, to splice into a built area (areas/ring/splice.py); the ground model still runs');_args.add_argument('--roof-evidence',action='store_true',help='Save the exact filtered roof returns used by the generator for reconstruction review.');_args.add_argument('--experimental-roof-reconstruction',action='store_true',help='Candidate-only roof observation validation and adaptive plane sampling; visual acceptance is still pending.');_args.add_argument('--roof-shell',action='store_true',help='Build candidate roofs as continuous LiDAR-supported shells with observed roof details.');_args.add_argument('--roof-observations',default=None,help='JSON roof observation list (or an object containing observations)');_args=_args.parse_args()
# --only FILE: the listed buildings alone (their parts by source id). Every other outline stays a neighbour for the walls'
# facing rules; trees, street furniture and street surfaces are skipped (the splice keeps the built area's own).
ONLY=set(json.load(open(_args.only))) if _args.only else None;ONLY_SRC={i.split('_p')[0] for i in ONLY} if ONLY else None
ROOT=pathlib.Path(__file__).parent;DATA=ROOT/'data'
# --area DIR: another model area (areas/<name>: its own config.json and data/). Every hand-made input of the pilot is
# keyed to the pilot's buildings, so an area is built as a regeneration: its facade readings (--regen-profiles), or none.
import sys as _sys
AREA=pathlib.Path(_sys.argv[_sys.argv.index('--area')+1]) if '--area' in _sys.argv else None
if AREA is not None:DATA=AREA/'data'
OUTDIR=pathlib.Path(_args.output);PUBLIC=OUTDIR/'public';PUBLIC.mkdir(parents=True,exist_ok=True)
# --regen-profiles FILE: the facade readings of a regeneration are the only facade inputs; hand-annotated profiles,
# observation packets, supplements, boundary fits, lettering, envelope descriptions and envelope colours are all left out
import sys as _sys
REGEN=None
if '--regen-profiles' in _sys.argv:
 REGEN=json.load(open(_sys.argv[_sys.argv.index('--regen-profiles')+1]))
elif AREA is not None:
 REGEN={'profiles':[]}
OUT=OUTDIR;OUT.mkdir(parents=True,exist_ok=True)
PROFILES=json.load(open(ROOT/'pipeline/facade-profiles.json'))['profiles'];FITS=json.load(open(ROOT/'pipeline/facade-layouts.json'))['fits']
if REGEN is not None:PROFILES=list(REGEN['profiles']);FITS=[]
# Every reading passes the facade coherence rules before anything is drawn from it (pipeline/facade_lint.py): a band
# through a window, a gable outline read as a parapet, a sliver of trim are repaired or dropped, and said so.
LINT={}
for _i,_p in enumerate(PROFILES):
 _q,_iss=facade_lint.clean(_p)
 if _iss:LINT[_p.get('id',str(_i))]=_iss
 PROFILES[_i]=_q
# buildings whose photographs show a pitched roof (a read eave, gable, mansard or tile roof, or a gable outline): the roof
# fit keeps a pitched form there on weaker LiDAR evidence (pipeline/roof_forms.fit pitched_evidence)
PITCHED_EV={}
for _p in PROFILES:
 _k={_r.get('kind') for _r in _p.get('roofs') or [] if _r.get('kind') in ('gable','mansard','tile-roof','eave','tile-eave','pent')}|({'gable'} if _p.get('gableOutline') else set())
 if _k:PITCHED_EV.setdefault(str((_p.get('match') or {}).get('featureId')),set()).update(_k)
# The eaves the photographs show under a pitched roof (a mansard, a front gable, a tiled hip), per building part. Where a
# part's readings give them and the LiDAR's wall top stands well above, the walls stop at the eaves and the roof rises
# from there to the LiDAR's top: the massing took the roof's lower slopes for wall, so mansards, hips and front gables
# stood as tall flat walls with a flat top (Palash's review, 10-15).
READ_EAVES={}
for _p in PROFILES:
 _rd=_p.get('reading') or {};_H=float(_p.get('heightRefM') or 0.)
 _pr=[_r for _r in _p.get('roofs') or [] if _r.get('kind') in ('gable','mansard','tile-roof','hip')]
 if not _pr or not _H or _rd.get('confidence') not in ('medium','high'):continue
 _e=[float(_r['z'])*_H for _r in _pr if isinstance(_r.get('z'),(int,float))]+([float(_rd['wallTopM'])] if isinstance(_rd.get('wallTopM'),(int,float)) else [])
 _m=_p.get('match') or {}
 if _e:READ_EAVES.setdefault(str(_m.get('partId') or _m.get('featureId')),[]).append((min(_e),{_r['kind'] for _r in _pr}))
# Observation packets produced by the frontage-reading step (one file per run) join the hand-annotated profiles.
EXTRA_OBSERVATIONS=[json.load(open(f)) for f in sorted((ROOT/'pipeline').glob('frontage-observations*.json'))] if REGEN is None else []
for _extra in EXTRA_OBSERVATIONS:PROFILES+=_extra.get('profiles',[])
ROOF_OBSERVATIONS=[]
if _args.roof_observations:
 _roof_obs_data=json.load(open(_args.roof_observations))
 ROOF_OBSERVATIONS=_roof_obs_data if isinstance(_roof_obs_data,list) else _roof_obs_data.get('observations',_roof_obs_data.get('roofObservations',[]))
# Colour zones, trims and sign boards measured on the photograph of each hand-read frontage (auto_read_frontages_v31.py
# --supplements): the hand reading keeps its layout and material; a kind it has none of is added from the photograph.
_SUP=ROOT/'pipeline/frontage-supplements-v31.json'
if _SUP.exists() and REGEN is None:
 _by_id={p['id']:p for p in PROFILES}
 for _s in json.load(open(_SUP)).get('supplements',[]):
  _t=_by_id.get(_s['supplementOf'])
  if _t is None:continue
  if _s.get('piers') and not _t.get('piers'):
   # a pier read near-black in the wall's own hue is the wall's material in shade (colour_agreement.shaded_wall)
   from pipeline.colour_agreement import shaded_wall as _shaded
   _wc=(_t.get('materialEvidence') or {}).get('colorRgb');_ps=[]
   for _q in _s['piers']:
    _hx=str(_q.get('material','')).partition('#')[2]
    if _wc and len(_hx)==6 and _shaded([int(_hx[i:i+2],16) for i in (0,2,4)],_wc):
     _q={k:v for k,v in _q.items() if k!='material'};_q['materialEvidence']=_t['materialEvidence'];_q['evidence']=_q.get('evidence','')+'; shaded side of the wall material'
    _ps.append(_q)
   _t['piers']=_ps
  if _s.get('bands') and not _t.get('bands'):_t['bands']=_s['bands']
  _zones=[q for q in _s.get('panels',[]) if 'zone' in q.get('evidence','')]
  if _zones and not any(q['rect'][0]<=.02 and q['rect'][2]>=.98 and q['rect'][3]-q['rect'][1]>.15 for q in _t.get('panels',[])):_t['panels']=_zones+_t.get('panels',[])
  if _s.get('windowFrame'):
   # a measured surround replaces a thin generic frame; a frame the hand reading names in its own colour stays
   _wf=_s['windowFrame']
   for _g in _t.get('groups',[])+_t.get('openings',[]):
    _st=_g.setdefault('style',{})
    if _st.get('kind','window')=='window' and float(_st.get('frameWidthM',.06))<.12:_st.update(frameMaterial=_wf['frameMaterial'],frameWidthM=_wf['frameWidthM'])
  _signs=[q for q in _s.get('panels',[]) if 'sign' in q.get('evidence','')]
  if _signs and not any('sign' in str(q.get('evidence','')) for q in _t.get('panels',[])):_t['panels']=_t.get('panels',[])+_signs
BOUNDARY_FITS=json.load(open(ROOT/'pipeline/boundary-layouts.json'))['fits'] if REGEN is None else []
for fit in BOUNDARY_FITS:
 profile=next(p for p in PROFILES if p['id']==fit['profileId']);profile.setdefault('openings',[]).extend(fit['openings'])
LETTERING_DATA=json.load(open(ROOT/'pipeline/lettering-observations.json'));lettering_records=[]
if REGEN is not None:LETTERING_DATA={**LETTERING_DATA,'signs':[]}
ENVELOPE_DATA=json.load(open(ROOT/'pipeline/envelope-observations.json'));ENVELOPES=ENVELOPE_DATA['profiles'];envelope_records=[];envelope_frontages=[]
if REGEN is not None:ENVELOPES=[]
# Photographed colours of envelope-designed frontages (read_envelope_colours_v32.py), by envelope profile id.
ENVELOPE_COLOURS=json.load(open(ROOT/'pipeline/envelope-colours-v32.json'))['envelopes'] if (ROOT/'pipeline/envelope-colours-v32.json').exists() else {}
if REGEN is not None:ENVELOPE_COLOURS={}
CONFIG=json.load(open((AREA or ROOT/'pipeline')/'config.json'));OBS=json.load(open(ROOT/'pipeline/observations.json'));facade_records=[];sign_records=[]
if REGEN is not None:
 OBS={**OBS,'facades':[]};facades._PARAPETS=[];facades._INCUMBENTS={}
ll=Transformer.from_crs(4326,32610,always_xy=True);inv=Transformer.from_crs(32610,4326,always_xy=True)
web=Transformer.from_crs(3857,32610,always_xy=True)
osm=ET.parse(DATA/'osm.xml').getroot()
nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in osm.findall('node')}
raw_ways=[];way_metadata={w.attrib['id']:{'editedAt':w.attrib.get('timestamp'),'version':w.attrib.get('version'),'captureAgeKnown':False} for w in osm.findall('way')}
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
 if g.intersects(boundary):ways.append({'id':id,'tags':t,'g':g,'metadata':way_metadata[id]})

pts=np.load(DATA/'points.npz')['points'];px,py=web.transform(pts[:,0],pts[:,1]);pts[:,0]=px-cx;pts[:,1]=py-cy
ground=pts[pts[:,3]==2];_zlo,_zhi=np.quantile(ground[:,2],[.005,.995]) if AREA is not None else (5.,10.);ground=ground[(ground[:,2]>_zlo-10)&(ground[:,2]<_zhi+10)]   # the area's own ground band (a hill stands 100 m up; a fixed -5..20 m band held only the waterfront)
base=float(np.median(ground[np.linalg.norm(ground[:,:2],axis=1)<40,2]));pts[:,2]-=base;ground[:,2]-=base
tree=cKDTree(ground[:,:2]);STEP=3.;MIN=-math.ceil((CONFIG['region']['contextRadiusM']+2)/STEP)*STEP;N=int(round(-2*MIN/STEP))+1   # the terrain grid spans the context radius
gx,gy=np.meshgrid(np.arange(N)*STEP+MIN,np.arange(N)*STEP+MIN)
dist,idx=tree.query(np.c_[gx.ravel(),gy.ravel()],k=8)
grid=np.median(ground[idx,2],axis=1).reshape(N,N);grid=gaussian_filter(grid,.8)
def elev(x,y):
 fx=np.clip((np.asarray(x)-MIN)/STEP,0,N-1.00001);fy=np.clip((np.asarray(y)-MIN)/STEP,0,N-1.00001);ix=np.floor(fx).astype(int);iy=np.floor(fy).astype(int);fx=fx-ix;fy=fy-iy
 a,b,c,d=grid[iy,ix],grid[iy,ix+1],grid[iy+1,ix],grid[iy+1,ix+1]
 return np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,d+(c-d)*(1-fx)+(b-d)*(1-fy))
# Streets stand on the LiDAR ground under the carriageway, not on the 3 m terrain grid, which climbs into raised lots beside a
# hillside sidewalk; the terrain grid is held just under them (pipeline/street_ground.py).
_carriageway=unary_union([mobility.road_polygon(w['g'],mobility.spec(w['tags'])).intersection(boundary) for w in ways if w['tags'].get('highway') in ['trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service'] and w['g'].geom_type=='LineString'])
elev,grid=street_ground.build(ground,_carriageway,grid,MIN,STEP,N)

# Apply the renderer coordinate frame when each node is added; a second graph-wide
# transform recomputes the changing graph hash for every child (quadratic in area size).
VIEW_FRAME=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float)
scene=trimesh.Scene();features=[];batches=collections.defaultdict(lambda:[[],[],[]]);counts=collections.Counter();materials={}
palette={'asphalt':'#41484b','sidewalk':'#b4b5ac','brick':'#97604b','brick2':'#b4795d','stone':'#c5beb0','dark':'#283c42','glass':'#4b7686','frame':'#c4d0d1','roof':'#737b79','grass':'#566e4b','trunk':'#75644d','leaf':'#425e48','metal':'#546661','paint':'#e4e5d8','yellow':'#d7b351','rail':'#87918e','water':'#386c7a'}
def rgba(c):
 c=palette.get(c,c).lstrip('#');return [int(c[i:i+2],16) for i in (0,2,4)]+[255]
material_bank=Materials(ROOT/'material-assets',palette)
batches=collections.defaultdict(lambda:[[],[],[],[]]);_batch_vertices=collections.Counter()
front_trees=[]   # (x, y, ground z, height, crown radius): small trees the facade readings stand in front of walls (facade_detail)
CLIP=None
def set_clip(halfspaces):
 """While a gable end or a part's stretch of a shared frontage is compiled, everything emitted is cut to the given
 half-spaces: (normal, offset) keeps normal.p <= offset (a roof plane z <= a*x+b*y+c is ((-a,-b,1), c)); an item
 ('solid', manifold) keeps what lies inside that solid (a wall cut to its measured parapet outline)."""
 global CLIP;CLIP=halfspaces
def _clip(mesh):
 import manifold3d
 cut=[];solids=[c_[1] for c_ in CLIP if isinstance(c_[0],str) and c_[0]=='solid']
 if solids:
  # a solid clip (a wall cut to a measured outline, facades.parapet_solid): closed boxes and prisms are intersected with
  # it; an open mesh keeps its faces whose centre lies inside the solid's bounds
  try:
   m=manifold3d.Manifold(manifold3d.Mesh64(np.asarray(mesh.vertices,float),np.asarray(mesh.faces,np.uint64)))
   if m.is_empty():raise ValueError('not a closed solid')
   for sd in solids:m=m^sd
   raw=m.to_mesh64();tri=np.asarray(raw.tri_verts)
   if not len(tri):return None
   mesh=trimesh.Trimesh(np.asarray(raw.vert_properties)[:,:3],tri,process=False)
  except Exception:pass
 for nrm,off in [c_ for c_ in CLIP if not isinstance(c_[0],str)]:
  nrm=np.asarray(nrm,float);d=np.asarray(mesh.vertices)@nrm-off
  if (d<=1e-6).all():continue
  if (d>=-1e-6).all():return None
  cut.append((nrm,off))
 if not cut:return mesh
 # closed boxes and prisms are trimmed as solids (capped cut faces); anything else is sliced open
 try:
  m=manifold3d.Manifold(manifold3d.Mesh64(np.asarray(mesh.vertices,float),np.asarray(mesh.faces,np.uint64)))
  if m.is_empty():raise ValueError('not a closed solid')
  for nrm,off in cut:
   L=float(np.linalg.norm(nrm));m=m.trim_by_plane((-nrm/L).tolist(),-off/L)
  raw=m.to_mesh64();tri=np.asarray(raw.tri_verts)
  return trimesh.Trimesh(np.asarray(raw.vert_properties)[:,:3],tri,process=False) if len(tri) else None
 except Exception:
  for nrm,off in cut:
   mesh=trimesh.intersections.slice_mesh_plane(mesh,-nrm,nrm*off/float(nrm@nrm),cap=False)
   if mesh is None or len(mesh.faces)==0:return None
  return mesh
_roof_component_cutters=[]
_crown_clearance=None
def put(mesh,key,color,uv=None):
 if _crown_clearance is not None and key=='vegetation' and color in ('leaf1','leaf2','leaf3'):
  mesh=_crown_clearance.leaves(mesh)
  if mesh is None:return
 if len(mesh.vertices)==0:return
 if _roof_component_cutters:
  from pipeline.mesh_attributes import transfer_metric_uv
  original=mesh
  if uv is not None:
   original=mesh.copy();original.metadata['uvMeters']=np.asarray(uv,float)
  mesh=transfer_metric_uv(original,roof_shell.clip_component_intersections(mesh,_roof_component_cutters))
  if mesh is None or not len(mesh.faces):return
  if uv is not None:uv=mesh.metadata['uvMeters']
 if CLIP is not None and uv is None:
  from pipeline.mesh_attributes import transfer_metric_uv
  original=mesh;mesh=transfer_metric_uv(original,_clip(mesh))
  if mesh is None or len(mesh.vertices)==0:return
 material,scale=material_bank.get(color);batchkey=key+'___'+material.name
 # a running vertex count per batch: summing the batch's pieces on every call was quadratic in its size (a tree adds
 # 280 pieces to one batch: hours for a few thousand trees)
 vertices,faces,uvs,normals=batches[batchkey];off=_batch_vertices[batchkey]
 # Split face corners to preserve hard normals and per-face, metric texture coordinates.
 inds=mesh.faces.ravel();v=np.asarray(mesh.vertices)[inds];norm=np.repeat(mesh.face_normals,3,axis=0)
 metric_uv=np.asarray(mesh.metadata.get('uvMeters',[]),float)
 if uv is None and metric_uv.shape==(len(mesh.vertices),2) and np.isfinite(metric_uv).all():
  # Curved surface builders supply one continuous metric chart. Projecting
  # world positions along a different tangent per facet resets texture phase.
  coord=metric_uv[inds]/np.asarray(scale)
 elif uv is None:
  axis=np.argmax(np.abs(norm),axis=1);coord=np.zeros((len(v),2));horizontal=axis==2
  coord[horizontal]=v[horizontal,:2]/scale
  mask=~horizontal;tangent=np.c_[-norm[mask,1],norm[mask,0]];tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
  coord[mask,0]=np.sum(v[mask,:2]*tangent,axis=1)/scale[0];coord[mask,1]=v[mask,2]/scale[1]
 else:coord=np.asarray(uv)[inds]
 if mesh.metadata.get('smoothLeafSurface') and color.startswith('groundcover#'):
  # Project on fixed cardinal planes; rotating the texture with each tiny
  # facet created triangular phase jumps even under continuous leaf normals.
  axis=np.argmax(np.abs(norm),axis=1);coord=np.stack([np.where(axis==0,v[:,1],v[:,0]),np.where(axis==2,v[:,1],v[:,2])],axis=1)/np.asarray(scale)
  norm=np.asarray(mesh.vertex_normals)[inds]
 # kept at the GLB's own precision (float32, int32) until export: in float64 an area of rich facade readings outgrew the
 # machine's memory before its last building
 vertices.append(v.astype(np.float32));faces.append((np.arange(len(v)).reshape(-1,3)+off).astype(np.int32));uvs.append(coord.astype(np.float32));normals.append(norm.astype(np.float32));_batch_vertices[batchkey]+=len(v)
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
 # Intersections at curb/track boundaries can split a touching polygon during
 # densification. Normalize and enumerate the result before triangulating.
 for a in polys(make_valid(segmentize(make_valid(p),1.5))):
  if a.area<.02:continue
  if a.area<4:
   v,f=trimesh.creation.triangulate_polygon(a,engine='earcut')
  else:
   coords=np.concatenate([np.array(r.coords)[:-1] for r in [a.exterior]+list(a.interiors)])
   xmin,ymin,xmax,ymax=a.bounds;xx,yy=np.meshgrid(np.arange(math.ceil((xmin-MIN)/STEP),math.floor((xmax-MIN)/STEP)+1)*STEP+MIN,np.arange(math.ceil((ymin-MIN)/STEP),math.floor((ymax-MIN)/STEP)+1)*STEP+MIN)
   inside=contains_xy(a,xx.ravel(),yy.ravel());inner=np.c_[xx.ravel()[inside],yy.ravel()[inside]];v=np.unique(np.r_[coords,inner],axis=0)
   if len(v)<3:continue
   f=Delaunay(v).simplices;triangles=polygon_array(v[f]);f=f[covered_by(triangles,a.buffer(.002))]
  vv=np.c_[v,elev(v[:,0],v[:,1])+offset];put(trimesh.Trimesh(vv,f,process=False),key,color)

def cylinder(a,b,r,key,color,sections=8):
 if _crown_clearance is not None and key=='vegetation' and color=='bark':
  b=_crown_clearance.limb(a,b)
  if b is None:return
 v=np.asarray(b)-a;l=np.linalg.norm(v)
 if l<.001:return
 m=trimesh.creation.cylinder(radius=r,height=l,sections=sections);T=trimesh.geometry.align_vectors([0,0,1],v/l);T[:3,3]=(np.asarray(a)+b)/2;m.apply_transform(T);put(m,key,color)

# Garage doors at the foot of a read wall: the 3 m terrain grid smoothed the hillside over the driveways cut into it,
# burying the lower part of a fifth of the garage doors the photographs show (Palash's review, 10-15). The grid nodes in
# front of each door are held down to the LiDAR's own ground at its threshold, rising 5 cm a metre toward the street.
def _hold_driveways():
 kd=cKDTree(ground[:,:2]);held=0;cap=np.full((N*3,N*3),np.inf,np.float32)   # the cap on elev, 1 m cells
 for _p in PROFILES:
  try:
   n=np.asarray(_p['normal'],float)[:2];n=n/np.linalg.norm(n);t=np.array([-n[1],n[0]]);u0,u1=map(float,_p['frameU']);v=float(_p['planeV']);H=float(_p.get('heightRefM') or 0.)
  except (KeyError,TypeError,ValueError,ZeroDivisionError):continue
  for o in _p.get('openings') or []:
   if (o.get('style') or {}).get('kind')!='rollup' or not H:continue
   x0,y0,x1,y1=o['rect']
   if y0*H>.6:continue                                            # a door on an upper storey is no driveway
   a,b=u0+x0*(u1-u0),u0+x1*(u1-u0);c=n*v+t*(a+b)/2
   q=ground[kd.query_ball_point(c+n*.8,max(1.5,(b-a)/2+.5))]
   if not len(q):continue
   rel=q[:,:2]-n*v;along=rel@t;out=rel@n
   z=q[(along>a-.2)&(along<b+.2)&(out>.15)&(out<1.5),2]
   if len(z)<3:continue
   sill=float(np.percentile(z,25))
   ix0,ix1=int((min(c[0],c[0]+n[0]*4)-4-MIN)//STEP),int((max(c[0],c[0]+n[0]*4)+4-MIN)//STEP)+1
   iy0,iy1=int((min(c[1],c[1]+n[1]*4)-4-MIN)//STEP),int((max(c[1],c[1]+n[1]*4)+4-MIN)//STEP)+1
   for iy in range(max(0,iy0),min(N,iy1+1)):
    for ix in range(max(0,ix0),min(N,ix1+1)):
     d=np.array([ix*STEP+MIN,iy*STEP+MIN])-n*v;al,ou=d@t,d@n
     if a-1.5<=al<=b+1.5 and 0.<=ou<=4.:
      zc=sill+.05*ou
      if grid[iy,ix]>zc:grid[iy,ix]=zc;held+=1
   # the ground everything else stands on (sidewalk, planting, the wall's foot) dips with it in front of the door
   cx,cy=np.meshgrid(np.arange(int((c[0]-6-MIN)),int((c[0]+6-MIN))+1)+.5,np.arange(int((c[1]-6-MIN)),int((c[1]+6-MIN))+1)+.5)
   d=np.c_[cx.ravel()+MIN,cy.ravel()+MIN]-n*v;al,ou=d@t,d@n;k=(al>=a-.3)&(al<=b+.3)&(ou>=0.)&(ou<=4.)
   iy_,ix_=np.clip(cy.ravel()[k].astype(int),0,N*3-1),np.clip(cx.ravel()[k].astype(int),0,N*3-1)
   cap[iy_,ix_]=np.minimum(cap[iy_,ix_],sill+.05*ou[k])
 return held,cap
counts['drivewayGridNodesHeld'],_DRIVE_CAP=_hold_driveways()
_elev_free=elev
def elev(x,y):
 z=_elev_free(x,y);xs=np.asarray(x,float);ys=np.asarray(y,float)
 c=_DRIVE_CAP[np.clip((ys-MIN).astype(int),0,_DRIVE_CAP.shape[0]-1),np.clip((xs-MIN).astype(int),0,_DRIVE_CAP.shape[1]-1)]
 return np.minimum(z,c)
# Ground surface uses LiDAR elevations, with an explicit circular model boundary.
vv=np.c_[gx.ravel(),gy.ravel(),grid.ravel()];ff=[]
for y in range(N-1):
 for x in range(N-1):
  i=y*N+x
  if math.hypot(gx[y,x],gy[y,x])<CONFIG['region']['contextRadiusM']:ff.extend([[i,i+1,i+N],[i+1,i+N+1,i+N]])
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

grouped=collections.defaultdict(list)
for c in candidates:
 if c['osm']:grouped[c['osm']['id']].append(c)
consolidations=[]
for oid,parts in grouped.items():
 if len(parts)<2:continue
 w=parts[0]['osm'];old_union=unary_union([p['g'] for p in parts]);coverage=old_union.intersection(w['g']).area/max(1,w['g'].area);agreement=old_union.intersection(w['g']).area/max(1,old_union.area)
 if coverage>.55 and agreement>.60:
  candidates=[c for c in candidates if c not in parts];candidates.append({'id':'osm_'+oid,'g':w['g'],'old':{},'osm':w,'fp':'OSM; fragmented municipal identity reconciled'})
  consolidations.append({'osmWayId':oid,'municipalParts':len(parts),'coverage':round(coverage,3),'agreement':round(agreement,3),'decision':'one OSM identity envelope; source disagreement retained'})
candidates,spatial_audit=spatial_rules.reconcile(candidates)
# --massing FILE: an outline the street photographs and the LiDAR contradict is dropped (a driveway or garden read as a
# building) or replaced by the outline they show (a garage set back up its lot); every entry says why in its file.
MASSING=json.load(open(_args.massing)) if _args.massing else {}
if MASSING:
 candidates=[c for c in candidates if not MASSING.get(c['id'],{}).get('drop')]
 for c in candidates:
  if MASSING.get(c['id'],{}).get('outline'):c['g']=Polygon(MASSING[c['id']]['outline']).buffer(0)
 # a building the outlines miss but the photographs and the LiDAR both show (a low house beside a garden wall)
 for _k,_m in MASSING.items():
  if _m.get('add') and _m.get('outline'):candidates.append({'id':_k,'g':Polygon(_m['outline']).buffer(0),'old':{},'osm':None,'fp':'street photographs and LiDAR (areas/ring/massing)'})
def mass_spec(c):
 """The massing given for this part (areas/ring/massing): its wall top, its roof form, the photographs' and LiDAR's
 reading of it; a part named in a building's `parts` carries its own."""
 m=c['massSpec'] if c.get('massSpec') is not None else (MASSING.get(c['id'],{}) or (MASSING.get(c.get('sourceId'),{}) if c.get('sourceId',c['id'])==c['id'] else {}))
 return m if (m.get('wallTopM') is not None or m.get('roof')) else None
MASS_DUMP=[]
foundation_records=[];mobility_records=[];duplicate_triangles=0;panel_audits=[];structure_returns={};structure_context={};structure_raster={};roof_envelope_records=[]
for c in candidates:
 c['full_area']=c['g'].area;c['clipped']=not boundary.covers(c['g'])
 # A landmark entry may keep its footprint whole past the model boundary; its builder lays ground under the rest.
 _lm=landmarks.entry(c)
 if _lm and _lm.get('extendBeyondBoundary'):c['extendedBeyondBoundary']=c['clipped'];c['clipped']=False
 else:c['g']=c['g'].intersection(boundary)
all_buildings=unary_union([c['g'] for c in candidates]);ground_kd=cKDTree(ground[:,:2])
# The points sorted into 8 m cells once: an outline's points are gathered from the cells under its bounds (the same
# points, in their original order), not from a pass over every point (20 million in a larger area, read per outline).
_GC=8.;_gx0,_gy0=float(pts[:,0].min()),float(pts[:,1].min());_gnx=int((pts[:,0].max()-_gx0)//_GC)+1;_gny=int((pts[:,1].max()-_gy0)//_GC)+1
_gkey=((pts[:,1]-_gy0)//_GC).astype(np.int64)*_gnx+((pts[:,0]-_gx0)//_GC).astype(np.int64)
_gorder=np.argsort(_gkey,kind='stable');_gstart=np.searchsorted(_gkey[_gorder],np.arange(_gnx*_gny+1))
def get_box(a,b,c,d):
 """The non-noise points with a<=x<=c and b<=y<=d, in their original order."""
 i0,i1=max(0,int((a-_gx0)//_GC)),min(_gnx-1,int((c-_gx0)//_GC));j0,j1=max(0,int((b-_gy0)//_GC)),min(_gny-1,int((d-_gy0)//_GC))
 if i1<i0 or j1<j0:return pts[:0]
 idx=np.sort(np.concatenate([_gorder[_gstart[j*_gnx+i0]:_gstart[j*_gnx+i1+1]] for j in range(j0,j1+1)]))
 q=pts[idx];return q[(q[:,0]>=a)&(q[:,0]<=c)&(q[:,1]>=b)&(q[:,1]<=d)&(q[:,3]!=7)&(q[:,3]!=18)]
def get_points(g):
 q=get_box(*g.bounds);return q[contains_xy(g,q[:,0],q[:,1])]
# The measured surface top on a 1 m grid (highest non-noise return per cell), for height-aware wall visibility.
_st_x0,_st_y0=float(pts[:,0].min()),float(pts[:,1].min());_st_W=int(pts[:,0].max()-_st_x0)+2;_st_H=int(pts[:,1].max()-_st_y0)+2
_st=np.full((_st_H,_st_W),-1e9,np.float32);_sm=(pts[:,3]!=7)&(pts[:,3]!=18)
np.maximum.at(_st,((pts[_sm,1]-_st_y0).astype(int),(pts[_sm,0]-_st_x0).astype(int)),pts[_sm,2].astype(np.float32))
def surfaceTop(x,y):
 i,j=int(y-_st_y0),int(x-_st_x0)
 return float(_st[i,j]) if 0<=i<_st_H and 0<=j<_st_W and _st[i,j]>-1e8 else -1e9
# One outline can hold several buildings; the measured roof separates them before any per-building rule runs.
# Every part keeps the source identity as its join key for external observations; only its mesh key changes.
from pipeline import building_parts
_split=[];part_audit=[]
for c in candidates:
 if ONLY is not None and c['id'] not in ONLY_SRC:_split.append(c);continue
 if landmarks.entry(c):_split.append(c);continue       # a landmark's builder owns the whole outline
 if mass_spec(c) is not None and not MASSING.get(c['id'],{}).get('parts'):_split.append(c);continue   # one given massing for the outline
 if MASSING.get(c['id'],{}).get('parts'):
  # parts the photographs and the LiDAR show (a lower wing, a set-back top storey), in place of the LiDAR's own split
  for _i,_mp in enumerate(MASSING[c['id']]['parts']):
   _q=Polygon(_mp['outline']).buffer(0)
   _split.append({**c,'g':_q,'sourceId':c['id'],'id':_mp.get('id') or (c['id'] if _i==0 else f"{c['id']}_p{20+_i}"),'full_area':_q.area,'partOf':c['id'],'sourceG':c['g'],'massSpec':_mp,'footOf':c['g'].centroid})
  continue
 _a={};_parts=building_parts.split(c['g'],get_points(c['g'].buffer(-.65))[:,:3],float(elev(c['g'].centroid.x,c['g'].centroid.y)),audit=_a)
 if len(_parts)<2:_split.append(c);continue
 _parts.sort(key=lambda q:-q.area);part_audit.append({'sourceId':c['id'],'parts':[round(q.area,1) for q in _parts],**_a})
 for _i,_q in enumerate(_parts):_split.append({**c,'g':_q,'sourceId':c['id'],'id':c['id'] if _i==0 else f"{c['id']}_p{_i}",'full_area':_q.area,'partOf':c['id'],'sourceG':c['g']})
candidates=_split

landmark_sites=[];landmark_records=[]
# What each wall faces: every outline with its source identity (split parts share one), and a street corridor of the
# mapped carriageways and pedestrian ways (5 m either side of the centre line; alleys 4 m; parking aisles excluded).
from shapely.strtree import STRtree
building_index=STRtree([c['g'] for c in candidates]);building_sources=[c.get('sourceId',c['id']) for c in candidates]
street_corridor=unary_union([w['g'].buffer(5.) for w in ways if w['g'].geom_type=='LineString' and w['tags'].get('highway') in ('trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','pedestrian')]
                            +[w['g'].buffer(4.) for w in ways if w['g'].geom_type=='LineString' and w['tags'].get('highway')=='service' and w['tags'].get('service') not in ('parking_aisle','driveway')]
                            +[w['g'] for w in ways if w['g'].geom_type in ('Polygon','MultiPolygon') and w['tags'].get('highway')=='pedestrian'])
import os as _os;_GEN_TRACE=bool(_os.environ.get('GEN_TRACE'))
for c in ([] if _args.streets_only else [c for c in candidates if ONLY is None or c.get('sourceId',c['id']) in ONLY_SRC]):
 id,g=c['id'],c['g'];og=c['old'];tags=c['osm']['tags'] if c['osm'] else {};name=tags.get('name') or (tags.get('addr:housenumber','').replace(';',' / ')+' '+tags.get('addr:street','')).strip() or 'Building '+og.get('sf16_bldgid',id)
 if _GEN_TRACE:print('START',id,name,flush=True)   # GEN_TRACE=1: name each building as it starts (a slow one shows itself)
 stadium=c['full_area']>18000
 sample=get_points(g.buffer(-.65));_ft=c.get('footOf') or g.centroid;z0=float(elev(_ft.x,_ft.y))   # the parts of a given massing share their building's foot
 # areas/ring/massing: a foot the photographs place off the terrain grid at the outline's centre (a garage at the head of a drive)
 z0+=float(MASSING.get(c.get('sourceId',id),{}).get('baseOffsetM',0.)) if MASSING else 0.;roof=sample[sample[:,2]>z0+3]
 # A small free-standing outline under tree canopy: its returns spread over the canopy's whole height instead of
 # gathering in a roof layer (its densest 4 m band holds under 40 % of them). Its roof is the lowest 2 m band holding
 # 5 % of them, and the canopy above is not the building (a park's 56 m2 shelter stood 25 m tall as a dark tower among
 # its trees). An outline with another building within 3 m is left alone: in a row the spread can be a stepped roof or
 # an outline taking in its neighbours, and a stub would open a gap in the street wall.
 under_canopy=None
 if g.area<200 and not any(building_sources[j]!=c.get('sourceId',id) and candidates[j]['g'].distance(g)<3. for j in building_index.query(g.buffer(3.))):
  _h=np.sort(sample[:,2]-z0);_h=_h[_h>1.]
  if len(_h)>=50 and (np.searchsorted(_h,_h+4.,'right')-np.arange(len(_h))).max()<.4*len(_h):
   _k=np.flatnonzero(np.histogram(_h,bins=np.arange(1.,_h.max()+2.,2.))[0]>=.05*len(_h))
   if len(_k):under_canopy=1.+2.*(_k[0]+1);roof=roof[roof[:,2]<z0+under_canopy+.5]
 roof_step=2.0 if stadium else 1.5
 wxmin,wymin,wxmax,wymax=g.bounds;wnx=max(2,math.ceil((wxmax-wxmin)/roof_step));wny=max(2,math.ceil((wymax-wymin)/roof_step))
 # Roof heights come from roof SURFACES: each raster cell reads its highest dense thin layer of returns,
 # and returns above that layer (signs, antennas, railings, plant) are dropped before any height statistic.
 rr=None
 panels=[]
 if len(roof)>50:
  rr,surface_returns=spatial_rules.surface_raster(roof,wxmin,wymax,roof_step,wnx,wny)
  # Thin tall linear structures above the roof surface (signs, screens) are panels, not roof volume.
  around=get_box(wxmin-10,wymin-10,wxmax+10,wymax+10)[:,:3]
  panel_audit=[];panels=spatial_rules.rooftop_panels(roof,surface_returns,rr,wxmin,wymax,roof_step,context=around,audit=panel_audit)
  if panel_audit:panel_audits.append({'buildingId':id,'clusters':panel_audit})
  if (~surface_returns).sum()>=40:structure_returns[id]=roof[~surface_returns];structure_context[id]=around;structure_raster[id]=(rr,wxmin,wymax,roof_step)
  roof=roof[surface_returns]
 if _args.roof_evidence:
  _ed=OUT/'roof-evidence';_ed.mkdir(exist_ok=True);np.savez_compressed(_ed/(id+'.npz'),points=roof[:,:3])
 # The tier raster (base roof, tiers, drapes) runs along the outline's own axes (spatial_rules.OutlineFrame), over the
 # surface returns kept above; its products are turned back to the world. Which returns count as roof surface (for the
 # plane fits, panels and wall levels) is still decided on the grid those rules were validated on.
 frame=spatial_rules.OutlineFrame(g);gF=frame.geom(g)
 xmin,ymin,xmax,ymax=gF.bounds;nx=max(2,math.ceil((xmax-xmin)/roof_step));ny=max(2,math.ceil((ymax-ymin)/roof_step))
 if rr is not None and len(roof)>50:rr,_=spatial_rules.surface_raster(frame.points(roof),xmin,ymax,roof_step,nx,ny)
 if MASSING.get(c.get('sourceId',id),{}).get('heightM'):
  height=float(MASSING[c.get('sourceId',id)]['heightM']);lo=z0+height;top=lo;source='street photographs (areas/ring/massing)'
 elif len(roof)>50:
  lo,top=np.quantile(roof[:,2],[.55,.97]);height=max(3.,float(top-z0));source='USGS LiDAR 2023'
 elif under_canopy is not None:
  height=max(3.,under_canopy);lo=z0+height;top=lo;source='USGS LiDAR 2023 (roof under canopy: its lowest dense layer)'
 else:
  try:height=float(tags.get('height',og.get('hgt_median_m','12')).split()[0])
  except:height=12.
  lo=z0+height;top=lo;source='Catalog estimate'
 # The foundation is drawn after the facades: a door or garage a street lower than the building's ground opens into it,
 # and the facades say where (pockets left out of it).
 def _foundation(pockets=None,c=c,z0=z0):foundation_records.append(spatial_rules.foundation(globals(),{**c,'g':c['g'].difference(unary_union(pockets))} if pockets else c,z0))
 # A split part fits a shared frontage's levels on the whole outline's returns, as it frames it on the whole outline.
 resolved=envelopes.prepare(globals(),c,z0,get_points(c['sourceG'].buffer(-.65)) if c.get('sourceG') is not None else sample,ENVELOPES);envelope_mask=envelopes.replacement_mask(resolved)
 profile_strip=spatial_rules.perimeter_strip(globals(),c,PROFILES);tier_mask=envelope_mask.union(profile_strip)
 if resolved:name=resolved[0]['profile'].get('displayName',name)
 lm=landmarks.entry(c)
 if lm:
  # A landmark's builder owns the whole footprint (surfaces, walls, field, props); observed street frontages are
  # compiled on top of it, reduced to their facade planes.
  lm_view=landmarks.frontage_view(resolved,lm);lm_ctx=dict(globals());lm_ctx['landmarks_view']=lm_view
  built=landmarks.build(lm_ctx,c,z0,resolved,lm);landmark_sites.append(built['site']);landmark_records.append({'buildingId':id,**built['record']})
  er,ec=envelopes.compile(globals(),c,z0,lm_view);envelope_records.extend(er);facade_records.extend(er);counts['windowAssemblies']+=ec
  lettering_records.extend(lettering.compile(globals(),c,z0,LETTERING_DATA['signs'],resolved))
  for r in resolved:envelope_frontages.append({'frame':r['frame'],'label':r['profile'].get('displayName',name)+' / '+r['profile']['id']})
  height=max([built['height']]+[r['heightM'] for r in er])
  features.append({'id':id,'name':name,'height':round(height,2),'heightSource':'USGS LiDAR 2023','footprintSource':c['fp'],'outlineMetadata':c['osm'].get('metadata',{}) if c['osm'] else {'captureAgeKnown':False},'roofParts':built['roofParts'],'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'clippedAtBoundary':c['clipped'],'pointCount':len(roof),'facade':'Landmark builder: the measured LiDAR surface and playing field, observed street frontages on top, authored props where named.','facadeEvidence':'Landmark: LiDAR surface + return intensity + observed frontages','fittedWalls':len(er),'totalWalls':len(er),'landmark':lm.get('label'),'sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id'] if c['osm'] else None})
  _foundation();print(name,round(height,1),'landmark',flush=True);continue
 style=facades.prior(tags,height,c['full_area']);color=style['material']
 roofparts=0
 roof_grid_audit={};roof_patches=[];roof_bands=[];roof_xy=None;venue=None;patch_sides=[];outer_rounded=[]
 if rr is not None and len(roof)>50:
  rr,roof_known,roof_xy,roof_grid_audit=roof_envelope.prepare_grid(rr,gF,xmin,ymax,roof_step)
  base_roof=roof_envelope.base_level(rr,roof_known,roof_step,g.area) if rr is not None else z0+height
  if stadium:base_roof=max(z0+6,min(base_roof,z0+12))
  else:base_roof=max(z0+4,min(base_roof,lo))
 else:base_roof=z0+height;rr=None
 if _args.roof_shell:
  MS=None
  # Ordinary mass-height/tier statistics intentionally use an eroded footprint.
  # A steep observed roof face can fit entirely in that excluded edge strip.
  # Supply separately filtered, exactly owned boundary evidence to the shell;
  # do not change those established interior statistics or neighboring domains.
  _boundary_domain=g
  _overlap_neighbors=[candidates[int(_j)]['g'] for _j in building_index.query(g)
                      if building_sources[int(_j)]!=c.get('sourceId',id)
                      and candidates[int(_j)]['g'].intersection(g).area>1e-6]
  if _overlap_neighbors:_boundary_domain=g.difference(unary_union(_overlap_neighbors))
  _boundary_roof=get_points(_boundary_domain) if not _boundary_domain.is_empty else np.empty((0,5))
  _boundary_roof=_boundary_roof[_boundary_roof[:,2]>z0+3.]
  if under_canopy is not None:_boundary_roof=_boundary_roof[_boundary_roof[:,2]<z0+under_canopy+.5]
  if len(_boundary_roof)>50:
   _,_boundary_surface_mask=spatial_rules.surface_raster(_boundary_roof,wxmin,wymax,roof_step,wnx,wny)
   _boundary_roof=_boundary_roof[_boundary_surface_mask]
  roof_observation_audit={"status":"no-observations","eave":None,"features":{"status":"no-observations","accepted":[],"rejected":[]}}
  _eave_candidates=[];_roof_edge_candidates=[];_eave_constraints=[];_roof_colors=[];_host_candidates=[];_host_exclusions=[];_frontage_roofline_sides=[];_source_front_inputs=[]
  for _obs in ROOF_OBSERVATIONS:
   if _obs.get('confidence') not in ('medium','high'):continue
   _side=str(_obs.get('side',''))
   _profiles=[_p for _p in PROFILES if str(_p.get('id','')).removeprefix('regen-')==_side]
   _exact=[];_fallback=[]
   for _p in _profiles:
    _match=_p.get('match') or {};_pid=_match.get('partId')
    if _pid is not None and str(_pid)==str(id):_exact.append(_p)
    elif _pid is None and (not _match or str(_match.get('featureId'))==str(c.get('sourceId',id))):_fallback.append(_p)
   _joined=_exact if _exact else _fallback
   if not _joined:continue
   _source_front_inputs.extend((_obs,_p) for _p in _joined)
   _edge_range=(_obs.get('eaveEvidence') or {}).get('visibleRoofEdgeHeightRangeM')
   if isinstance(_edge_range,(list,tuple)) and len(_edge_range)==2:
    try:
     _edge_lo,_edge_hi=map(float,_edge_range)
     if math.isfinite(_edge_lo) and math.isfinite(_edge_hi) and _edge_hi>=_edge_lo:
      _roof_edge_candidates.append(((_edge_lo+_edge_hi)/2.,_side,_obs.get('confidence'),[_edge_lo,_edge_hi]))
    except (TypeError,ValueError):pass
   _semantic_eave=isinstance(_obs.get('eaveM'),(int,float)) and math.isfinite(float(_obs['eaveM']))
   _host_anchor=float(_obs['eaveM']) if _semantic_eave else (_roof_edge_candidates[-1][0] if _roof_edge_candidates and _roof_edge_candidates[-1][1]==_side else None)
   if _obs.get('hostRoofKind') in ('hip','gable','mansard') and _host_anchor is not None:
    _host_candidates.append({'kind':_obs['hostRoofKind'],'eaveZ':z0+_host_anchor,'ridgeZ':z0+float(_obs['hostRidgeM']) if _obs.get('hostRidgeM') is not None else None,'confidence':_obs['confidence'],'side':_side,
                             'anchorRole':'eave' if _semantic_eave else 'visible-roof-edge',
                             'frontageKind':_obs.get('frontageKind'),
                             'anchorUncertaintyM':float(_obs.get('anchorUncertaintyM',0.))})
   _hex=_obs.get('roofColor')
   if isinstance(_hex,str) and len(_hex)==7 and _hex.startswith('#'):
    try:_roof_colors.append([int(_hex[i:i+2],16) for i in (1,3,5)])
    except ValueError:pass
   try:_eave=float(_obs['eaveM'])
   except (TypeError,ValueError):_eave=None
   if _eave is not None and not math.isfinite(_eave):_eave=None
   if _eave is not None:_eave_candidates.append((_eave,_side,_obs.get('eaveConfidence',_obs.get('confidence'))))
   _boundary_level=_eave if _eave is not None else (_roof_edge_candidates[-1][0] if _roof_edge_candidates and _roof_edge_candidates[-1][1]==_side else None)
   for _profile in _joined:
    if _profile.get('normal') is None or not _profile.get('frameU') or _profile.get('planeV') is None:continue
    _n=np.asarray(_profile['normal'],float);_n/=np.linalg.norm(_n)
    for _component in _obs.get('features',[]):
     if _component.get('confidence') in ('medium','high') and all(isinstance(_component.get(_k),(int,float)) for _k in ('x0M','x1M')):
      _host_exclusions.append({'normal':_n.tolist(),'planeV':float(_profile['planeV']),'u0':float(_profile['frameU'][0])+float(_component['x0M']),'u1':float(_profile['frameU'][0])+float(_component['x1M']),'maxDepthM':6.})
   if _boundary_level is not None:
    _n=np.asarray(_profile['normal'],float);_n/=np.linalg.norm(_n);_t=np.array([-_n[1],_n[0]])
    _a=_n*float(_profile['planeV'])+_t*float(_profile['frameU'][0]);_b=_n*float(_profile['planeV'])+_t*float(_profile['frameU'][1])
    _roofline=_obs.get('frontageRoofline') or {}
    _roofline_points,_roofline_measurement=roof_observations.measured_frontage(_roofline)
    if _roofline:
     roof_observation_audit.setdefault('frontageMeasurements',[]).append({'side':_side,**_roofline_measurement})
    _roofline_ok=isinstance(_roofline_points,list) and len(_roofline_points)>=2
    if _roofline_ok:
     try:
      _rp=np.asarray(_roofline_points,dtype=float)
      _roofline_ok=(_rp.ndim==2 and _rp.shape[1]==2 and np.isfinite(_rp).all()
                    and np.all(np.diff(_rp[:,0])>0)
                    and _rp[0,0]>=-.25 and _rp[-1,0]<=float(_profile['frameU'][1])-
                    float(_profile['frameU'][0])+.25)
     except (TypeError,ValueError):_roofline_ok=False
    if _roofline_ok:
     _frontage_roofline_sides.append(_side)
     _world=_n[None,:]*float(_profile['planeV'])+_t[None,:]*(float(_profile['frameU'][0])+_rp[:,0,None])
     for _j in range(len(_rp)-1):
      if np.linalg.norm(_world[_j+1]-_world[_j])<.1:continue
      _eave_constraints.append({'a':_world[_j].tolist(),'b':_world[_j+1].tolist(),
                                'z':z0+float(_rp[_j,1]),'z1':z0+float(_rp[_j+1,1]),
                                'role':'gable-frontage-roofline','sourceSide':_side,
                                'sourceConfidence':_roofline.get('confidence'),
                                'apexEstimated':bool(_roofline.get('apexEstimated',False))})
     roof_observation_audit.setdefault('frontageRooflines',[]).append({
      'side':_side,'status':'accepted-as-local-frontage-boundary','pointsM':_rp.tolist(),
      'apexEstimated':bool(_roofline.get('apexEstimated',False)),
      'semantic':'front facade roofline only; not a whole-building ridge'})
    else:
     _eave_constraints.append({'a':_a.tolist(),'b':_b.tolist(),'z':z0+_boundary_level,
                               'role':'eave' if _eave is not None else 'visible-roof-edge',
                               'uncertaintyM':float(_obs.get('anchorUncertaintyM',0.))})
  if _eave_candidates or _roof_edge_candidates:
   _boundary_candidates=_eave_candidates if _eave_candidates else [(q[0],q[1],q[2]) for q in _roof_edge_candidates]
   _raw_eave=float(np.median([q[0] for q in _boundary_candidates]))
   _low=z0+3.
   _high=max(_low,min(float(base_roof),float(lo))) if _eave_candidates else max(_low,float(lo))
   base_roof=max(_low,min(_raw_eave+z0,_high))
   if _eave_candidates:
    roof_observation_audit['eave']={'status':'accepted' if all(q[2] in ('medium','high') for q in _eave_candidates) else 'uncertain-prior','rawEaveM':_raw_eave,'baseRoofZ':base_roof,
                                    'sides':[q[1] for q in _eave_candidates],
                                    'confidence':[q[2] for q in _eave_candidates],
                                    'boundsZ':[ _low,_high ]}
   else:
    roof_observation_audit['eave']={'status':'no-matched-confident-eave','baseRoofZ':base_roof}
    roof_observation_audit['visibleRoofEdge']={'status':'accepted-as-boundary-anchor','heightRangeM':_roof_edge_candidates[0][3],
      'sides':[q[1] for q in _roof_edge_candidates],'baseRoofZ':base_roof,'semantic':'visible roof edge; wall/roof junction remains unresolved'}
  else:
   # The roof-grid base statistic can land above the actual sloped surface and
   # flatten it when the shell enforces its floor. With no photo boundary
   # evidence, use a conservative lower roof-return quantile as the shell
   # floor; it is not an eave estimate.
   if len(roof):
    _inside_roof=np.asarray([g.covers(Point(float(p[0]),float(p[1]))) for p in roof],dtype=bool)
    _roof_z=np.asarray(roof,dtype=float)[_inside_roof,2]
    _roof_z=_roof_z[np.isfinite(_roof_z)]
    if len(_roof_z)>=30:base_roof=max(z0+3.,min(float(base_roof),float(np.quantile(_roof_z,.05))))
   roof_observation_audit['eave']={'status':'no-matched-confident-eave','baseRoofZ':base_roof,
                                  'fallback':'lower supported roof-return quantile; not an eave estimate'}
  roof_g=g.difference(envelopes.roof_mask(resolved,base_roof-z0)) if resolved else g
  envelope_spans=envelopes.coverage_spans(resolved,z0,base_roof-z0)
  roof_lids=list(envelopes.lids(resolved,base_roof-z0) if resolved else [])
  for _slot,_z1 in roof_lids:
   _lid=_slot.intersection(g)
   if not _lid.is_empty:extrude(_lid,z0+_z1-.10,.10,'roofs__'+id,'roof')
  _usable_host_candidates=[q for q in _host_candidates if q.get('side') not in _frontage_roofline_sides]
  _host_spec=max(_usable_host_candidates,key=lambda q:(q['confidence']=='high',q['ridgeZ'] is not None)) if _usable_host_candidates and len({q['kind'] for q in _usable_host_candidates})==1 else None
  if _host_spec:_host_spec={**_host_spec,'excludeStrips':_host_exclusions}
  _shell=(roof_shell.build(roof_g,roof,base_roof,cell_m=min(.5,roof_step),eave_constraints=_eave_constraints,host_spec=_host_spec,boundary_points=_boundary_roof[:,:3]) if not roof_g.is_empty else
          {'surface':trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),dtype=np.int64),process=False),
           'skirt':None,'audit':{'status':'empty-roof-domain','inputPointCount':0}})
  _source_parts,_source_part_audit=roof_source_parts.resolve(ROOF_OBSERVATIONS,PROFILES,id,c.get('sourceId',id),roof_g)
  _source_part_neighbors={int(_j) for _part in _source_parts for _j in building_index.query(_part['polygon'])
                          if building_sources[int(_j)]!=c.get('sourceId',id)}
  _front_priors=[];_front_prior_validation=[]
  for _front_obs,_front_profile in _source_front_inputs:
   # Contract: observation metres have already been mapped to this model datum.
   _front_prior,_front_validation=roof_front_ownership.prepare_front_prior(
       _front_obs,_front_profile,z0,base_roof)
   _front_prior_validation.append({'side':_front_obs.get('side'),**_front_validation})
   if _front_prior is not None:_front_priors.append(_front_prior)
  _gable_priors=[];_gable_validation=[]
  for _gable_profile in PROFILES:
   _gable_match=_gable_profile.get('match') or {}
   _gable_part=_gable_match.get('partId')
   if not (str(_gable_part)==str(id) or (_gable_part is None and
           str(_gable_match.get('featureId'))==str(c.get('sourceId',id)))):continue
   if not _gable_profile.get('gableOutline'):continue
   _gable_prior,_gable_audit=roof_gable_ownership.prepare_gable_prior(_gable_profile,z0)
   _gable_validation.append(_gable_audit)
   if _gable_prior is not None:
    if float(_gable_prior['traceUZ'][:,1].min())<base_roof-1e-8:
     _gable_validation.append({'profileId':_gable_profile['id'],'status':'unresolved',
       'reason':'source-outline-below-shell-floor-requires-separate-base-resolution'})
    else:_gable_priors.append(_gable_prior)
  if _gable_priors:
   _gable_state=_shell.get('nativeSupportState')
   if _gable_state is None:_gable_priors=[];_gable_validation.append({'status':'unresolved-no-native-support-state'})
   elif _front_priors:_gable_priors=[];_gable_validation.append({'status':'unresolved-competing-front-ownership-priors'})
   else:
    _gable_fixed,_gable_fixed_audit=roof_assemblies.resolve(
        ROOF_OBSERVATIONS,PROFILES,id,c.get('sourceId',id),roof_g,z0,_shell['surface'])
    if _gable_fixed:
     _gable_priors=[];_gable_validation.append({'status':'unresolved-existing-source-component-host-composition-required'})
    else:
     _gable_preview=roof_gable_ownership.solve(
         _gable_state['xy'],_gable_state['faces'],_gable_state['z'],
         _gable_state['known'],_gable_state['unknown'],_gable_priors,roof_g,
         protected_domains=[_part['polygon'] for _part in _source_parts],cell_m=min(.5,roof_step))
     _gable_validation.append(_gable_preview[3])
     if not _gable_preview[3].get('appliedBoundaryVertices',0):_gable_priors=[]
  _gable_mesh_points,_gable_mesh_audit=roof_gable_ownership.outline_mesh_points(_gable_priors,roof_g) if _gable_priors else ([],[])
  _native_baseline_shell=_shell
  _front_fixed_features=None;_front_fixed_audit=None
  _neighbors=[candidates[_j]['g'] for _j in _source_part_neighbors]
  if (_front_priors or _gable_priors) and not _shell['audit'].get('observedHostForm',{}).get('status','').startswith('applied-'):
   import copy as _front_copy
   _baseline_with_parts=roof_source_parts.compose(_front_copy.deepcopy(_native_baseline_shell),_source_parts,_neighbors)
   _front_fixed_features,_front_fixed_audit=roof_assemblies.resolve(
       ROOF_OBSERVATIONS,PROFILES,id,c.get('sourceId',id),roof_g,z0,_baseline_with_parts['surface'])
   _shell=roof_shell.build(roof_g,roof,base_roof,cell_m=min(.5,roof_step),
       eave_constraints=_eave_constraints,host_spec=_host_spec,boundary_points=_boundary_roof[:,:3],
       front_priors=_front_priors,native_baseline_surface=_native_baseline_shell['surface'],
       source_owned_domains=[_part['polygon'] for _part in _source_parts],
       gable_priors=_gable_priors,source_outline_points=_gable_mesh_points)
  _shell=roof_source_parts.compose(_shell,_source_parts,_neighbors)
  _shell['audit']['sourceFrontPriorValidation']=_front_prior_validation
  _shell['audit']['sourceGablePriorValidation']=_gable_validation
  _shell['audit']['sourceGableTopologyRegistration']=_gable_mesh_audit
  _shell['audit']['boundedSourceRoofPartValidation']=_source_part_audit
  _roof_material=('#'+''.join(f'{int(v):02x}' for v in np.median(_roof_colors,axis=0))) if _roof_colors else 'roof'
  roof_patches=[];roof_bands=[];roof_xy=None;venue=None;outer_rounded=[];owned_roof=None
  roof_grid_audit={'roofShell':_shell['audit']}
  plane_audit={'kind':'continuous-roof-shell','shell':_shell['audit']}
  roof_observation_audit['features']={'status':'pending','accepted':[],'rejected':[]}
  if ROOF_OBSERVATIONS:
   _resolved_features,_feature_audit=roof_assemblies.resolve(ROOF_OBSERVATIONS,PROFILES,id,c.get('sourceId',id),roof_g,z0,_shell['surface'])
   if (_front_fixed_features is not None and
       _shell['audit'].get('sourceFrontOwnership',{}).get('appliedBoundaryVertices',0)):
    _resolved_features,_feature_audit=roof_front_ownership.merge_preserved_components(
        _front_fixed_features,_front_fixed_audit,_resolved_features,_feature_audit)
   roof_observation_audit['features']=_feature_audit
   for _rf in _resolved_features:
    if not _rf.get('roofMaterial'):_rf['roofMaterial']=_roof_material
  else:_resolved_features=[]
  _component_cutters,_component_cut_audit=roof_shell.component_cutters(_resolved_features,z0)
  _before_component_roof_area=float(_shell['surface'].area)
  if _component_cutters:
   _shell['surface']=roof_shell.clip_component_intersections(_shell['surface'],_component_cutters)
   _shell['skirt']=roof_shell.clip_component_intersections(_shell.get('skirt'),_component_cutters)
  _shell['audit']['sourceComponentComposition']=_component_cut_audit
  _shell['audit']['sourceComponentRemovedRoofAreaM2']=max(0.,_before_component_roof_area-float(_shell['surface'].area))
  _shell['audit']['surfaceVertexCount']=len(_shell['surface'].vertices)
  _shell['audit']['surfaceFaceCount']=len(_shell['surface'].faces)
  if len(_shell['surface'].faces):put(_shell['surface'],'roofs__'+id,_roof_material);roofparts+=1
  if _shell.get('skirt') is not None and len(_shell['skirt'].faces):patch_sides.append(_shell['skirt'])
  roof_observation_audit['status']=roof_observation_audit['features']['status']
  wall_top_cap=base_roof
  height=max(float(base_roof-z0),float(height))
  c['massHeightM']=float(base_roof-z0)
  roof_envelope_records.append({'buildingId':id,**roof_grid_audit,'continuousPatches':0,'tierCount':0,
                                'roundedCorners':0,'roundedOuterCorners':0,'baseRoofZ':base_roof,
                                'slopedCellsDroppedToBase':0,'wallHeightSource':'continuous roof shell',
                                'planeFit':plane_audit,'observation':roof_observation_audit})
 else:
  MS=mass_spec(c)
  roof_observation_audit={}
  if MS is None and _args.experimental_roof_reconstruction:
   _proposal,_pa=roof_observations.proposal(PROFILES,id,c.get('sourceId',id))
   roof_observation_audit={'reading':_pa}
   if _proposal is not None:
    _validated,_va=roof_observations.validate(g,roof,_proposal,z0)
    roof_observation_audit['validation']=_va
    if _validated:
     _rs={k:v for k,v in _proposal.items() if k!='wallTopM'}
     _rs['ridgeM']=_proposal['wallTopM']+_proposal['riseM']
     MS={'wallTopM':_proposal['wallTopM'],'roof':_rs,'fromReadings':True}
     counts['roofsFromReadings']+=1
  if not _args.experimental_roof_reconstruction and MS is None and str(c['id']) in READ_EAVES:
   _ev=READ_EAVES[str(c['id'])];_E=float(np.median([e for e,_ in _ev]));_K=set().union(*[k for _,k in _ev])
   # the readings' eaves well under the LiDAR's wall top, with a roof's height of LiDAR above them (not a parapet's)
   if _E>=2.5 and (base_roof-z0)-_E>=.6 and 1.2<=float(height)-_E<=8.:
    MS={'wallTopM':round(_E,2),'roof':{'kind':'mansard' if 'mansard' in _K else 'hip','ridgeM':round(float(height),2)},'fromReadings':True}   # a read gable is most often a dormer on a hip (a full-width front gable raised a tall brick triangle)
    counts['roofsFromReadings']+=1
  if MS is not None and MS.get('wallTopM') is not None:
   # the wall top the photographs show (the eaves of a pitched roof, a cornice the LiDAR's roof level misses)
   base_roof=z0+float(MS['wallTopM']);height=max(float(MS['wallTopM']),float(MS.get('roof',{}).get('ridgeM') or 0.));lo=top=z0+height
  # The host cap is removed only under skins whose top return reaches it; lower skins keep the host roof
  # and the host wall band closes the front above them. Spans tell the facade compiler what each box replaces.
  roof_g=g.difference(envelopes.roof_mask(resolved,base_roof-z0)) if resolved else g;envelope_spans=envelopes.coverage_spans(resolved,z0,base_roof-z0)
  # A skin that rises above the host cap replaces the roof under it with a lid at its own top.
  roof_lids=list(envelopes.lids(resolved,base_roof-z0) if resolved else [])
  for slot,z1 in roof_lids:
   lid=slot.intersection(g)
   if not lid.is_empty:extrude(lid,z0+z1-.10,.10,'roofs__'+id,'roof')
  # Facade compiler builds walls with real openings; no opaque block behind glazing.
  # Roof caps are emitted only where a higher envelope does not own the surface.
  # walls whose photographs show a gable (facade_lint's gableOutline): the roof behind turns its gable to them
  _gw=[(p['normal'],p['planeV']) for p in PROFILES if p.get('gableOutline') and p.get('planeV') is not None and str((p.get('match') or {}).get('featureId'))==str(c.get('sourceId',id))]
  _ge=roof_forms.gable_edges(roof_g,_gw) if _gw else []
  if rr is not None:
   LR=CONFIG.get('lidarRole',{})
   if MS is not None:
    # a given massing: its roof form (or a flat roof at its wall top), no LiDAR tiers and no draped surface over it
    roof_patches,plane_audit=roof_forms.build(roof_g,{**MS.get('roof',{}),**({'gableEdges':_ge} if _ge else {})},base_roof,base_roof-z0)
   else:roof_patches,plane_audit=roof_envelope.continuous_patches(roof_g,roof,roof_step,LR.get('slopedPatchMinSlope',.015),LR.get('slopedPatchMinAreaM2',25.),LR.get('slopedPatchMinRiseM',.6),refine=_args.experimental_roof_reconstruction)
   if not roof_patches and not stadium and MS is None:
    # A house's gable, hip or twin gable is two to four planes, each too small for that fit to validate: the simple forms
    # are fitted on the part's rectangle (pipeline/roof_forms.py) and kept only when they clearly beat a flat roof.
    roof_patches,form_audit=roof_forms.fit(roof_g,roof,gable_ends=_ge,pitched_evidence=PITCHED_EV.get(str(c.get('sourceId',id)),set()));plane_audit={**plane_audit,'roofForm':form_audit}
   # Unvalidated height correlation does not establish a hip roof. Keep measured
   # patches/tiers for unresolved shapes instead of inventing a whole-roof form.
   # A roof the LiDAR shows pitched that neither fit kept: most of its returns stand a metre or more over the wall top and
   # rise toward the middle of the outline (a hip or gable seen from above), not a penthouse or chimney on a flat roof.
   # 114 of the area's parts stood as flat boxes with their roof's height lost (Palash's review, 10-16): a hip from the
   # wall top to the returns' 95th percentile.
   if not _args.experimental_roof_reconstruction and not roof_patches and not stadium and MS is None and len(roof)>=30 and float(height)-(base_roof-z0)>=1.5:
    _W=base_roof-z0;_h=roof[:,2]-z0;_above=float((_h>_W+1.).mean())
    try:
     import shapely as _sh
     _de=_sh.distance(_sh.points(roof[:,:2]),g.boundary);_cr=float(np.corrcoef(_de,_h)[0,1]) if np.std(_h)>.1 else 0.
    except Exception:_cr=0.
    if _above>=.4 and _cr>=.3:
     MS={'wallTopM':round(_W,2),'roof':{'kind':'hip','ridgeM':round(float(np.percentile(_h,95)),2)},'fromLidarShape':True}
     roof_patches,plane_audit=roof_forms.build(roof_g,MS['roof'],base_roof,base_roof-z0);counts['roofsFromLidarShape']+=1
   plane_audit['observation']=roof_observation_audit
   if _args.experimental_roof_reconstruction and MS is None:base_roof=max(z0+3.,roof_envelope.supported_eave_base(roof_g,roof_patches,base_roof))
   slope_area=unary_union([p['geometry'] for p in roof_patches]);tier_domain=roof_g.difference(slope_area)
   # A continuous measured slope no plane fit accepted (a seating bowl, a ramp) is one draped surface, not tiers.
   drape_keep,drape_area=roof_envelope.drape_regions(rr,roof_known,frame.geom(roof_g.difference(slope_area)),xmin,ymax,roof_step);drape_area=frame.world(drape_area)
   venue=roof_envelope.venue_form(rr,roof_known,frame.geom(roof_g),xmin,ymax,roof_step,z0) if stadium else None
   if venue:venue={**venue,'ring':frame.world(venue['ring']),'field':frame.world(venue['field'])}
   if venue:
    # A stadium-scale structure: outer ring at its measured height with walls and a cap, a raked seating bowl down
    # to an open field. No tiers and no drape over the noisy bowl returns.
    base_roof=venue['ringTopZ'];tier_domain=make_valid(tier_domain.intersection(venue['ring'])).buffer(0)
    drape_keep=venue['bowlMask'];drape_area=frame.world(roof_envelope.cells_area(drape_keep,frame.geom(roof_g),xmin,ymax,roof_step)) if drape_keep.any() else None
   if MS is not None:drape_area=None
   if drape_area is not None and not drape_area.is_empty:drape_area=make_valid(drape_area).buffer(0);tier_domain=make_valid(tier_domain.difference(drape_area)).buffer(0) if not venue else tier_domain
   # Quantise bin indices, not metres: a 2.5 m tier must remain 2.5 m.
   step=CONFIG['defaults']['roofRegularizationM'];quant=np.maximum(0,np.round((rr-base_roof)/step)).astype(np.int16)
   quant[(rr-base_roof)>max(height+3,5)]=0
   if drape_area is not None and not drape_area.is_empty:quant[drape_keep]=0
   # A tier is a storey or more above the base roof. The first level bin (2.5 m) is parapets, plant and screens
   # standing on the roof, not a volume with its own walls; making it a tier covered roofs in stacked boxes.
   # The storey test reads the measured rise of each tier's own roof, after the bins have found the tiers
   # (spatial_rules.measured_tiers): testing the rounded bin dropped a main roof 3.35 m above a lower strip as plant.
   if venue:quant[:]=0
   roof_bands=spatial_rules.roof_bands(quant,from_origin(xmin,ymax,roof_step,roof_step),frame.geom(tier_domain),max(LR.get('tierMinAreaM2',25),g.area*LR.get('tierMinShare',.012)),simplify_m=roof_step*.75,cell_m=roof_step,level_step_m=step,measured_mask=roof_known)
   roof_bands=spatial_rules.measured_tiers(roof_bands,rr,roof_known,from_origin(xmin,ymax,roof_step,roof_step),base_roof)
   # LiDAR sets the envelope, not the detail: at most a few substantial tiers are kept, those with the largest roofs.
   roof_bands=spatial_rules.keep_tiers(roof_bands,LR.get('tierMaxCount',len(roof_bands)),rr,roof_known,from_origin(xmin,ymax,roof_step,roof_step),base_roof)
   if MS is not None:roof_bands=[]
   # Corners the returns show rounded are rounded, in every tier standing flush at them (spatial_rules.round_tier_corners).
   outer_rounded=[]
   roof_bands,rounded_corners=spatial_rules.round_tier_corners(roof_bands,np.c_[frame.points(roof[:,:2]),roof[:,2]],base_roof,outer=gF.boundary,records=outer_rounded)
   # A rounded corner standing on the outline: the outline's wall stops under the rounded tiers over the arc and the tier
   # draws the arc (facades.compile, roundedOuterCorners / roundedZones).
   outer_rounded=[{'corner':frame.world_points(np.asarray(r['corner'])[None])[0],'along':[frame.world_vectors(np.asarray(r[a])[None])[0] for a in r['outerAxes']],'radiusM':r['radiusM'],'marginM':r['marginM'],'bottomZ':r['bottomZ'],'cut':frame.world(r['cut'])} for r in outer_rounded]
   _valid=lambda q:q if q.is_empty or q.is_valid else make_valid(q).buffer(0)
   roof_bands=[{**b,'geometry':_valid(frame.world(b['geometry'])),'cap':_valid(frame.world(b['cap']))} for b in roof_bands]
   # Review aid (no effect on output): TIER_DEBUG=<id,id> dumps those buildings' roof raster and tiers for inspection.
   import os as _os
   if id in _os.environ.get('TIER_DEBUG','').split(','):
    import pickle as _pk;_pk.dump({'rr':rr,'known':roof_known,'quant':quant,'base':base_roof,'z0':z0,'frame':(frame.angle,frame.ox,frame.oy),'xmin':xmin,'ymax':ymax,'step':roof_step,'g':g,'tier_domain':tier_domain,'drape_area':drape_area,'bands':roof_bands},open(OUT/f'tier-debug-{id}.pkl','wb'))
   if roof_bands:roof_bands[-1]['cap']=roof_bands[-1]['geometry']
   base_cap=tier_domain.difference(roof_bands[0]['geometry']) if roof_bands else tier_domain
   extrude(base_cap,base_roof,.10,'roofs__'+id,'roof')
   c['massHeightM']=float(base_roof-z0)   # the part's main roof level: what a hand reading's frame is compared with
   for bi,band in enumerate(roof_bands):
    # A tier stops behind the strip an observed frontage owns; its wall would stand in the frontage plane, blank
    # and coplanar with the observed facade.
    # ... except at a rounded corner standing on the outline: the arc is the tier's (the frontage's strip covered it and
    # nobody drew it)
    rounded_zone=unary_union([r['cut'].buffer(.3) for r in outer_rounded]) if outer_rounded else None
    own_strip=profile_strip.difference(rounded_zone) if rounded_zone is not None else profile_strip
    part=band['geometry'].difference(own_strip)
    tier=dict(c);tier['g']=part;tier['elevationOffset']=base_roof+band['bottom']-z0;tier['roofBand']=True;tier['outerBoundary']=g.boundary
    # The tiers round it decide where its walls stand (facades.compile): a wall with a higher tier flush over it is that
    # tier's, and a wall comes down to whichever roof stands in front of it.
    tier['tierBase']=base_roof;tier['tierAbove']=roof_bands[bi+1]['geometry'] if bi+1<len(roof_bands) else None
    tier['tiersBelow']=[(b['geometry'],base_roof+b['top']) for b in roof_bands[:bi]]
    tier_ctx=dict(globals());tier_ctx['all_buildings']=all_buildings.difference(g);tier_ctx['envelopeMask']=envelope_mask.union(own_strip);tier_ctx['envelopeSpans']=envelope_spans
    tier_ctx['roundedZones']=rounded_zone
    rr_records,rr_components=facades.compile(tier_ctx,tier,base_roof+band['bottom'],float(band['top']-band['bottom']),height,tags,OBS['facades']);facade_records.extend(rr_records);counts['windowAssemblies']+=rr_components
    extrude(band['cap'],base_roof+band['top'],.10,'roofs__'+id,'roof');roofparts+=1
   if drape_area is not None and not drape_area.is_empty:
    # The draped surface follows the smoothed measured surface: seat rows and stairs would crumple it.
    if venue:
     drape_surface,drape_skirt=roof_envelope.drape_mesh(venue['bowlZ'],drape_keep,xmin,ymax,roof_step,z0);drape_surface,drape_skirt=frame.world_mesh(drape_surface),frame.world_mesh(drape_skirt)
     put(drape_surface,'roofs__'+id,'concrete');roofparts+=1   # seating rows read as stepped concrete; the bowl is the roof surface there
    else:
     drape_surface,drape_skirt=roof_envelope.drape_mesh(np.where(np.isfinite(rr),roof_envelope.drape_regions.smooth,np.nan),drape_keep,xmin,ymax,roof_step,base_roof);drape_surface,drape_skirt=frame.world_mesh(drape_surface),frame.world_mesh(drape_skirt)
     put(drape_surface,'roofs__'+id,'roof');roofparts+=1
    if drape_skirt is not None:put(drape_skirt,'buildings__'+id,color)
   for patch in roof_patches:
    xy=np.concatenate([np.asarray(p.exterior.coords) for p in polys(patch['geometry'])]);bottom=min(base_roof,float(roof_evaluate(patch['model'],xy).min())-.2)
    # a front cut to its measured parapet stands above this roof: the roof stops 0.3 m inside the wall's face (inside the
    # wall's 0.38 m), else its surface meets the face along the old rake and draws a line across the brick
    pg=patch['geometry']
    for _pr in facades.parapets_of(id):pg=pg.difference(LineString([tuple(_pr['a']),tuple(_pr['b'])]).buffer(.3,cap_style=2))
    solid=roof_solid(pg if not pg.is_empty else patch['geometry'],patch['model'],bottom)
    up=solid.face_normals[:,2]>.05;side=np.abs(solid.face_normals[:,2])<=.05
    put(solid.submesh([np.where(up)[0]],append=True,repair=False),'roofs__'+id,'roof')
    patch_sides.append(solid.submesh([np.where(side)[0]],append=True,repair=False))   # coloured once the walls are known
    roofparts+=1
   owned_roof=roof_envelope.owned_samples(g,roof_xy,base_roof,roof_bands,roof_patches,[(slot,z0+z1) for slot,z1 in roof_lids])
   height=max(height,float(owned_roof[:,2].max()-z0))
   roof_envelope_records.append({'buildingId':id,**roof_grid_audit,'continuousPatches':len(roof_patches),'tierCount':len(roof_bands),'roundedCorners':len(rounded_corners),'roundedOuterCorners':len(outer_rounded),'baseRoofZ':base_roof,'slopedCellsDroppedToBase':0,'wallHeightSource':'same generated roof envelope','planeFit':plane_audit})
  if rr is None and MS is not None:
   # a part too small for the LiDAR roof grid (a turret holds a few dozen returns) still takes its given roof form
   roof_patches,plane_audit=roof_forms.build(roof_g,{**MS.get('roof',{}),**({'gableEdges':_ge} if _ge else {})},base_roof,base_roof-z0)
   for patch in roof_patches:
    xy=np.concatenate([np.asarray(p.exterior.coords) for p in polys(patch['geometry'])]);bottom=min(base_roof,float(roof_evaluate(patch['model'],xy).min())-.2)
    solid=roof_solid(patch['geometry'],patch['model'],bottom);up=solid.face_normals[:,2]>.05;side=np.abs(solid.face_normals[:,2])<=.05
    put(solid.submesh([np.where(up)[0]],append=True,repair=False),'roofs__'+id,'roof');patch_sides.append(solid.submesh([np.where(side)[0]],append=True,repair=False));roofparts+=1
  if rr is None and not roof_patches:extrude(roof_g,base_roof,.10,'roofs__'+id,'roof')
  if rr is not None:
   # The top of the envelope actually built (base roof, kept tiers, tilted planes, lids), for the wall-height cap.
   _tops=[base_roof]+[base_roof+b['top'] for b in roof_bands]+[z0+z1 for _,z1 in roof_lids]
   for _p in roof_patches:
    # along the whole outline, not only at its corners: a gable's ridge meets its end walls between two corners
    _xy=np.concatenate([np.asarray([q.exterior.interpolate(t).coords[0] for t in np.linspace(0,q.exterior.length,max(8,int(q.exterior.length/.5)))]) for q in polys(_p['geometry'])]);_tops.append(float(roof_evaluate(_p['model'],_xy).max()))
   wall_top_cap=max(_tops)+1.5
  else:wall_top_cap=None
 c['massHeightM']=float(base_roof-z0)
 outer_ctx=dict(globals());outer_ctx['venueField']=venue['field'] if (rr is not None and venue) else None;outer_ctx['wallTopCapZ']=wall_top_cap;outer_ctx['roofSamplesOwned']=bool(_args.experimental_roof_reconstruction and roof_patches and rr is not None);outer_ctx['roofSamples']=owned_roof if _args.experimental_roof_reconstruction and roof_patches and rr is not None else roof;outer_ctx['envelopeMask']=envelope_mask;outer_ctx['envelopeSpans']=envelope_spans;outer_ctx['roundedOuterCorners']=outer_rounded
 pitched=[p['geometry'] for p in roof_patches if max(float(np.linalg.norm(q[:2])) for q in p['model']['planes'])>=roof_envelope.PITCH_SLOPE]
 outer_ctx['pitchedRoof']=unary_union(pitched) if pitched else None
 outer_ctx['pitchedModels']=[(p['geometry'],p['model']) for p in roof_patches if max(float(np.linalg.norm(q[:2])) for q in p['model']['planes'])>=roof_envelope.PITCH_SLOPE]
 if _args.roof_shell:
  outer_ctx['roofShell']=True;outer_ctx['roofSamples']=None;outer_ctx['roofSamplesOwned']=False
  outer_ctx['pitchedRoof']=roof_g;outer_ctx['pitchedModels']=[];outer_ctx['wallTopCapZ']=base_roof
  outer_ctx['roofSkirtOwners']=[]
  outer_ctx['wallTopBoundaryChainsByProfile']={_p['id']:_shell.get('boundaryChains',[]) for _p in PROFILES if ((_p.get('match') or {}).get('partId')==id or ((_p.get('match') or {}).get('partId') is None and (_p.get('match') or {}).get('featureId')==c.get('sourceId',id)))}
 outer_ctx['pockets']=[]
 _roof_component_cutters=_component_cutters if _args.roof_shell else []
 records,component_count=facades.compile(outer_ctx,c,z0,base_roof-z0,base_roof-z0 if _args.roof_shell else height,tags,OBS['facades']);facade_records.extend(records);counts['windowAssemblies']+=component_count
 _roof_component_cutters=[]
 _foundation(outer_ctx['pockets'])
 _feats=list((MASSING.get(c.get('sourceId',id),{}) or {}).get('roofFeatures') or []) if MASSING else []
 if _args.roof_shell:
  for _candidate in _resolved_features:
   _xy_candidate=np.asarray(_candidate.get('xy',[np.nan,np.nan]),float)
   _duplicate=any(_existing.get('kind')==_candidate.get('kind') and len(_existing.get('xy',[]))==2 and
                  np.linalg.norm(np.asarray(_existing['xy'],float)-_xy_candidate)<1. for _existing in _feats)
   if not _duplicate:_feats.append(_candidate)
 if _feats:
  # dormers and chimneys the photographs show and the LiDAR's bumps confirm (pipeline/roof_features.py); a dormer's walls
  # take the building's main wall material (the longest walls'), its window a white frame
  _len={}
  for _r in records:
   if _r.get('material') and _r.get('lengthM'):_len[_r['material']]=_len.get(_r['material'],0.)+float(_r['lengthM'])
  _wall=max(_len,key=_len.get) if _len else color
  for _m,_role in roof_features.build(_feats,g,z0,base_roof,roof_patches):
   _mat={'roof':'roof','wall':_wall,'trim':'#efece4','brick':'brick_modular#7b4a3b','glass':'glass'}.get(_role,_role)
   put(_m,'roofs__'+id if _role=='roof' else 'buildings__'+id,_mat);roofparts+=1
 MASS_DUMP.append({'id':id,'sourceId':c.get('sourceId',id),'outline':[list(map(list,np.round(np.asarray(q.exterior.coords),3))) for q in polys(g)],'z0':round(float(z0),3),'baseRoofZ':round(float(base_roof),3),
  'heightM':round(float(height),2),'given':MS is not None,'patches':[{'form':p.get('form'),'planes':p['model']['planes']} for p in roof_patches],'tiers':[[round(float(b['bottom']),2),round(float(b['top']),2)] for b in roof_bands]})
 for side_mesh in patch_sides:
  if _args.roof_shell:side_mesh=facade_wall_domain.clip_owned_skirt(side_mesh,outer_ctx.get('roofSkirtOwners',[]))
  if side_mesh is None or not len(side_mesh.faces):continue
  # The vertical faces of a pitched roof (a gable's triangle over its end wall) are that wall's material; a face on no
  # wall (between two roof planes) keeps the building's.
  for key,faces in roof_envelope.side_materials(side_mesh,records,color).items():
   if key is not None:put(side_mesh.submesh([np.asarray(faces)],append=True,repair=False),'buildings__'+id,key)
 for panel in panels:
  # A LiDAR-measured sign panel: a thin slab on the roof with two posts; appearance unresolved (no imagery).
  pa=panel['axis'];pangle=math.atan2(pa[1],pa[0]);pc=panel['centre'];hgt=panel['topZ']-panel['bottomZ']
  cube([pc[0],pc[1],(panel['topZ']+panel['bottomZ'])/2],[panel['lengthM'],.3,hgt],'facades__'+id,'panel_dark',pangle)
  if panel['bottomZ']-panel['roofZ']>.5:
   for sgn in (-.35,.35):
    post=pc+pa*panel['lengthM']*sgn;cube([post[0],post[1],(panel['bottomZ']+panel['roofZ'])/2],[.2,.2,panel['bottomZ']-panel['roofZ']],'facades__'+id,'steel',pangle)
  counts['rooftopPanels']+=1
 er,ec=envelopes.compile(globals(),c,z0,resolved);envelope_records.extend(er);facade_records.extend(er);records.extend(er);counts['windowAssemblies']+=ec
 lettering_records.extend(lettering.compile(globals(),c,z0,LETTERING_DATA['signs'],resolved))
 for r in resolved:envelope_frontages.append({'frame':r['frame'],'label':r['profile'].get('displayName',name)+' / '+r['profile']['id']})
 observed_count=sum(r['status'] in ['visual-fit','image-layout-fit','visual-grammar','envelope-graph'] for r in records)
 if er:height=max(height,max(r['heightM'] for r in er))
 features.append({'id':id,'name':name,'height':round(height,2),'heightSource':source,'footprintSource':c['fp'],'outlineMetadata':c['osm'].get('metadata',{}) if c['osm'] else {'captureAgeKnown':False},'roofParts':roofparts,'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'clippedAtBoundary':c['clipped'],'pointCount':len(roof),'facade':'Image-derived envelope parts capture changing heights, materials and depth. Windows belong to individual surfaces; measurements still need independent checking.' if resolved else 'Street-facing wall regions use observation-derived layouts; unseen walls remain inferred.' if observed_count else 'Detailed facade generated from an unverified architectural prior.','facadeEvidence':'Image hierarchy + LiDAR levels' if resolved else 'Image layout / visual grammar' if observed_count else 'Unverified prior','fittedWalls':observed_count,'totalWalls':len(records),'sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id'] if c['osm'] else None})
 print(name,round(height,1),len(roof),roofparts,flush=True)

# Paved carriageways use OSM centerlines and lane counts; widths are inferred.
roads=[];rails=[];footpaths=[];road_specs={};road_polygons=[]
for w in ways:
 t,g=w['tags'],w['g'];h=t.get('highway')
 # trunk roads and the link roads too: a city boulevard mapped as trunk (Van Ness Avenue) had markings and sidewalks but no carriageway
 if h in ['trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service'] and g.geom_type=='LineString':
  road_spec=mobility.spec(t);road_specs[w['id']]=road_spec;lanes=road_spec['lanes'];width=road_spec['leftM']+road_spec['rightM']
  road=mobility.road_polygon(g,road_spec).intersection(boundary).difference(all_buildings)
  road_polygons.append(road);roads.append((w,width,lanes))
 elif h in ['footway','pedestrian','path','cycleway']:
  footpaths.append(w)
 if t.get('railway') in ['light_rail','tram','rail'] and g.geom_type=='LineString':rails.append(w)

road_area=unary_union(road_polygons);counts['roadSegments']=len(roads)
street_plan=street_space.prepare(globals());road_area=street_plan['road']
(OUT/'street-space-geometry.json').write_text(json.dumps({k:street_plan[k].__geo_interface__ for k in ['road','cycle','sidewalk','crossings','curb','forbidden']}))
surface(road_area,'roads','asphalt',.09)
tree_placements=street_space.tree_candidates(globals()) if ONLY is None else []
for w in footpaths:
 g=w['g'];t=w['tags']
 if t.get('footway')=='crossing':
  for line in lines(g.intersection(boundary)):
   if line.length>45:continue
   for s in np.arange(.3,line.length,.95):
    pt=line.interpolate(s);p2=line.interpolate(min(s+.1,line.length));angle=math.atan2(p2.y-pt.y,p2.x-pt.x)
    # Paint follows the pavement instead of making raised rigid slabs on slopes.
    u=np.array([math.cos(angle),math.sin(angle)]);n=np.array([-u[1],u[0]]);p=np.array([pt.x,pt.y])
    stripe=Polygon([p+u*a+n*b for a,b in [(-.225,-1.3),(.225,-1.3),(.225,1.3),(-.225,1.3)]])
    surface(stripe.intersection(street_plan['road'].union(street_plan['cycle'])),'markings','paint',.132)
  counts['crosswalks']+=1
 elif g.geom_type in ['Polygon','MultiPolygon']:surface(g.intersection(boundary).difference(all_buildings),'paths','sidewalk',.12)

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
from pipeline.crown_clearance import CrownClearance
_crown_clearance=CrownClearance(street_plan,elev)
if ONLY is None:vegetation.compile(globals())
# Trees the LiDAR measures in the mapped open ground (parks, gardens, lawns), which no street-tree inventory lists.
from pipeline import canopy
# Small trees the facade readings saw in front of the walls (front gardens), drawn as the LiDAR's trees are.
for _t in front_trees:canopy.draw_tree(put,cylinder,*_t)
counts['frontGardenTrees']=len(front_trees)
canopy_records=canopy.compile(globals()) if ONLY is None else []
counts.update(lowTravelLeafCardsRemoved=_crown_clearance.card_count,lowTravelLimbsTrimmed=_crown_clearance.limb_count)
if ONLY is None:street_space.furniture(globals())

# Barriers are generated only from mapped barrier or explicit handrail evidence.
for w in ways:
 t=w['tags'];g=w['g']
 if t.get('leisure')=='pitch' and g.geom_type in ['Polygon','MultiPolygon'] and not any(s.intersects(g) for s in landmark_sites):surface(g.intersection(boundary),'landscape','grass',.2)
# Open ground (parks, gardens, lawns, woods) is turf: the terrain under it was bare surface, so a park read as a grey slab.
# Its paths, pitches, streets and buildings keep their own surfaces.
_open=canopy.open_ground(ways,boundary)
if _open is not None and not _open.is_empty:
 _pitches=unary_union([w['g'] for w in ways if w['tags'].get('leisure')=='pitch' and w['g'].geom_type in ['Polygon','MultiPolygon']])
 _turf=_open.difference(all_buildings.buffer(.05))
 for _k in ('road','sidewalk','cycle','paths'):_turf=_turf.difference(street_plan[_k])
 _turf=_turf.difference(_pitches)
 for _site in landmark_sites:_turf=_turf.difference(_site)
 surface(_turf,'landscape','grass',.12);counts['openGroundTurfM2']=round(_turf.area,1)

# Street compiler uses source topology and ROW; every snap remains an uncertain record.
make_sign=signs.install(globals());sign_records,sidewalk_geometry=streets.compile(globals()) if ONLY is None else ([],None)
if ONLY is None:mobility_records=mobility.compile(globals());mobility_records.extend(ground_paint.compile(globals()))
(OUT/'street-space-audit.json').write_text(json.dumps({'surfaceSource':street_plan['surfaceSource'],'measuredSidewalkAreaM2':street_plan['measuredSidewalkAreaM2'],'sidewalkAreaM2':round(street_plan['sidewalk'].area,2),'placements':street_plan['records']},indent=2))
street_views=[]
# New envelope viewpoints lead the street route. Positions snap to mapped road centers.
for item in envelope_frontages:
 f=item['frame'];n=f['normal'];u=f['tangent'];focus=u*(f['left']+f['width']*.5)+n*f['plane'];desired=Point(*(focus+n*25+u*min(24,f['width']*.22)));ranked=[]
 for w,width,lanes in roads:
  q=w['g'].interpolate(w['g'].project(desired));pos=np.array(q.coords[0]);delta=pos-focus
  if np.dot(delta,n)<5 or np.linalg.norm(delta)>90 or all_buildings.contains(q):continue
  ranked.append((desired.distance(q),pos))
 if ranked:
  _,pos=min(ranked,key=lambda a:a[0]);street_views.append({'position':np.round(pos,2).tolist(),'lookAt':[float(focus[0]),float(focus[1]),7.0],'label':item['label']})
for profile in PROFILES:
 matches=[c for c in candidates if c.get('sourceId',c['id'])==profile['match'].get('featureId') or (c['osm'] and profile['match'].get('osmWayId') is not None and str(c['osm']['id'])==str(profile['match'].get('osmWayId')))]
 if not matches:continue
 c=matches[0];n=np.asarray(profile['normal']);n=n/np.linalg.norm(n);u=np.array([-n[1],n[0]]);coord=np.concatenate([np.array(p.exterior.coords) for p in polys(c['g'])]);d=np.max(coord@n);front=coord[(coord@n)>d-3];focus=front.mean(axis=0) if len(front) else np.array(c['g'].centroid.coords[0]);desired=Point(*(focus+n*20))
 ranked=[]
 for w,width,lanes in roads:
  q=w['g'].interpolate(w['g'].project(desired));pos=np.array(q.coords[0]);v=pos-focus
  if np.dot(v,n)<4 or np.linalg.norm(v)>60 or all_buildings.contains(q):continue
  ranked.append((desired.distance(q),pos,w))
 if ranked:
  _,pos,w=min(ranked,key=lambda v:v[0]);tangent=(u if np.dot(u,focus-pos)>0 else -u);look=focus+tangent*8
  street_views.append({'position':np.round(pos,2).tolist(),'lookAt':[round(float(look[0]),2),round(float(look[1]),2),3.8],'label':w['tags'].get('name','Street')+' / '+profile['id']})
street_views.sort(key=lambda v:sum(float(x)**2 for x in v['position'][:2]))
# One mesh per material and feature; image payloads are shared by glTF.
for key,(v,f,uv,normals) in batches.items():
 if not v:continue
 semantic,material_name=key.split('___',1);mat,_=material_bank.get(material_name)
 mesh=trimesh.Trimesh(np.concatenate(v),np.concatenate(f),vertex_normals=np.concatenate(normals),visual=TextureVisuals(uv=np.concatenate(uv),material=mat),process=False)
 v.clear();f.clear();uv.clear();normals.clear()   # the pieces, once joined: holding both outgrew the machine's memory
 mesh.merge_vertices(merge_tex=False,merge_norm=False)
 duplicate_triangles+=spatial_rules.remove_duplicate_triangles(mesh)
 layer=semantic.split('__')[0];fid=semantic.split('__')[1] if '__' in semantic else None
 status='measured-fit' if layer in ['terrain','roofs'] else 'visual-fit' if '__observed_' in semantic else 'inferred'
 mesh.metadata={'layer':layer,'featureId':fid,'evidence':status,'material':material_name}
 scene.add_geometry(mesh,node_name=key,geom_name=key,metadata=mesh.metadata,transform=VIEW_FRAME)
counts.update(buildings=len(features),lidarPoints=len(pts),roofParts=sum(f['roofParts'] for f in features))
coarse_signs=[]
for s in json.load(open(DATA/'signs.json')):
 try:x,y=geom(s['shape']).coords[0]
 except:continue
 if math.hypot(x,y)<200:coarse_signs.append({'id':s.get('asset_id'),'legend':s.get('legend'),'x':round(x,2),'y':round(y,2),'precision':'municipal approximate location; not rendered as surveyed placement'})
counts['unresolvedSigns']=len(coarse_signs)
counts['facadeWalls']=len(facade_records);counts['imageFittedWalls']=sum(r['status'] in ['visual-fit','image-layout-fit','visual-grammar','envelope-graph'] for r in facade_records);counts['pbrMaterials']=len(material_bank.cache)
counts['numericalFacadeFits']=len(FITS);counts['profileFrontages']=len(PROFILES);counts['inferredRepeatCells']=sum(f['completedCells'] for f in FITS)
counts['envelopeFrontages']=len(envelope_records);counts['envelopeNodes']=sum(r['nodeCount'] for r in envelope_records);counts['boundedLidarLevels']=sum(e['status']=='bounded-lidar-fit' for r in envelope_records for e in r['levels'])
counts['foundationChecks']=len(foundation_records);counts['removedDuplicateTriangles']=duplicate_triangles;counts['spatialCorrections']=len(spatial_audit)
counts['generationSeconds']=round(time.perf_counter()-START,2)
counts['facadeLetteringPlacements']=len(lettering_records);counts['directBoundaryOpenings']=sum(len(f['openings']) for f in BOUNDARY_FITS)
source_ledger={**OBS['sources'],**{a['id']:a for a in ENVELOPE_DATA['sources']},**{k:v for _extra in EXTRA_OBSERVATIONS for k,v in _extra.get('sources',{}).items()}}
freshness_records=freshness.ledger(ENVELOPES,source_ledger,CONFIG['asOfDate'],CONFIG['defaults']['maxFacadeSampleAgeMonths'])
world={'pipelineVersion':CONFIG['version'],'streetViews':street_views,'facadeCoverage':{'walls':len(facade_records),'visuallyFittedWalls':counts['imageFittedWalls'],'numericalFits':len(FITS),'datedImageFrontages':sum(r['hasDatedStreetEvidence'] for r in freshness_records),'freshCaptureNeededFrontages':sum(r['needsFreshCapture'] for r in freshness_records),'independentlyVerifiedWalls':0,'accuracyGate':'NOT PASSED'},'label':CONFIG['region'].get('label'),'studyRadiusM':CONFIG['region']['studyRadiusM'],'contextRadiusM':CONFIG['region']['contextRadiusM'],'center':{'lat':lat,'lon':lon},'coordinateSystem':'Local UTM zone 10N meters; GLB X east, Y up, Z south','verticalOrigin':base,'buildings':features,'counts':dict(counts),'ground':np.round(grid,3).ravel().tolist(),'grid':{'min':MIN,'step':STEP,'n':N},'sources':CONFIG['areaSources']}
audit={'freshness':freshness_records,'spatialCorrections':spatial_audit,'foundations':foundation_records,'mobility':mobility_records,'removedDuplicateTriangles':duplicate_triangles,'rules':surface_evidence.AUDIT,'signAttachments':[r['attachmentCheck'] for r in lettering_records],'unverifiedFacadeSections':sum(r['status']=='inferred' for r in facade_records),'independentAccuracyGate':'NOT PASSED'}
(OUT/(CONFIG['outputPrefix']+'-rule-audit.json')).write_text(json.dumps(audit,indent=2));(OUT/'rooftop-panel-audit.json').write_text(json.dumps(panel_audits,indent=1))
(OUT/'roof-envelope-audit.json').write_text(json.dumps(roof_envelope_records,indent=2))
(OUT/'building-parts-audit.json').write_text(json.dumps(part_audit,indent=2))
(OUT/'massing-dump.json').write_text(json.dumps(MASS_DUMP,default=float))
(OUT/'facade-lint.json').write_text(json.dumps(LINT,indent=1))
(OUT/'landmark-audit.json').write_text(json.dumps(landmark_records,indent=2,default=float))
np.savez_compressed(OUT/'rooftop-structure-returns.npz',**{k+'__above':v for k,v in structure_returns.items()},**{k+'__context':v for k,v in structure_context.items()},**{k+'__rr':v[0] for k,v in structure_raster.items()},**{k+'__frame':np.array([v[1],v[2],v[3]]) for k,v in structure_raster.items()})
glb=scene.export(file_type='glb');(PUBLIC/(CONFIG['outputPrefix']+'.glb')).write_bytes(glb);(OUT/(CONFIG['outputPrefix']+'-pilot.glb')).write_bytes(glb)
(PUBLIC/'world.json').write_text(json.dumps(world,separators=(',',':')));(OUT/(CONFIG['outputPrefix']+'-evidence.json')).write_text(json.dumps({'world':world,'unresolvedSigns':coarse_signs,'freshness':freshness_records,'facadeWalls':facade_records,'signPlacements':sign_records,'envelopeGraphs':envelope_records,'observationSources':{**OBS['sources'],**{a['id']:a for a in ENVELOPE_DATA['sources']}},'footprintConsolidations':consolidations,'spatialCorrections':spatial_audit,'foundations':foundation_records,'mobility':mobility_records,'layoutFits':FITS,'profiles':PROFILES,'materials':{**material_bank.manifest,'appearanceRecipes':'pipeline/material_recipes.py; synthesized family appearance, not building-specific scans'},'boundaryFits':BOUNDARY_FITS,'facadeLettering':lettering_records,'surfaceSources':LETTERING_DATA['sources'],'ruleAudit':audit},indent=2))
print('OUTPUT',len(glb),'bytes',dict(counts),'center',lat,lon,flush=True)

(PUBLIC/'facade-evidence.json').write_text(json.dumps({'walls':facade_records,'signs':sign_records,'envelopes':envelope_records,'lettering':lettering_records,'boundaryFits':BOUNDARY_FITS,'surfaceSources':LETTERING_DATA['sources'],'ruleAudit':audit,'sources':{**OBS['sources'],**{a['id']:a for a in ENVELOPE_DATA['sources']}}},separators=(',',':')))
print('GENERATION_SECONDS',round(time.perf_counter()-START,2),flush=True)

import gzip,hashlib
with gzip.GzipFile(PUBLIC/(CONFIG['outputPrefix']+'.glb.gz'),'wb',compresslevel=7,mtime=0) as stream:stream.write(glb)
(PUBLIC/(CONFIG['outputPrefix']+'.glb')).unlink()
# Bounded transfer chunks; raw GLB is retained only as a local interchange artifact.
packed=(PUBLIC/(CONFIG['outputPrefix']+'.glb.gz')).read_bytes();chunks=[];digest=hashlib.sha256(glb).hexdigest()[:12]
for i in range(0,len(packed),4*1024*1024):
 name=f"{CONFIG['outputPrefix']}-{digest}-{i//(4*1024*1024):02d}.bin";(PUBLIC/name).write_bytes(packed[i:i+4*1024*1024]);chunks.append('/'+name)
for old in PUBLIC.glob(CONFIG['outputPrefix']+'-*.bin'):
 if '/'+old.name not in chunks:old.unlink()
(PUBLIC/'model.json').write_text(json.dumps({'compression':'gzip','contentHash':digest,'chunks':chunks,'compressedBytes':len(packed),'uncompressedBytes':len(glb)}))
(PUBLIC/(CONFIG['outputPrefix']+'.glb.gz')).unlink()
