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
from pipeline import facades,streets,vegetation,signs,envelopes,lettering,surface_evidence,spatial_rules,mobility,freshness,street_space,ground_paint,roof_envelope,landmarks
from general_world.roof_planes import roof_solid,evaluate as roof_evaluate
from trimesh.visual.texture import TextureVisuals

import argparse
_args=argparse.ArgumentParser(description='Detailed pilot generator; writes a candidate (raw GLB, chunks, world.json, evidence) into --output instead of the live preview.');_args.add_argument('--output',required=True);_args.add_argument('--streets-only',action='store_true',help='Emit street layers only for validated layer replacement; never publish this partial scene directly.');_args=_args.parse_args()
ROOT=pathlib.Path(__file__).parent;DATA=ROOT/'data';OUTDIR=pathlib.Path(_args.output);PUBLIC=OUTDIR/'public';PUBLIC.mkdir(parents=True,exist_ok=True)
OUT=OUTDIR;OUT.mkdir(parents=True,exist_ok=True)
PROFILES=json.load(open(ROOT/'pipeline/facade-profiles.json'))['profiles'];FITS=json.load(open(ROOT/'pipeline/facade-layouts.json'))['fits']
# Observation packets produced by the frontage-reading step (one file per run) join the hand-annotated profiles.
EXTRA_OBSERVATIONS=[json.load(open(f)) for f in sorted((ROOT/'pipeline').glob('frontage-observations*.json'))]
for _extra in EXTRA_OBSERVATIONS:PROFILES+=_extra.get('profiles',[])
# Colour zones, trims and sign boards measured on the photograph of each hand-read frontage (auto_read_frontages_v31.py
# --supplements): the hand reading keeps its layout and material; a kind it has none of is added from the photograph.
_SUP=ROOT/'pipeline/frontage-supplements-v31.json'
if _SUP.exists():
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
BOUNDARY_FITS=json.load(open(ROOT/'pipeline/boundary-layouts.json'))['fits']
for fit in BOUNDARY_FITS:
 profile=next(p for p in PROFILES if p['id']==fit['profileId']);profile.setdefault('openings',[]).extend(fit['openings'])
LETTERING_DATA=json.load(open(ROOT/'pipeline/lettering-observations.json'));lettering_records=[]
ENVELOPE_DATA=json.load(open(ROOT/'pipeline/envelope-observations.json'));ENVELOPES=ENVELOPE_DATA['profiles'];envelope_records=[];envelope_frontages=[]
# Photographed colours of envelope-designed frontages (read_envelope_colours_v32.py), by envelope profile id.
ENVELOPE_COLOURS=json.load(open(ROOT/'pipeline/envelope-colours-v32.json'))['envelopes'] if (ROOT/'pipeline/envelope-colours-v32.json').exists() else {}
CONFIG=json.load(open(ROOT/'pipeline/config.json'));OBS=json.load(open(ROOT/'pipeline/observations.json'));facade_records=[];sign_records=[]
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
ground=pts[pts[:,3]==2];ground=ground[(ground[:,2]>-5)&(ground[:,2]<20)]
base=float(np.median(ground[np.linalg.norm(ground[:,:2],axis=1)<40,2]));pts[:,2]-=base;ground[:,2]-=base
tree=cKDTree(ground[:,:2]);N=179;STEP=3.;MIN=-267.
gx,gy=np.meshgrid(np.arange(N)*STEP+MIN,np.arange(N)*STEP+MIN)
dist,idx=tree.query(np.c_[gx.ravel(),gy.ravel()],k=8)
grid=np.median(ground[idx,2],axis=1).reshape(N,N);grid=gaussian_filter(grid,.8)
def elev(x,y):
 fx=np.clip((np.asarray(x)-MIN)/STEP,0,N-1.00001);fy=np.clip((np.asarray(y)-MIN)/STEP,0,N-1.00001);ix=np.floor(fx).astype(int);iy=np.floor(fy).astype(int);fx=fx-ix;fy=fy-iy
 a,b,c,d=grid[iy,ix],grid[iy,ix+1],grid[iy+1,ix],grid[iy+1,ix+1]
 return np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,d+(c-d)*(1-fx)+(b-d)*(1-fy))

scene=trimesh.Scene();features=[];batches=collections.defaultdict(lambda:[[],[],[]]);counts=collections.Counter();materials={}
palette={'asphalt':'#41484b','sidewalk':'#b4b5ac','brick':'#97604b','brick2':'#b4795d','stone':'#c5beb0','dark':'#283c42','glass':'#4b7686','frame':'#c4d0d1','roof':'#737b79','grass':'#566e4b','trunk':'#75644d','leaf':'#425e48','metal':'#546661','paint':'#e4e5d8','yellow':'#d7b351','rail':'#87918e','water':'#386c7a'}
def rgba(c):
 c=palette.get(c,c).lstrip('#');return [int(c[i:i+2],16) for i in (0,2,4)]+[255]
material_bank=Materials(ROOT/'material-assets',palette)
batches=collections.defaultdict(lambda:[[],[],[],[]])
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
def put(mesh,key,color,uv=None):
 if len(mesh.vertices)==0:return
 if CLIP is not None and uv is None:
  mesh=_clip(mesh)
  if mesh is None or len(mesh.vertices)==0:return
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
foundation_records=[];mobility_records=[];duplicate_triangles=0;panel_audits=[];structure_returns={};structure_context={};structure_raster={};roof_envelope_records=[]
for c in candidates:
 c['full_area']=c['g'].area;c['clipped']=not boundary.covers(c['g'])
 # A landmark entry may keep its footprint whole past the model boundary; its builder lays ground under the rest.
 _lm=landmarks.entry(c)
 if _lm and _lm.get('extendBeyondBoundary'):c['extendedBeyondBoundary']=c['clipped'];c['clipped']=False
 else:c['g']=c['g'].intersection(boundary)
all_buildings=unary_union([c['g'] for c in candidates]);ground_kd=cKDTree(ground[:,:2])
def get_points(g):
 a,b,c,d=g.bounds;m=(pts[:,0]>=a)&(pts[:,0]<=c)&(pts[:,1]>=b)&(pts[:,1]<=d)&(pts[:,3]!=7)&(pts[:,3]!=18)
 a=pts[m];return a[contains_xy(g,a[:,0],a[:,1])]
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
 if landmarks.entry(c):_split.append(c);continue       # a landmark's builder owns the whole outline
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
street_corridor=unary_union([w['g'].buffer(5.) for w in ways if w['g'].geom_type=='LineString' and w['tags'].get('highway') in ('primary','secondary','tertiary','unclassified','residential','living_street','pedestrian')]
                            +[w['g'].buffer(4.) for w in ways if w['g'].geom_type=='LineString' and w['tags'].get('highway')=='service' and w['tags'].get('service') not in ('parking_aisle','driveway')]
                            +[w['g'] for w in ways if w['g'].geom_type in ('Polygon','MultiPolygon') and w['tags'].get('highway')=='pedestrian'])
for c in ([] if _args.streets_only else candidates):
 id,g=c['id'],c['g'];og=c['old'];tags=c['osm']['tags'] if c['osm'] else {};name=tags.get('name') or (tags.get('addr:housenumber','').replace(';',' / ')+' '+tags.get('addr:street','')).strip() or 'Building '+og.get('sf16_bldgid',id)
 stadium=c['full_area']>18000
 sample=get_points(g.buffer(-.65));z0=float(elev(g.centroid.x,g.centroid.y));roof=sample[sample[:,2]>z0+3]
 roof_step=2.0 if stadium else 1.5
 wxmin,wymin,wxmax,wymax=g.bounds;wnx=max(2,math.ceil((wxmax-wxmin)/roof_step));wny=max(2,math.ceil((wymax-wymin)/roof_step))
 # Roof heights come from roof SURFACES: each raster cell reads its highest dense thin layer of returns,
 # and returns above that layer (signs, antennas, railings, plant) are dropped before any height statistic.
 rr=None
 panels=[]
 if len(roof)>50:
  rr,surface_returns=spatial_rules.surface_raster(roof,wxmin,wymax,roof_step,wnx,wny)
  # Thin tall linear structures above the roof surface (signs, screens) are panels, not roof volume.
  around=pts[(pts[:,0]>=wxmin-10)&(pts[:,0]<=wxmax+10)&(pts[:,1]>=wymin-10)&(pts[:,1]<=wymax+10)&(pts[:,3]!=7)&(pts[:,3]!=18)][:,:3]
  panel_audit=[];panels=spatial_rules.rooftop_panels(roof,surface_returns,rr,wxmin,wymax,roof_step,context=around,audit=panel_audit)
  if panel_audit:panel_audits.append({'buildingId':id,'clusters':panel_audit})
  if (~surface_returns).sum()>=40:structure_returns[id]=roof[~surface_returns];structure_context[id]=around;structure_raster[id]=(rr,wxmin,wymax,roof_step)
  roof=roof[surface_returns]
 # The tier raster (base roof, tiers, drapes) runs along the outline's own axes (spatial_rules.OutlineFrame), over the
 # surface returns kept above; its products are turned back to the world. Which returns count as roof surface (for the
 # plane fits, panels and wall levels) is still decided on the grid those rules were validated on.
 frame=spatial_rules.OutlineFrame(g);gF=frame.geom(g)
 xmin,ymin,xmax,ymax=gF.bounds;nx=max(2,math.ceil((xmax-xmin)/roof_step));ny=max(2,math.ceil((ymax-ymin)/roof_step))
 if rr is not None and len(roof)>50:rr,_=spatial_rules.surface_raster(frame.points(roof),xmin,ymax,roof_step,nx,ny)
 if len(roof)>50:
  lo,top=np.quantile(roof[:,2],[.55,.97]);height=max(3.,float(top-z0));source='USGS LiDAR 2023'
 else:
  try:height=float(tags.get('height',og.get('hgt_median_m','12')).split()[0])
  except:height=12.
  lo=z0+height;top=lo;source='Catalog estimate'
 foundation_records.append(spatial_rules.foundation(globals(),c,z0))
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
  features.append({'id':id,'name':name,'height':round(height,2),'heightSource':'USGS LiDAR 2023','footprintSource':c['fp'],'outlineMetadata':c['osm'].get('metadata',{}) if c['osm'] else {'captureAgeKnown':False},'roofParts':built['roofParts'],'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'clippedAtBoundary':c['clipped'],'pointCount':len(roof),'facade':'Landmark builder: the measured LiDAR surface and playing field, observed street frontages on top, authored props where named.','facadeEvidence':'Landmark: LiDAR surface + return intensity + observed frontages','fittedWalls':len(er),'totalWalls':len(er),'landmark':lm.get('label'),'sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id']})
  print(name,round(height,1),'landmark',flush=True);continue
 style=facades.prior(tags,height,c['full_area']);color=style['material']
 roofparts=0
 roof_grid_audit={};roof_patches=[];roof_bands=[];roof_xy=None;venue=None;patch_sides=[]
 if rr is not None and len(roof)>50:
  rr,roof_known,roof_xy,roof_grid_audit=roof_envelope.prepare_grid(rr,gF,xmin,ymax,roof_step)
  base_roof=roof_envelope.base_level(rr,roof_known,roof_step,g.area) if rr is not None else z0+height
  if stadium:base_roof=max(z0+6,min(base_roof,z0+12))
  else:base_roof=max(z0+4,min(base_roof,lo))
 else:base_roof=z0+height;rr=None
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
 if rr is not None:
  LR=CONFIG.get('lidarRole',{})
  roof_patches,plane_audit=roof_envelope.continuous_patches(roof_g,roof,roof_step,LR.get('slopedPatchMinSlope',.015),LR.get('slopedPatchMinAreaM2',25.),LR.get('slopedPatchMinRiseM',.6))
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
  # Corners the returns show rounded are rounded, in every tier standing flush at them (spatial_rules.round_tier_corners).
  roof_bands,rounded_corners=spatial_rules.round_tier_corners(roof_bands,np.c_[frame.points(roof[:,:2]),roof[:,2]],base_roof,outer=gF.boundary)
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
   part=band['geometry'].difference(profile_strip)
   tier=dict(c);tier['g']=part;tier['elevationOffset']=base_roof+band['bottom']-z0;tier['roofBand']=True;tier['outerBoundary']=g.boundary
   # The tiers round it decide where its walls stand (facades.compile): a wall with a higher tier flush over it is that
   # tier's, and a wall comes down to whichever roof stands in front of it.
   tier['tierBase']=base_roof;tier['tierAbove']=roof_bands[bi+1]['geometry'] if bi+1<len(roof_bands) else None
   tier['tiersBelow']=[(b['geometry'],base_roof+b['top']) for b in roof_bands[:bi]]
   tier_ctx=dict(globals());tier_ctx['all_buildings']=all_buildings.difference(g);tier_ctx['envelopeMask']=tier_mask;tier_ctx['envelopeSpans']=envelope_spans
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
  roof_envelope_records.append({'buildingId':id,**roof_grid_audit,'continuousPatches':len(roof_patches),'tierCount':len(roof_bands),'roundedCorners':len(rounded_corners),'baseRoofZ':base_roof,'slopedCellsDroppedToBase':0,'wallHeightSource':'same generated roof envelope','planeFit':plane_audit})
 if rr is None:extrude(roof_g,base_roof,.10,'roofs__'+id,'roof')
 if rr is not None:
  # The top of the envelope actually built (base roof, kept tiers, tilted planes, lids), for the wall-height cap.
  _tops=[base_roof]+[base_roof+b['top'] for b in roof_bands]+[z0+z1 for _,z1 in roof_lids]
  for _p in roof_patches:
   # along the whole outline, not only at its corners: a gable's ridge meets its end walls between two corners
   _xy=np.concatenate([np.asarray([q.exterior.interpolate(t).coords[0] for t in np.linspace(0,q.exterior.length,max(8,int(q.exterior.length/.5)))]) for q in polys(_p['geometry'])]);_tops.append(float(roof_evaluate(_p['model'],_xy).max()))
  wall_top_cap=max(_tops)+1.5
 else:wall_top_cap=None
 c['massHeightM']=float(base_roof-z0)
 outer_ctx=dict(globals());outer_ctx['venueField']=venue['field'] if (rr is not None and venue) else None;outer_ctx['wallTopCapZ']=wall_top_cap;outer_ctx['roofSamplesOwned']=False;outer_ctx['roofSamples']=roof;outer_ctx['envelopeMask']=envelope_mask;outer_ctx['envelopeSpans']=envelope_spans
 pitched=[p['geometry'] for p in roof_patches if max(float(np.linalg.norm(q[:2])) for q in p['model']['planes'])>=roof_envelope.PITCH_SLOPE]
 outer_ctx['pitchedRoof']=unary_union(pitched) if pitched else None
 outer_ctx['pitchedModels']=[(p['geometry'],p['model']) for p in roof_patches if max(float(np.linalg.norm(q[:2])) for q in p['model']['planes'])>=roof_envelope.PITCH_SLOPE]
 records,component_count=facades.compile(outer_ctx,c,z0,base_roof-z0,height,tags,OBS['facades']);facade_records.extend(records);counts['windowAssemblies']+=component_count
 for side_mesh in patch_sides:
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
 features.append({'id':id,'name':name,'height':round(height,2),'heightSource':source,'footprintSource':c['fp'],'outlineMetadata':c['osm'].get('metadata',{}) if c['osm'] else {'captureAgeKnown':False},'roofParts':roofparts,'center':[round(g.centroid.x,2),round(g.centroid.y,2),round(z0,2)],'area':round(g.area,1),'clippedAtBoundary':c['clipped'],'pointCount':len(roof),'facade':'Image-derived envelope parts capture changing heights, materials and depth. Windows belong to individual surfaces; measurements still need independent checking.' if resolved else 'Street-facing wall regions use observation-derived layouts; unseen walls remain inferred.' if observed_count else 'Detailed facade generated from an unverified architectural prior.','facadeEvidence':'Image hierarchy + LiDAR levels' if resolved else 'Image layout / visual grammar' if observed_count else 'Unverified prior','fittedWalls':observed_count,'totalWalls':len(records),'sourceUrl':'https://data.sfgov.org/d/ynuv-fyni' if c['fp'].startswith('Data') else 'https://www.openstreetmap.org/way/'+c['osm']['id']})
 print(name,round(height,1),len(roof),roofparts,flush=True)

# Paved carriageways use OSM centerlines and lane counts; widths are inferred.
roads=[];rails=[];footpaths=[];road_specs={};road_polygons=[]
for w in ways:
 t,g=w['tags'],w['g'];h=t.get('highway')
 if h in ['primary','secondary','tertiary','unclassified','residential','living_street','service'] and g.geom_type=='LineString':
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
tree_placements=street_space.tree_candidates(globals())
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
vegetation.compile(globals())
street_space.furniture(globals())

# Barriers are generated only from mapped barrier or explicit handrail evidence.
for w in ways:
 t=w['tags'];g=w['g']
 if t.get('leisure')=='pitch' and g.geom_type in ['Polygon','MultiPolygon'] and not any(s.intersects(g) for s in landmark_sites):surface(g.intersection(boundary),'landscape','grass',.2)

# Street compiler uses source topology and ROW; every snap remains an uncertain record.
make_sign=signs.install(globals());sign_records,sidewalk_geometry=streets.compile(globals())
mobility_records=mobility.compile(globals())
mobility_records.extend(ground_paint.compile(globals()))
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
 mesh.merge_vertices(merge_tex=False,merge_norm=False)
 duplicate_triangles+=spatial_rules.remove_duplicate_triangles(mesh)
 layer=semantic.split('__')[0];fid=semantic.split('__')[1] if '__' in semantic else None
 status='measured-fit' if layer in ['terrain','roofs'] else 'visual-fit' if '__observed_' in semantic else 'inferred'
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
counts['facadeWalls']=len(facade_records);counts['imageFittedWalls']=sum(r['status'] in ['visual-fit','image-layout-fit','visual-grammar','envelope-graph'] for r in facade_records);counts['pbrMaterials']=len(material_bank.cache)
counts['numericalFacadeFits']=len(FITS);counts['profileFrontages']=len(PROFILES);counts['inferredRepeatCells']=sum(f['completedCells'] for f in FITS)
counts['envelopeFrontages']=len(envelope_records);counts['envelopeNodes']=sum(r['nodeCount'] for r in envelope_records);counts['boundedLidarLevels']=sum(e['status']=='bounded-lidar-fit' for r in envelope_records for e in r['levels'])
counts['foundationChecks']=len(foundation_records);counts['removedDuplicateTriangles']=duplicate_triangles;counts['spatialCorrections']=len(spatial_audit)
counts['generationSeconds']=round(time.perf_counter()-START,2)
counts['facadeLetteringPlacements']=len(lettering_records);counts['directBoundaryOpenings']=sum(len(f['openings']) for f in BOUNDARY_FITS)
source_ledger={**OBS['sources'],**{a['id']:a for a in ENVELOPE_DATA['sources']},**{k:v for _extra in EXTRA_OBSERVATIONS for k,v in _extra.get('sources',{}).items()}}
freshness_records=freshness.ledger(ENVELOPES,source_ledger,CONFIG['asOfDate'],CONFIG['defaults']['maxFacadeSampleAgeMonths'])
world={'pipelineVersion':CONFIG['version'],'streetViews':street_views,'facadeCoverage':{'walls':len(facade_records),'visuallyFittedWalls':counts['imageFittedWalls'],'numericalFits':len(FITS),'datedImageFrontages':sum(r['hasDatedStreetEvidence'] for r in freshness_records),'freshCaptureNeededFrontages':sum(r['needsFreshCapture'] for r in freshness_records),'independentlyVerifiedWalls':0,'accuracyGate':'NOT PASSED'},'studyRadiusM':200,'contextRadiusM':265,'center':{'lat':lat,'lon':lon},'coordinateSystem':'Local UTM zone 10N meters; GLB X east, Y up, Z south','verticalOrigin':base,'buildings':features,'counts':dict(counts),'ground':np.round(grid,3).ravel().tolist(),'grid':{'min':MIN,'step':STEP,'n':N},'sources':CONFIG['areaSources']}
audit={'freshness':freshness_records,'spatialCorrections':spatial_audit,'foundations':foundation_records,'mobility':mobility_records,'removedDuplicateTriangles':duplicate_triangles,'rules':surface_evidence.AUDIT,'signAttachments':[r['attachmentCheck'] for r in lettering_records],'unverifiedFacadeSections':sum(r['status']=='inferred' for r in facade_records),'independentAccuracyGate':'NOT PASSED'}
(OUT/(CONFIG['outputPrefix']+'-rule-audit.json')).write_text(json.dumps(audit,indent=2));(OUT/'rooftop-panel-audit.json').write_text(json.dumps(panel_audits,indent=1))
(OUT/'roof-envelope-audit.json').write_text(json.dumps(roof_envelope_records,indent=2))
(OUT/'building-parts-audit.json').write_text(json.dumps(part_audit,indent=2))
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
