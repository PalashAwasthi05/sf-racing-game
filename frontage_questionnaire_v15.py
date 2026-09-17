"""One questionnaire entry per frontage (building x facing direction) that lacks an observation.

Each entry fixes the frame a profile is written in: the frontage width W (extent along the wall tangent of the
footprint edges parallel to the wall within the plane tolerance), where each wall piece sits inside [0, W]
(left to right as seen from the street), the wall heights the pipeline measured, the identity keys a profile
must carry, and the crops that show it. The reader fills a profile per entry; nothing here decides content.
"""
import argparse,json,pathlib,re,xml.etree.ElementTree as ET
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon,shape
from shapely.ops import transform
WORK=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--selection',required=True);ap.add_argument('--manifest',default=str(WORK/'facade-priority-v15/crops/manifest.json'))
ap.add_argument('--world',default=str(WORK/'king-street/public/world.json'));ap.add_argument('--plane-tolerance',type=float,default=2.5);a=ap.parse_args()
sel=json.load(open(a.selection));world=json.load(open(a.world));manifest=json.load(open(a.manifest))
crops=manifest if isinstance(manifest,list) else next(v for v in manifest.values() if isinstance(v,list))
ll=Transformer.from_crs(4326,32610,always_xy=True);cx,cy=ll.transform(world['center']['lon'],world['center']['lat'])
def local(x,y,z=None):
 p,q=ll.transform(x,y);return np.asarray(p)-cx,np.asarray(q)-cy
osm=ET.parse(WORK/'data/osm.xml').getroot();nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in osm.findall('node')};footprints={}
for w in osm.findall('way'):
 t={x.attrib['k']:x.attrib['v'] for x in w.findall('tag')};coords=[nodes[r.attrib['ref']] for r in w.findall('nd') if r.attrib['ref'] in nodes]
 if t.get('building') and len(coords)>3:
  x,y=local(*zip(*coords));footprints['osm_'+w.attrib['id']]=Polygon(np.c_[x,y]).buffer(0)
for b in json.load(open(WORK/'data/buildings.json')):footprints['sf_'+b['sf16_bldgid'].replace('.','_')]=transform(local,shape(b['shape']))
buildings={b['id']:b for b in world['buildings']}
def frontage_frame(g,n,tol):
 """W and the left origin along u, as compile_wall computes them: parallel edges near the outermost one."""
 n=n/np.linalg.norm(n);u=np.array([-n[1],n[0]]);edges=[]
 polys=[g] if g.geom_type=='Polygon' else list(g.geoms)
 for p in polys:
  ring=np.array(p.exterior.coords)
  for pa,pb in zip(ring,ring[1:]):
   v=pb-pa;L=np.linalg.norm(v)
   if L>2 and abs(np.dot(v/L,u))>.96:edges.append((L,float(((pa+pb)/2)@n),pa,pb))
 if not edges:return None
 outer=max(d for _,d,_,_ in edges);chosen=[e for e in edges if e[1]>outer-tol]
 xy=np.concatenate([np.array([e[2],e[3]]) for e in chosen]);left=float(np.min(xy@u));right=float(np.max(xy@u))
 return {'u':u.tolist(),'n':n.tolist(),'left':left,'W':right-left,'edgesUsed':len(chosen)}
entries={}
for w in sel['walls']:
 if w['status']!='inferred' or w['coveredBy']:continue
 b=buildings[w['buildingId']];n=np.asarray(w['normal'],float)
 key=None
 for k,e in entries.items():
  if k[0]==w['buildingId'] and np.dot(np.asarray(e['normal']),n/np.linalg.norm(n))>.95:key=k;break
 if key is None:
  g=footprints.get(w['buildingId'])
  frame=frontage_frame(g,n,a.plane_tolerance) if g is not None else None
  key=(w['buildingId'],len([k for k in entries if k[0]==w['buildingId']]))
  entries[key]={'buildingId':w['buildingId'],'profileIdHint':f"{w['buildingId']}-f{key[1]}",'match':{'osmWayId':w['osmWayId'],'featureId':w['buildingId']},'normal':[round(float(v),3) for v in n/np.linalg.norm(n)],
                'buildingHeightM':b['height'],'buildingEvidence':b['facadeEvidence'],'frontageWidthM':round(frame['W'],2) if frame else None,'frame':frame,'walls':[],'crops':[]}
 e=entries[key]
 piece={'edge':w['edge'],'lengthM':w['lengthM'],'heightM':w['heightM'],'midpoint':w['midpoint']}
 if e['frame']:
  u=np.asarray(e['frame']['u']);m=np.asarray(w['midpoint']);c=float(m@u)-e['frame']['left'];W=e['frame']['W']
  piece['xFractionRange']=[round(max(0,(c-w['lengthM']/2)/W),3),round(min(1,(c+w['lengthM']/2)/W),3)]
 e['walls'].append(piece)
 for c in crops:
  if c['building']==w['buildingId'] and c['wallEdge']==w['edge']:
   e['crops'].append({'path':c['cropPath'],'image':c['image'],'captureDate':c['captureDate'],'distanceM':c['cameraDistanceM'],'incidence':c['incidence'],'wallEdge':c['wallEdge'],'viewRank':c['viewRank'],'sizePx':c['sizePx']})
out=list(entries.values())
for e in out:
 e['walls'].sort(key=lambda p:p.get('xFractionRange',[0])[0]);e['crops'].sort(key=lambda c:(c['viewRank'],c['distanceM']));e['wallHeightsM']=sorted({p['heightM'] for p in e['walls']})
path=pathlib.Path(a.selection).with_name('questionnaire-'+pathlib.Path(a.selection).stem.replace('street-','')+'.json');path.write_text(json.dumps(out,indent=1))
print(len(out),'frontages ->',path)
for e in out:print(f"{e['profileIdHint']:28s} W={e['frontageWidthM']} H={e['wallHeightsM']} bh={e['buildingHeightM']} pieces={[p.get('xFractionRange') for p in e['walls']]} crops={len(e['crops'])} way={e['match']['osmWayId']}")
