"""Select the walls that front a named street, and what evidence each has.

General: the street is an argument (data). A wall fronts the street when its midpoint lies within
`--max-distance` of the street's mapped centreline, its outward normal points at that centreline, and the
sight line from the wall to the centreline crosses no other footprint. For each such wall the script reports
the building, its evidence class and whether an observation profile already covers that wall (a profile
matched by the building id or its OSM way whose normal is within 18 degrees).
"""
import argparse,json,pathlib,re,xml.etree.ElementTree as ET
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString,Point,Polygon,shape
from shapely.ops import unary_union,transform
from shapely.strtree import STRtree

WORK=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--street',required=True);ap.add_argument('--max-distance',type=float,default=30.)
ap.add_argument('--world',default=str(WORK/'king-street/public/world.json'));ap.add_argument('--evidence',default=str(WORK/'detailed-v14i/king-street-evidence.json'))
ap.add_argument('--out',default=None);a=ap.parse_args()
world=json.load(open(a.world));evidence=json.load(open(a.evidence))
ll=Transformer.from_crs(4326,32610,always_xy=True);cx,cy=ll.transform(world['center']['lon'],world['center']['lat'])
def local(x,y,z=None):
 p,q=ll.transform(x,y);return np.asarray(p)-cx,np.asarray(q)-cy
osm=ET.parse(WORK/'data/osm.xml').getroot();nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in osm.findall('node')}
street_lines=[];footprints={}
for w in osm.findall('way'):
 t={x.attrib['k']:x.attrib['v'] for x in w.findall('tag')};coords=[nodes[r.attrib['ref']] for r in w.findall('nd') if r.attrib['ref'] in nodes]
 if len(coords)<2:continue
 x,y=local(*zip(*coords));xy=np.c_[x,y]
 if t.get('highway') and t.get('name','').strip().lower()==a.street.strip().lower():street_lines.append(LineString(xy))
 if t.get('building') and len(coords)>3 and t.get('building') not in ('roof','bridge'):footprints['osm_'+w.attrib['id']]=Polygon(xy).buffer(0)
for b in json.load(open(WORK/'data/buildings.json')):footprints['sf_'+b['sf16_bldgid'].replace('.','_')]=transform(local,shape(b['shape']))
if not street_lines:raise SystemExit(f'no mapped highway named {a.street!r} in the study data')
street=unary_union(street_lines)
buildings={b['id']:b for b in world['buildings']}
def way_of(b):
 m=re.search(r'way/(\d+)',b.get('sourceUrl',''));return m.group(1) if m else None
# every observation profile in data: hand-annotated and produced
profiles=[]
for f in [WORK/'pipeline/facade-profiles.json',WORK/'pipeline/envelope-observations.json',*sorted((WORK/'pipeline').glob('frontage-observations*.json'))]:
 if f.exists():profiles+=[dict(p,_file=f.name) for p in json.load(open(f)).get('profiles',[])]
def covering(b,n):
 way=way_of(b)
 return [p['id']+' ('+p['_file']+')' for p in profiles if (p['match'].get('featureId')==b['id'] or (way and p['match'].get('osmWayId') is not None and str(p['match']['osmWayId'])==way)) and np.dot(n,np.asarray(p['normal'],float)/np.linalg.norm(p['normal']))>.95]
all_fp=[g for g in footprints.values() if not g.is_empty];tree=STRtree(all_fp)
def occluded(m,q,own):
 sight=LineString([m,q]);own_fp=footprints.get(own)
 for i in tree.query(sight):
  g=all_fp[i]
  if own_fp is not None and g.equals(own_fp):continue
  if own_fp is not None and g.intersection(own_fp).area>.5*min(g.area,own_fp.area):continue
  if sight.intersects(g):return True
 return False
rows=[]
for wall in evidence['facadeWalls']:
 m=np.asarray(wall['midpoint'],float);n=np.asarray(wall['normal'],float);start=Point(*(m+n*.3))
 d=street.distance(start)
 if d>a.max_distance:continue
 q=street.interpolate(street.project(start));v=np.asarray(q.coords[0])-m;dist=float(np.linalg.norm(v))
 if dist<1e-6 or np.dot(v/dist,n)<.6:continue
 if occluded(m+n*.3,np.asarray(q.coords[0]),wall['buildingId']):continue
 b=buildings.get(wall['buildingId'],{})
 rows.append({'buildingId':wall['buildingId'],'edge':wall.get('edge',wall.get('grammarId')),'lengthM':wall['lengthM'],'heightM':wall['heightM'],'midpoint':wall['midpoint'],'normal':wall['normal'],
              'status':wall['status'],'distanceToStreetM':round(dist,1),'buildingEvidence':b.get('facadeEvidence'),'osmWayId':way_of(b) if b else None,'footprintSource':b.get('footprintSource'),'coveredBy':covering(b,n) if b else []})
rows.sort(key=lambda r:(r['buildingId'],r['edge'] if isinstance(r['edge'],int) else -1))
by_building={}
for r in rows:by_building.setdefault(r['buildingId'],[]).append(r)
summary={'street':a.street,'maxDistanceM':a.max_distance,'streetLengthM':round(street.length,1),'buildings':len(by_building),'walls':len(rows),
         'wallsWithoutObservation':sum(1 for r in rows if r['status']=='inferred' and not r['coveredBy']),
         'buildingsNeedingObservation':sorted({r['buildingId'] for r in rows if r['status']=='inferred' and not r['coveredBy']})}
out=pathlib.Path(a.out) if a.out else WORK/'facade-priority-v15'/('street-'+re.sub(r'[^a-z0-9]+','-',a.street.lower()).strip('-')+'.json')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'summary':summary,'walls':rows},indent=1))
print(json.dumps(summary,indent=1))
for bid,ws in by_building.items():
 b=buildings.get(bid,{})
 print(f"{bid:22s} {str(b.get('facadeEvidence','?')):32s} way={way_of(b) if b else None} "+' | '.join(f"e{w['edge']} {w['lengthM']:.0f}x{w['heightM']:.1f}m {w['status']}{' covered:'+','.join(w['coveredBy']) if w['coveredBy'] else ''}" for w in ws))
