from pathlib import Path
import json,xml.etree.ElementTree as ET,collections
import numpy as np
from shapely.geometry import Polygon,Point,LineString,shape
from shapely.ops import transform
from pyproj import Transformer
P=Path('work');w=json.load(open(P/'king-street/public/world.json'));ll=Transformer.from_crs(4326,32610,always_xy=True);cx,cy=ll.transform(w['center']['lon'],w['center']['lat'])
def local(x,y,z=None):
 a,b=ll.transform(x,y);return np.array(a)-cx,np.array(b)-cy
x=ET.parse(P/'data/osm.xml').getroot();nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in x.findall('node')};osm={};paths=[]
for way in x.findall('way'):
 t={v.attrib['k']:v.attrib['v'] for v in way.findall('tag')};ps=[nodes[v.attrib['ref']] for v in way.findall('nd') if v.attrib['ref'] in nodes]
 if len(ps)<2:continue
 xx,yy=local(*zip(*ps));g=LineString(np.c_[xx,yy]);oid=way.attrib['id']
 if ps[0]==ps[-1] and len(ps)>3 and t.get('building'):osm['osm_'+oid]={'g':Polygon(g.coords).buffer(0),'tags':t,'id':oid}
 if g.distance(Point(0,0))<265 and (t.get('highway')=='steps' or t.get('barrier') or t.get('highway')=='cycleway'):paths.append({'id':oid,'tags':t,'lengthM':round(g.length,1),'center':list(g.centroid.coords[0]),'radiusM':round(g.distance(Point(0,0)),1)})
municipal={'sf_'+b['sf16_bldgid'].replace('.','_'):{'g':transform(local,shape(b['shape'])),'tags':b} for b in json.load(open(P/'data/buildings.json'))}
features=w['buildings'];geos={b['id']:(osm.get(b['id']) or municipal.get(b['id']))['g'] for b in features};overlap=[];subsumed=[]
for i,b in enumerate(features):
 g=geos[b['id']]
 for other in features[i+1:]:
  q=geos[other['id']];a=g.intersection(q).area
  if a>2:overlap.append({'a':b['id'],'b':other['id'],'areaM2':round(a,1),'fractionOfSmaller':round(a/min(g.area,q.area),3)})
 for key,c in osm.items():
  if not b['id'].startswith('sf_'):continue
  a=g.intersection(c['g']).area
  if a/g.area>.7:subsumed.append({'oldId':b['id'],'newId':key,'coverage':round(a/g.area,3),'newArea':round(c['g'].area,1),'name':c['tags'].get('name',c['tags'].get('addr:street'))})
print('OVERLAPS',json.dumps(sorted(overlap,key=lambda x:-x['areaM2']),indent=2));print('PATHS',json.dumps(paths,indent=2));print('SUBSUMED',json.dumps(subsumed,indent=2))
(P/'map-input-audit.json').write_text(json.dumps({'overlaps':overlap,'paths':paths,'subsumed':subsumed},indent=2))
