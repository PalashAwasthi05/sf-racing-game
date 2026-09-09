"""Normalize complete source footprints; provider IDs are correspondence keys only."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon,shape
from shapely.ops import transform
w=Path(__file__).resolve().parent;world=json.loads((w/'king-street/public/world.json').read_text());frame=json.loads((w/'normalized-v9/input.json').read_text())['coordinateFrame'];project=Transformer.from_crs('EPSG:4326',frame['projectedCrs'],always_xy=True);ox,oy=project.transform(world['center']['lon'],world['center']['lat'])
def local(x,y,z=None):
 a,b=project.transform(x,y);return np.asarray(a)-ox,np.asarray(b)-oy
rows=[]
def add(key,g):
 if not g.is_valid:g=g.buffer(0)
 if g.is_empty:return
 parts=[g] if g.geom_type=='Polygon' else list(g.geoms)
 # Complete polygon geometry preserves courtyards; outer rings retained for older adapters.
 from shapely.geometry import mapping
 rows.append({'id':key,'geometry':mapping(g),'rings':[list(p.exterior.coords) for p in parts if p.geom_type=='Polygon']})
for r in json.loads((w/'data/buildings.json').read_text()):add('sf_'+r['sf16_bldgid'].replace('.','_'),transform(local,shape(r['shape'])))
xml=ET.parse(w/'data/osm.xml').getroot();nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in xml.findall('node')}
for way in xml.findall('way'):
 tags={t.attrib['k']:t.attrib['v'] for t in way.findall('tag')}
 if 'building' not in tags:continue
 pts=[nodes[n.attrib['ref']] for n in way.findall('nd') if n.attrib['ref'] in nodes]
 if len(pts)>3 and pts[0]==pts[-1]:add('osm_'+way.attrib['id'],transform(local,Polygon(pts)))
(w/'support-footprints-v11.json').write_text(json.dumps(rows,separators=(',',':')))
print({'sourceFootprints':len(rows),'historicalReferenceOnly':True})
