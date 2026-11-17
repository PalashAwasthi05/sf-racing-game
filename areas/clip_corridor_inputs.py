"""Spatially select full source records and OSM way-node dependencies for a polygon AOI."""
import json,pathlib,argparse,xml.etree.ElementTree as ET
from shapely.geometry import shape,Point,LineString,Polygon
p=argparse.ArgumentParser();p.add_argument('area');a=p.parse_args();area=pathlib.Path(a.area);data=area/'data';boundary=shape(json.load(open(area/'context-boundary.geojson')));counts={}
fields={'buildings':'shape','trees':'latlon','streets':'line','signs':'shape','row':'the_geom','curbs':'the_geom','sidewalk-widths':'shape','trash-cans':'point','bike-network':'shape'}
for name,field in fields.items():
 path=data/(name+'.json');rows=json.load(open(path));keep=[]
 for r in rows:
  try:g=Point(float(r['longitude']),float(r['latitude'])) if field=='latlon' else shape(r[field])
  except (KeyError,ValueError,TypeError):continue
  if g.intersects(boundary):keep.append(r)
 path.write_text(json.dumps(keep));counts[name]={'fetched':len(rows),'selected':len(keep)}
r=ET.parse(data/'osm.xml').getroot();nodes={n.get('id'):n for n in r.findall('node')};ways={w.get('id'):w for w in r.findall('way')};keepways=set();keepnodes=set()
for wid,w in ways.items():
 coords=[(float(nodes[n.get('ref')].get('lon')),float(nodes[n.get('ref')].get('lat'))) for n in w.findall('nd') if n.get('ref') in nodes]
 if len(coords)>1 and LineString(coords).intersects(boundary):keepways.add(wid)
relations=[]
for rel in r.findall('relation'):
 tags={x.get('k'):x.get('v') for x in rel.findall('tag')}
 if tags.get('type')=='multipolygon' and any(m.get('type')=='way' and m.get('ref') in keepways for m in rel.findall('member')):
  relations.append(rel);keepways.update(m.get('ref') for m in rel.findall('member') if m.get('type')=='way' and m.get('ref') in ways)
for wid in keepways:keepnodes.update(n.get('ref') for n in ways[wid].findall('nd'))
for nid,n in nodes.items():
 if n.find('tag') is not None and boundary.covers(Point(float(n.get('lon')),float(n.get('lat')))):keepnodes.add(nid)
out=ET.Element('osm',r.attrib)
for n in r.findall('node'):
 if n.get('id') in keepnodes:out.append(n)
for w in r.findall('way'):
 if w.get('id') in keepways:out.append(w)
for rel in relations:out.append(rel)
ET.ElementTree(out).write(data/'osm.xml',encoding='utf-8',xml_declaration=True);counts['osm']={'nodes':len(keepnodes),'ways':len(keepways),'multipolygonRelations':len(relations)}
(area/'clip-report.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts))
