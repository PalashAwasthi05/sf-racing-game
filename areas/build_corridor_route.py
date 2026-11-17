"""Build one connected mapped route, preserving access-mode provenance."""
import argparse,pathlib,json,xml.etree.ElementTree as ET,heapq,math
from shapely.geometry import Point,LineString,mapping
from shapely.ops import transform
from pyproj import Transformer
p=argparse.ArgumentParser();p.add_argument('area');a=p.parse_args();area=pathlib.Path(a.area);src=json.load(open(area/'corridor-source.json'));r=ET.parse(area/'data/osm.xml').getroot();nodes={n.get('id'):(float(n.get('lon')),float(n.get('lat'))) for n in r.findall('node')};crs=src['projectedCrs'];f=Transformer.from_crs(4326,crs,always_xy=True).transform;inv=Transformer.from_crs(crs,4326,always_xy=True).transform;xy={k:f(*v) for k,v in nodes.items()};edges={};segments={}
for w in r.findall('way'):
 t={x.get('k'):x.get('v') for x in w.findall('tag')}
 if t.get('name')!=src['mappedRoadName'] or t.get('highway') not in ['tertiary','primary','secondary','unclassified','cycleway','footway']:continue
 ns=[x.get('ref') for x in w.findall('nd')]
 for u,v in zip(ns,ns[1:]):
  d=math.dist(xy[u],xy[v]);edges.setdefault(u,[]).append((v,d));segments[(u,v)]=(w.get('id'),t)
  if t.get('oneway')!='yes' or t.get('oneway:bicycle')=='no':edges.setdefault(v,[]).append((u,d));segments[(v,u)]=(w.get('id'),t)
start=min(edges,key=lambda k:math.dist(xy[k],f(*src['routeStartWgs84'])));goal=min(edges,key=lambda k:math.dist(xy[k],f(*src['routeSearchEndWgs84'])));q=[(0,start)];dist={start:0};prev={}
while q:
 d,u=heapq.heappop(q)
 if d!=dist[u]:continue
 if u==goal:break
 for v,w in edges[u]:
  nd=d+w
  if nd<dist.get(v,float('inf')):dist[v]=nd;prev[v]=u;heapq.heappush(q,(nd,v))
assert goal in dist,'No connected route';path=[goal]
while path[-1]!=start:path.append(prev[path[-1]])
path.reverse();line=LineString([nodes[k] for k in path]);metric=transform(f,line);(area/'route-drive.geojson').write_text(json.dumps(mapping(line),indent=2));cfg=json.load(open(area/'config.json'));anchors=[]
for i in range(10):
 d=metric.length*i/9;p=metric.interpolate(d);t=metric.interpolate(min(metric.length,d+45));
 if d==metric.length:p=metric.interpolate(d-8)
 anchors.append({'position':list(inv(p.x,p.y)),'lookAt':list(inv(t.x,t.y)),'label':(src['routeStartLabel'] if i==0 else src['routeEndLabel'] if i==9 else f'Corridor {i}'),'lookHeightM':4})
cfg['region']['routeAnchors']=anchors;cfg['region']['routeGeometry']=mapping(line);cfg['region']['routeLengthM']=metric.length;cfg['region']['routeMode']='bicycle / visual traversal; includes waterfront plaza and transit street';(area/'config.json').write_text(json.dumps(cfg,indent=2));used=[]
for u,v in zip(path,path[1:]):
 wid,t=segments[(u,v)]
 if not used or used[-1]['wayId']!=wid:used.append({'wayId':wid,'highway':t['highway'],'bicycle':t.get('bicycle'),'oneway':t.get('oneway'),'oneway:bicycle':t.get('oneway:bicycle'),'oneway:bus':t.get('oneway:bus')})
report={'lengthM':metric.length,'start':nodes[start],'end':nodes[goal],'direction':src.get('routeDirectionDescription','Configured route start to terminal property'),'connected':True,'mode':'Private simulation along the configured mapped corridor.','ways':used};(area/'route-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
# Keep the physical connection to the waterfront distinct from simulation driving.
road_start=next(i for i,(u,v) in enumerate(zip(path,path[1:])) if segments[(u,v)][1]['highway'] not in ['footway','cycleway'])
physical=line;connector=LineString(list(line.coords)[:road_start+1]);drive=LineString(list(line.coords)[road_start:]);metric=transform(f,drive)
for name,g in [('route-physical.geojson',physical),('route-walk.geojson',connector),('route-drive.geojson',drive)]: (area/name).write_text(json.dumps(mapping(g),indent=2))
anchors=[]
for i in range(10):
 d=metric.length*i/9;p=metric.interpolate(d if i<9 else d-8);t=metric.interpolate(min(metric.length,d+45));anchors.append({'position':list(inv(p.x,p.y)),'lookAt':list(inv(t.x,t.y)),'label':(src['routeStartLabel'] if i==0 else src['routeEndLabel'] if i==9 else f'Corridor {i}'),'lookHeightM':4})
cfg['region'].update({'routeAnchors':anchors,'routeGeometry':mapping(drive),'routeLengthM':metric.length,'routeMode':'private simulation along mapped transit/bicycle carriageway'});(area/'config.json').write_text(json.dumps(cfg,indent=2));report.update({'physicalLengthM':transform(f,physical).length,'walkConnectorLengthM':transform(f,connector).length,'driveLengthM':metric.length,'driveStart':list(drive.coords[0]),'driveEnd':list(drive.coords[-1]),'mode':'Private simulation on mapped transit/bicycle carriageway. Separate physical waterfront foot/cycle connector is modeled, but not used for driving spawn.'});(area/'route-report.json').write_text(json.dumps(report,indent=2))
