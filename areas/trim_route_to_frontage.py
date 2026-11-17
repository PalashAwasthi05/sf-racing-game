"""Trim a connected mapped corridor at its configured terminal property's far frontage."""
import argparse,json,pathlib,math
from shapely.geometry import shape,LineString,Point,mapping
from shapely.ops import transform,substring
from pyproj import Transformer
p=argparse.ArgumentParser();p.add_argument('area');a=p.parse_args();area=pathlib.Path(a.area);src=json.load(open(area/'corridor-source.json'));cfg=json.load(open(area/'config.json'));f=Transformer.from_crs(4326,src['projectedCrs'],always_xy=True).transform;r=Transformer.from_crs(src['projectedCrs'],4326,always_xy=True).transform
route=transform(f,shape(json.load(open(area/'route-drive.geojson'))));rows=json.load(open(area/'source-cache/buildings.json'));raw=next(x for x in rows if str(x[src['terminalFootprintIdField']])==src['terminalFootprintId']);building=transform(f,shape(raw['shape']));polys=list(building.geoms) if hasattr(building,'geoms') else [building];edges=[]
for poly in polys:
 for a,b in zip(poly.exterior.coords,list(poly.exterior.coords)[1:]):
  edge=LineString([a,b]);mid=edge.interpolate(.5,normalized=True);station=route.project(mid);q=route.interpolate(max(0,station-3));s=route.interpolate(min(route.length,station+3));d=(s.x-q.x,s.y-q.y);e=(b[0]-a[0],b[1]-a[1]);parallel=abs(e[0]*d[0]+e[1]*d[1])/max(.001,math.hypot(*e)*math.hypot(*d))
  if edge.length>5 and parallel>.95:edges.append((edge.distance(route),edge,a,b))
nearest=min(x[0] for x in edges);facades=[x for x in edges if x[0]<nearest+2];station=max(route.project(Point(p)) for _,_,a,b in facades for p in [a,b]);trim=substring(route,0,station);physical=transform(f,shape(json.load(open(area/'route-physical.geojson'))));walk=transform(f,shape(json.load(open(area/'route-walk.geojson'))));physical=LineString(list(walk.coords)+list(trim.coords)[1:])
for name,g in [('route-drive.geojson',trim),('route-physical.geojson',physical)]: (area/name).write_text(json.dumps(mapping(transform(r,g)),indent=2))
center=cfg['region']['center'];cx,cy=f(center['lon'],center['lat']);anchors=[]
for i in range(10):
 d=trim.length*i/9;q=trim.interpolate(d if i<9 else d-8);t=trim.interpolate(min(trim.length,d+45));anchors.append({'position':list(r(q.x,q.y)),'lookAt':list(r(t.x,t.y)),'label':src['routeStartLabel'] if i==0 else src['routeEndLabel'] if i==9 else f"{cfg['region']['label']} {i}",'lookHeightM':4})
cfg['region'].update({'routeGeometry':mapping(transform(r,trim)),'routeLengthM':trim.length,'routeAnchors':anchors});(area/'config.json').write_text(json.dumps(cfg,indent=2));report=json.load(open(area/'route-report.json'));report.update({'lengthM':physical.length,'end':list(r(*trim.coords[-1])),'driveLengthM':trim.length,'physicalLengthM':physical.length,'driveEnd':list(r(*trim.coords[-1])),'terminalFrontage':{'footprintId':src['terminalFootprintId'],'facadeEdgesM':[e.length for _,e,_,_ in facades],'nearestLaneDistanceM':nearest,'farProjectedStationM':station,'method':'Facade edges >5m parallel cosine>.95 to local mapped route, closest within2m, farthest endpoint projected to route'}});(area/'route-report.json').write_text(json.dumps(report,indent=2));print(report['terminalFrontage'],report['driveEnd'])
