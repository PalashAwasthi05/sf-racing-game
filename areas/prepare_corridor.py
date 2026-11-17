"""Select full city blocks paired across a mapped corridor; preserve whole footprints.
All geographic choices come from per-area source/configuration, never render rules.
"""
import json,pathlib,argparse,math
from shapely.geometry import shape,mapping,Point,LineString
from shapely.ops import transform,unary_union
from pyproj import Transformer
p=argparse.ArgumentParser();p.add_argument('area');a=p.parse_args();area=pathlib.Path(a.area);data=area/'data';src=json.load(open(area/'corridor-source.json'));cfg=json.load(open(area/'config.json'));crs=src.get('projectedCrs','EPSG:32610');fwd=Transformer.from_crs(4326,crs,always_xy=True).transform;rev=Transformer.from_crs(crs,4326,always_xy=True).transform
route=transform(fwd,shape(json.load(open(area/src.get('selectionRoute','route-physical.geojson')))));spacing=src.get('blockSampleSpacingM',5);minarea=src.get('minimumCityBlockAreaM2',1000);reach=src.get('blockSearchReachM',160);minfront=src.get('minimumBlockFrontageM',25)
blocks=[(r,transform(fwd,shape(r['multipolygon'])).buffer(0)) for r in json.load(open(data/'city-blocks.json'))];blocks=[(r,g) for r,g in blocks if g.area>=minarea];hits={};samples=[]
for i in range(math.ceil(route.length/spacing)+1):
 station=min(route.length,i*spacing);pt=route.interpolate(station);before=route.interpolate(max(0,station-2));after=route.interpolate(min(route.length,station+2));dx,dy=after.x-before.x,after.y-before.y;length=math.hypot(dx,dy);nx,ny=-dy/length,dx/length;row=[]
 for side in [-1,1]:
  # Census boundaries often follow either carriageway, not the street midpoint.
  # First land block beyond the ROW on each perpendicular side wins.
  found=None
  for distance in range(src.get('blockRayStartM',14),reach+1,2):
   q=Point(pt.x+side*nx*distance,pt.y+side*ny*distance)
   candidates=[(r,g) for r,g in blocks if g.covers(q)]
   if candidates:found=min(candidates,key=lambda x:x[1].area);break
  if found:
   key=found[0]['geoid20'];hits[key]=hits.get(key,0)+spacing;row.append(key)
 samples.append({'stationM':station,'firstBlocks':row})
selected=[(r,g) for r,g in blocks if hits.get(r['geoid20'],0)>=minfront or any(g.covers(transform(fwd,Point(pt))) for pt in src.get('endpointIncludePoints',[]))]
blockunion=unary_union([g for _,g in selected]);base=blockunion.union(route.buffer(src.get('routeRowHalfWidthM',24))).buffer(0)
footpath=area/'source-cache/buildings.json';footpath=footpath if footpath.exists() else data/'buildings.json';full=[]
for r in json.load(open(footpath)):
 g=transform(fwd,shape(r['shape'])).buffer(0)
 if g.intersection(base).area>min(10,g.area*.1):full.append(g)
study=unary_union([base,*full]).buffer(0);context=study.buffer(src.get('contextMarginM',25));center=cfg['region']['center'];cx,cy=fwd(center['lon'],center['lat'])
for name,g in [('study-boundary.geojson',study),('context-boundary.geojson',context)]: (area/name).write_text(json.dumps(mapping(transform(rev,g)),indent=2))
(area/'selected-blocks.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'geoid20':r['geoid20'],'sampledFrontageM':hits.get(r['geoid20'],0)},'geometry':r['multipolygon']} for r,_ in selected]},indent=2))
cfg['region'].update({'studyGeometry':mapping(transform(rev,study)),'contextGeometry':mapping(transform(rev,context))});(area/'config.json').write_text(json.dumps(cfg,indent=2));report={'center':center,'projectedCrs':crs,'originProjectedM':[cx,cy],'routeLengthM':route.length,'blockCount':len(selected),'selectedBlockIds':[r['geoid20'] for r,_ in selected],'studyAreaM2':study.area,'contextAreaM2':context.area,'source':'DataSF official Census2020 city block polygons; actual connected OSM route','selection':'Every5m, paired perpendicular rays select first land block on each side beyond14m ROW, sustained25m minimum frontage; minimum1000m2 block area excludes median/island slivers; endpoint property-containing blocks retained. Simultaneous whole-footprint union against fixed base prevents cascading.','parameters':{k:v for k,v in src.items() if k not in ['routeSegmentIds']},'samples':samples};(area/'aoi-report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['samples','parameters','selectedBlockIds']})
