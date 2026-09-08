"""Prioritize missing evidence by road-visible projected area and uncertainty."""
import json,math,pathlib,hashlib,xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import Point,Polygon,LineString,shape
from shapely.ops import unary_union,transform
from pyproj import Transformer
ROOT=pathlib.Path(__file__).resolve().parents[1];e=json.load(open(ROOT.parent/'outputs/king-street-evidence.json'));world=e['world'];jobs=[]
ll=Transformer.from_crs(4326,32610,always_xy=True);cx,cy=ll.transform(world['center']['lon'],world['center']['lat'])
def local(x,y,z=None):
 a,b=ll.transform(x,y);return np.asarray(a)-cx,np.asarray(b)-cy
osm=ET.parse(ROOT/'data/osm.xml').getroot();nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in osm.findall('node')};geoms={};stations=[]
for w in osm.findall('way'):
 t={a.attrib['k']:a.attrib['v'] for a in w.findall('tag')};coords=[nodes[a.attrib['ref']] for a in w.findall('nd') if a.attrib['ref'] in nodes]
 if len(coords)<2:continue
 x,y=local(*zip(*coords));xy=np.c_[x,y];line=LineString(xy)
 if t.get('building') and len(coords)>3:geoms['osm_'+w.attrib['id']]=Polygon(xy).buffer(0)
 if t.get('highway') in ['primary','secondary','tertiary','unclassified','residential','living_street'] and line.distance(Point(0,0))<265:
  for s in np.arange(3,line.length,8):
   p=line.interpolate(s);p2=line.interpolate(min(s+.2,line.length));v=np.array(p2.coords[0])-p.coords[0]
   if p.distance(Point(0,0))<265 and np.linalg.norm(v)>.01:stations.append((np.array(p.coords[0]),v/np.linalg.norm(v)))
for b in json.load(open(ROOT/'data/buildings.json')):geoms['sf_'+b['sf16_bldgid'].replace('.','_')]=transform(local,shape(b['shape']))
features={b['id']:b for b in world['buildings']};all_geometry=unary_union([geoms[i] for i in features if i in geoms]);others={i:all_geometry.difference(geoms[i].buffer(.1)) for i in features if i in geoms};pos=np.array([p for p,t in stations])
for wall in e['facadeWalls']:
 b=features[wall['buildingId']];m=np.asarray(wall['midpoint']);n=np.asarray(wall['normal']);area=wall['lengthM']*min(wall['heightM'],12);uncertainty=.45 if wall['sourceIds'] else 1.;scores=[]
 for i in np.argsort(np.linalg.norm(pos-m,axis=1))[:32]:
  p,t=stations[i];v=p-m;distance=np.linalg.norm(v)
  if distance<3 or distance>160:continue
  incidence=float(np.dot(v/distance,n))
  if incidence<.15 or abs(float(np.dot(t,v/distance)))<.25:continue
  sight=LineString([p,m+n*.3]);obstacles=all_geometry
  if obstacles is not None and sight.intersects(obstacles):continue
  score=area*incidence/max(25,distance**2);scores.append((score,p,distance))
 importance=sum(x[0] for x in scores[:12]);best=max(scores,key=lambda x:x[0]) if scores else (0,m+n*15,15)
 stable=hashlib.sha256((wall['buildingId']+str(wall['edge'])+str(wall['midpoint'])).encode()).hexdigest()[:16]
 jobs.append({'id':stable,'buildingId':wall['buildingId'],'wallEdge':wall['edge'],'priority':round(importance*uncertainty,4),'roadVisibleCandidate':bool(scores),'testedRoadStations':len(scores),'requestedCapturePositionLocal':np.round(best[1],2).tolist(),'lookAtLocal':[round(float(m[0]),2),round(float(m[1]),2),5],'minimumViews':2,'fields':['capture date','wall-region corners','opening landmarks','ground floor exceptions','depth-bearing second view','material','sign text and pose'],'existingSources':wall['sourceIds'],'status':'needs-independent-view' if wall['sourceIds'] else 'needs-observation'})
jobs.sort(key=lambda x:x['priority'],reverse=True)
result={'schemaVersion':2,'strategy':'Rank approximate driver-visible projected area from 8 m road samples, excluding footprint-occluded rays; annotate each multi-wall capture once and cache its packet. Occlusion is a 2D approximation, not a rendered visibility guarantee.','driverEyeHeightM':1.25,'jobs':jobs,'counts':{'total':len(jobs),'withoutDirectEvidence':sum(not x['existingSources'] for x in jobs),'roadVisibleCandidates':sum(x['roadVisibleCandidate'] for x in jobs)},'qualityGate':'NOT PASSED: independent street-view geometry validation is still absent'}
(ROOT.parent/'outputs/king-street-observation-queue.json').write_text(json.dumps(result,indent=2));print(result['counts'])
