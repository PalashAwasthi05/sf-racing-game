"""Runtime source adapter. All area/schema/CRS parameters are external job data."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon,Point,LineString,shape,mapping
from shapely.ops import transform,unary_union
from shapely import make_valid

p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',required=True);a=p.parse_args();jobpath=Path(a.job);j=json.load(open(jobpath));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
ll=Transformer.from_crs(j['geographicCrs'],j['projectedCrs'],always_xy=True);cp=Transformer.from_crs(j['cloudCrs'],j['projectedCrs'],always_xy=True);cx,cy=ll.transform(j['center']['lon'],j['center']['lat'])
def local(x,y,z=None):
 xx,yy=ll.transform(x,y);return np.asarray(xx)-cx,np.asarray(yy)-cy
boundary=Point(0,0).buffer(j['contextRadiusM'],quad_segs=64);study=Point(0,0).buffer(j['studyRadiusM']+25)
def loadpath(k):return jobpath.parent/j['paths'][k]
xml=ET.parse(loadpath('osm')).getroot();ns={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in xml.findall('node')};ways=[]
for w in xml.findall('way'):
 tags={t.attrib['k']:t.attrib['v'] for t in w.findall('tag')};coords=[ns[n.attrib['ref']] for n in w.findall('nd') if n.attrib['ref'] in ns]
 if len(coords)<2:continue
 x,y=local(*zip(*coords));xy=list(zip(x,y));g=Polygon(xy) if len(xy)>3 and xy[0]==xy[-1] else LineString(xy);g=make_valid(g)
 if g.intersects(boundary):ways.append({'key':'mapped:'+w.attrib['id'],'g':g,'tags':tags})
# Prefer complete explicit outlines. Old intersecting records are fallback pieces,
# selected by geometric overlap only; no list of individually favoured features.
current=[w for w in ways if 'building' in w['tags'] and w['g'].area>40 and w['g'].intersects(study)];old=[]
for raw in json.load(open(loadpath('municipalFootprints'))):
 g=transform(local,shape(raw[j['mapping']['geometry']])).buffer(0)
 if g.area<65 or not g.intersects(study):continue
 try:h=float(raw.get(j['mapping']['height'],0))
 except (TypeError,ValueError):h=None
 old.append({'key':'survey:'+str(raw[j['mapping']['identifier']]),'g':g,'heightM':h})
# Choose mapped outline if it substantially covers one or more old records;
# preserve historical fallback if mapped coverage is absent. This is source fusion,
# not evidence that a source was recently surveyed.
selected=[];covered_old=set()
for w in current:
 overlaps=[(i,c['g'].intersection(w['g']).area/max(1,c['g'].area)) for i,c in enumerate(old) if c['g'].intersects(w['g'])]
 for i,r in overlaps:
  if r>.55:covered_old.add(i)
 h=None
 try:h=float(str(w['tags'].get('height','')).split()[0])
 except (ValueError,IndexError):pass
 selected.append({'key':w['key'],'g':w['g'],'heightM':h,'sourcePriority':1})
selected.extend({**c,'sourcePriority':0} for i,c in enumerate(old) if i not in covered_old)
# Geometric tie order is invariant under relabelling; opaque IDs never win ownership.
selected.sort(key=lambda c:(-c['sourcePriority'],-c['g'].area,c['g'].centroid.x,c['g'].centroid.y));owned=Polygon();features=[]
for c in selected:
 g=c['g'].intersection(boundary).difference(owned).buffer(0)
 if g.area<4:continue
 owned=owned.union(g);features.append({'key':c['key'],'geometry':mapping(g),'heightM':c['heightM'],'sourceConfidence':'uncertain-source-age'})
cloud=np.load(loadpath('cloud'))['points'];x,y=cp.transform(cloud[:,0],cloud[:,1]);cloud[:,0]=x-cx;cloud[:,1]=y-cy;inside=np.linalg.norm(cloud[:,:2],axis=1)<j['contextRadiusM']+20;cloud=cloud[inside];ground=cloud[np.isin(cloud[:,3],j['classCodes']['ground']),:3];base=float(np.median(ground[:,2]));ground[:,2]-=base
surface=cloud[~np.isin(cloud[:,3],j['classCodes']['ground']+j['classCodes']['excluded']+j['classCodes']['vegetation']),:3];surface[:,2]-=base
np.savez_compressed(out/'ground.npz',points=ground);np.savez_compressed(out/'surface-points.npz',points=surface)
surfaces=[{'key':'ground','layer':'terrain','geometry':mapping(boundary),'offsetM':-.01,'color':[150,155,151,255]}];stairs=[]
for w in ways:
 t,g=w['tags'],w['g'];h=t.get('highway')
 if g.geom_type!='LineString':continue
 if h in ['primary','secondary','tertiary','residential','unclassified','living_street','service','cycleway','footway','path','pedestrian']:
  try:width=float(t['width'])
  except (KeyError,ValueError):
   try:width=max(1,int(t.get('lanes',2)))*3.15 if h not in ['cycleway','footway','path','pedestrian'] else 1.8
   except ValueError:width=6.3
  layer='roads' if h not in ['footway','path','pedestrian'] else 'paths';surfaces.append({'key':w['key'],'layer':layer,'geometry':mapping(g.buffer(width/2).intersection(boundary).difference(owned)),'offsetM':.025 if layer=='roads' else .04,'color':[78,84,87,255] if layer=='roads' else [174,178,173,255]})
 if h=='steps':
  try:count=int(t['step_count'])
  except (KeyError,ValueError):count=max(2,round(g.length/.32))
  if boundary.covers(g):stairs.append({'key':w['key'],'path':list(g.coords),'count':count,'handrail':t.get('handrail')=='yes'})
d={'schema':'normalized-world-input/v1','snapshotDate':j['snapshotDate'],'groundFile':'ground.npz','surfacePointsFile':'surface-points.npz','buildings':features,'surfaces':surfaces,'stairs':stairs,'images':{},'derivedDetails':[],'provenance':{'automaticNormalization':True,'manualFacadeSourcesRead':0,'sourceJobHashInput':str(jobpath.name)}}
d['coordinateFrame']={'projectedCrs':j['projectedCrs'],'originProjectedM':[cx,cy],'verticalOriginM':base,'units':'m','axes':'east,north,up'}
(out/'input.json').write_text(json.dumps(d,separators=(',',':')));print({'buildings':len(features),'groundPoints':len(ground),'surfacePoints':len(surface),'rawSurfaces':len(surfaces),'stairs':len(stairs)})
