"""Scene-wide physical invariants and unfamiliar-input regressions, not photo accuracy."""
import sys,json
from pathlib import Path
import numpy as np,trimesh
from shapely.geometry import box,Polygon
from rasterio.transform import from_origin
sys.path.insert(0,str(Path(__file__).parent))
from pipeline import spatial_rules as s,mobility,envelopes,freshness
# Foundation bottoms cover every sampled boundary even with a slope in two axes.
made=[];c={'id':'unfamiliar-sloping-site','g':box(-8,-12,22,10)}
f=s.foundation({'polys':lambda g:[g],'elev':lambda x,y:.06*x-.08*y,'extrude':lambda *a:made.append(a)},c,1.7)
assert f['maximumUnfilledGapM']==0 and f['originalMaximumGapM']>1 and made[0][1]<-1
# Ownership never generates intersecting building volumes from two overlapping sources.
a={'id':'old','g':box(0,0,12,12),'old':{},'osm':None,'fp':'DataSF footprint'};b={'id':'new','g':box(8,4,18,16),'old':{},'osm':None,'fp':'OSM footprint'}
cs,audit=s.reconcile([a,b]);assert cs[0]['g'].intersection(cs[1]['g']).area<1e-9 and audit
# Nested contours own disjoint vertical intervals and disjoint exposed caps.
q=np.zeros((30,30),np.int16);q[2:28,2:28]=5;q[7:23,7:23]=10;q[11:20,11:20]=15
bs=s.roof_bands(q,from_origin(0,30,1,1),box(0,0,30,30),10)
assert len(bs)==3 and [(x['bottom'],x['top']) for x in bs]==[(0,5),(5,10),(10,15)]
for a,b in zip(bs,bs[1:]):assert a['geometry'].buffer(1e-8).covers(b['geometry']) and a['cap'].intersection(b['geometry']).area<1e-9
# Equal-depth adjoining root faces do not create duplicate buried return walls.
ns=envelopes.resolved_nodes({'nodes':[{'id':'a','span':[0,.5],'z':[0,10],'depthM':0},{'id':'b','span':[.5,1],'z':[0,10],'depthM':0}]},{})
sh={n['id']:box(*n['rect']) for n in ns};edges=list(envelopes.return_edges(ns[0],sh['a'],ns,sh));assert len(edges)==3
ns[1]['depthM']=-.7;edges=list(envelopes.return_edges(ns[0],sh['a'],ns,sh));assert any(abs(d+.7)<1e-8 for a,b,d in edges)
# Arch profiles have curved heads, not rectangular openings beneath an arch decal.
p=Polygon(envelopes.aperture_shape({'shapeKind':'arch','springRatio':.6}));assert .85<p.area<1 and not p.covers(box(.01,.98,.1,.99))
# Coincident triangles with swapped winding are removed deterministically.
m=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[2,1,0]],process=False);assert s.remove_duplicate_triangles(m)==1 and len(m.faces)==1
# Side-specific and both-side tags stay distinct; separated tracks are not lane widths.
a=mobility.spec({'lanes':'2','cycleway:both':'lane','parking:lane:left':'diagonal'});assert a['bike']=={'left':1.8,'right':1.8} and a['leftM']>a['rightM']
a=mobility.spec({'lanes':'1','oneway':'yes','cycleway':'lane'});assert a['bike']=={'left':0,'right':1.8}
a=mobility.spec({'cycleway':'separate'});assert sum(a['bike'].values())==0
# Publication dates and directory names never satisfy photographic freshness or literal text.
assert freshness.evaluate({'publishedDate':'2026-09-08'},'2026-09-08')['status']=='capture-date-unknown'
assert freshness.evaluate({'captureDate':'2023-01'},'2026-09-08')['status']=='stale-review'
assert not freshness.sign_allowed({'text':'New business','textEvidence':'directory'})
assert not freshness.sign_allowed({'text':'Old business','supersededBy':'lease-change-event'})
report={'testsPassed':['sloped foundation supports all boundary samples','single building owner for overlapping sources','nested roof volumes and exclusive caps','adjacent facade returns own only exposed depth','arched aperture geometry','duplicate triangle removal across winding','both-side bike lanes and asymmetric parking','capture date distinct from publication','changed or unobserved lettering excluded'],'independentStreetAccuracyValidated':False}
Path('outputs/king-street-spatial-validation.json').write_text(json.dumps(report,indent=2));print(report)
