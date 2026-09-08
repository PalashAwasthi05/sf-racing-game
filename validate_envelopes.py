"""Structural regression checks; no claim of photographic accuracy."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from shapely.geometry import box,Point
sys.path.insert(0,str(Path(__file__).parent))
from pipeline import envelopes
rng=np.random.default_rng(140)
# A near-street roof and a taller background tower must resolve independently.
samples=np.c_[rng.uniform(0,20,1600),rng.uniform(-6,0,1600),rng.normal(22,.08,1600)]
samples=np.r_[samples,np.c_[rng.uniform(0,20,3000),rng.uniform(-35,-20,3000),rng.normal(53,.1,3000)]]
f={'tangent':np.array([1.,0]),'normal':np.array([0.,1.]),'left':0.,'width':20.,'plane':0.}
p={'levels':{'street':{'priorM':23,'searchRadiusM':3},'background':{'priorM':53,'searchRadiusM':2,'depthRangeM':[-35,-20]}}}
levels,ev=envelopes.fit_levels(p,f,samples,0);assert abs(levels['street']-22)<.25 and abs(levels['background']-53)<.25
# Translation/rotation of the same sampled building must not change fitted datums.
angle=.43;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);offset=np.array([81,-29]);ss=samples.copy();ss[:,:2]=ss[:,:2]@R.T+offset
ff={**f,'tangent':R@f['tangent'],'normal':R@f['normal']};ff['left']=offset@ff['tangent'];ff['plane']=offset@ff['normal'];other,_=envelopes.fit_levels(p,ff,ss,0)
assert max(abs(levels[k]-other[k]) for k in levels)<1e-9
# Hierarchy: nested region uses local coordinates and accumulates signed depth.
nodes=envelopes.resolved_nodes({'replaceDepthM':8,'nodes':[{'id':'a','span':[.2,.8],'z':[2,20],'depthM':.5,'children':[{'id':'b','rect':[.25,.1,.75,.8],'offsetM':-.9}]}]}, {})
assert np.allclose(nodes[1]['rect'],[.35,3.8,.65,16.4]);assert abs(nodes[1]['depthM']+.4)<1e-9
# Compile an unfamiliar rotated building containing a nonrectangular opening.
meshes=[]
ctx={'put':lambda m,k,c:meshes.append((m,k,c)),'cube':lambda *a:None,'cylinder':lambda *a:None}
profile={'id':'test-unfamiliar','match':{'featureId':'test'},'normal':[0,1],'replaceDepthM':2,'levels':{},'nodes':[{'id':'test-surface','span':[0,1],'z':[0,9],'material':'stone','depthM':0,'grids':[{'columns':1,'rows':1,'style':{'shape':[[0,0],[1,0],[1,.65],[.75,1],[.25,1],[0,.65]],'widthRatio':.5,'heightRatio':.5}}]}]}
c={'id':'test','g':box(0,-4,12,0)};r=envelopes.prepare({},c,0,np.empty((0,3)),[profile]);records,n=envelopes.compile(ctx,c,0,r)
assert n==1 and len(meshes)>5
assert all(np.isfinite(m.vertices).all() and (m.area>0) for m,k,col in meshes)
# Check wall skin doesn't fill the aperture: ray through its center sees no exterior wall.
wall=[m for m,k,col in meshes if k.startswith('buildings')]
for m in wall:
 triangles=m.triangles;front=np.all(np.isclose(triangles[:,:,1],0),axis=1)
 from shapely.geometry import Polygon
 assert not any(Polygon(t[:,[0,2]]).contains(Point(6,4.5)) for t in triangles[front])
# Shared corner and named roof level constraints are independent of window rules.
pa={'id':'a','normal':[0,1],'match':{'featureId':'test'},'referenceLine':[[0,0],[11.5,0]],'levels':{'roof':{'priorM':20,'searchRadiusM':2}},'shareLevels':['roof'],'joins':[{'end':'start','profile':'b','otherEnd':'end'}],'nodes':[{'id':'a','span':[0,1],'z':[0,'roof']}]}
pb={'id':'b','normal':[1,0],'match':{'featureId':'test'},'referenceLine':[[12,-8],[12,-.4]],'levels':{'roof':{'priorM':21,'searchRadiusM':2}},'shareLevels':['roof'],'nodes':[{'id':'b','span':[0,1],'z':[0,'roof']}]}
rr=envelopes.prepare({},c,0,np.empty((0,3)),[pa,pb]);assert rr[0]['joins'][0]['accepted'];assert rr[0]['levels']['roof']==rr[1]['levels']['roof']
f1,f2=[v['frame'] for v in rr];corner_a=f1['tangent']*f1['left']+f1['normal']*f1['plane'];corner_b=f2['tangent']*(f2['left']+f2['width'])+f2['normal']*f2['plane'];assert np.linalg.norm(corner_a-corner_b)<1e-9
profiles=json.load(open('work/pipeline/envelope-observations.json'))['profiles']
for p in profiles:
 ls={k:v['priorM'] for k,v in p['levels'].items()};ns=envelopes.resolved_nodes(p,ls);ids=[n['id'] for n in ns];assert len(ids)==len(set(ids)),p['id']
 for n in ns:
  assert 0<=n['rect'][0]<n['rect'][2]<=1
  if n['parentId']:
   par=next(q for q in ns if q['id']==n['parentId']);assert box(*par['rect']).buffer(1e-8).covers(box(*n['rect']))
report={'testsPassed':['roof-vs-background separation','rotation and translation invariance','hierarchical region and signed depth resolution','shared corners and explicit roof datums','nonrectangular aperture stays open','finite positive-area geometry','all real profile children contained and IDs unique'],'profiles':len(profiles),'sameImageGroundTruth':False,'independentPhotoAccuracyValidated':False}
Path('outputs/king-street-envelope-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
