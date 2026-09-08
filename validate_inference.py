"""Checks synthetic perspective/occlusion recovery and real annotation consistency."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'work/pipeline')
from infer_facades import homography,project,infer
from facade_grammar import openings
rng=np.random.default_rng(44)
checks=[]
for nx,ny,quad in [(5,4,[[80,90],[720,170],[620,710],[160,620]]),(3,7,[[300,50],[750,280],[780,860],[240,940]]),(8,2,[[70,80],[1200,140],[980,580],[130,680]])]:
 uv=np.array([[(x+.5)/nx,(y+.5)/ny] for x in range(nx) for y in range(ny)]);H=homography([[0,1],[1,1],[1,0],[0,0]],quad);pixels=project(H,uv);keep=rng.random(len(uv))>.25;keep[0]=True
 packet={'id':'synthetic','match':{},'normal':[1,0],'quadPx':quad,'region':[0,0,1,1],'style':{},'sourceIds':['synthetic'],'openingCentersPx':(pixels[keep]+rng.normal(0,.55,(keep.sum(),2))).tolist(),'holdoutCentersPx':pixels[~keep].tolist()}
 fit=infer(packet);assert fit['columns']['count']==nx,(nx,ny,fit['columns']);assert fit['rows']['count']==ny,(nx,ny,fit['rows']);error=max(fit['holdoutPixelErrors'],default=0);assert error<3,error
 checks.append({'expectedGrid':[nx,ny],'recoveredGrid':[fit['columns']['count'],fit['rows']['count']],'retainedSamples':int(keep.sum()),'totalCells':len(uv),'maxHeldOutPixelError':round(error,3)})
fits=json.load(open('work/pipeline/facade-layouts.json'))['fits'];profiles=json.load(open('work/pipeline/facade-profiles.json'))['profiles'];overlaps=[]
for p in profiles:
 ops=openings(p,fits)
 for i,a in enumerate(ops):
  x1,y1,x2,y2=a['rect'];assert x2>x1 and y2>y1
  for b in ops[i+1:]:
   xx1,yy1,xx2,yy2=b['rect'];intersection=max(0,min(x2,xx2)-max(x1,xx1))*max(0,min(y2,yy2)-max(y1,yy1))
   if intersection>1e-5:overlaps.append({'profile':p['id'],'a':a['rect'],'b':b['rect'],'overlap':intersection})
result={'syntheticPerspectiveOcclusionTests':checks,'profileOpeningOverlaps':overlaps,'realSameImageHoldouts':[{'id':f['id'],'grid':[f['columns']['count'],f['rows']['count']],'holdoutPixels':f['holdoutPixelErrors'],'independentValidation':False} for f in fits]}
Path('outputs/king-street-inference-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert not overlaps,'Conflicting opening regions'
