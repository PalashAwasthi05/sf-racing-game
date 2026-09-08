"""Check false-opening vetoes, edge fitting, actual sign geometry and material identity.
These are compiler checks, not a photographic accuracy score.
"""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from pipeline.surface_evidence import admissible,resolve_material
from pipeline.facade_grammar import openings
from pipeline.fit_boundaries import infer
from pipeline.infer_facades import homography,project
from pipeline.lettering import contours,attachment
from pipeline.materials import Materials

surface={'surfaceEvidence':[{'rect':[.4,.2,.6,.8],'label':'solid'}]}
assert not admissible({'rect':[.3,.3,.7,.7]},surface)
assert not admissible({'rect':[.3,.3,.7,.7]},surface,explicit=True)
assert admissible({'rect':[.01,.1,.2,.2]},surface)
assert admissible({'rect':[.3,.3,.7,.7]},{'surfaceEvidence':[{'rect':[.4,.2,.6,.8],'label':'unknown'}]})
assert not admissible({'rect':[0,0,1,1]},{'openingPolicy':'observed-only'})
assert admissible({'rect':[0,0,1,1]},{'openingPolicy':'observed-only'},explicit=True)
explicit={'rect':[.2,.15,.6,.7],'style':{}}
assert not admissible({'rect':[.2,.2,.65,.75]},{'openings':[explicit]})
assert admissible(explicit,{'openings':[explicit]},explicit=True)
# The same contradictory input is repaired for arbitrary names and geometry.
for name in ['unfamiliar warehouse','renamed example','case-900']:
 assert resolve_material({'id':name,'material':'concrete','materialEvidence':{'family':'render','sourceIds':['test-photo']}})=='stucco'
 assert resolve_material({'id':name,'material':'brick','materialEvidence':{'family':'terracotta','finish':'ribbed','sourceIds':['test-photo']}})=='terracotta_ribbed'
 a={'id':name,'bottomM':9,'centerU':.1,'depthM':.2,'attachment':{'kind':'opening-canopy','profileId':name,'openingId':'door'}}
 ctx={'PROFILES':[{'id':name,'openings':[{'id':'door','rect':[.5,0,.8,.2]}]}],'facade_records':[{'buildingId':name,'grammarId':name,'heightM':20}]}
 z,u,d,check=attachment(a,ctx,{'id':name},[]);assert abs(z-4.19)<1e-8 and abs(u-.65)<1e-8 and check['changed']
 a={'id':name,'profileId':name,'bottomM':4,'centerU':.5,'depthM':-.4,'attachment':{'kind':'glazing','nodeId':'wall'},'carrierEvidence':{'type':'opaque-surface','sourceIds':['test-photo']}}
 z,u,d,check=attachment(a,{}, {'id':name},[{'profile':{'id':name},'nodes':[{'id':'wall','depthM':.5}]}]);assert abs(d-.525)<1e-8 and check['kind']=='envelope-surface'
# A rectangle under a projective camera must recover all four edges and its size.
quad=[[80,20],[710,120],[630,900],[140,760]];h=homography([[0,1],[1,1],[1,0],[0,0]],quad)
uv=[[.21,.79],[.44,.79],[.44,.31],[.21,.31]]
p={'id':'synthetic-edges','profileId':'unfamiliar','quadPx':quad,'region':[0,0,1,1],'sourceIds':['synthetic'],'openings':[{'id':'one','cornersPx':project(h,uv).tolist(),'style':{}}]}
fit=infer(p);assert np.allclose(fit['openings'][0]['rect'],[.21,.31,.44,.79],atol=1e-8)
# A hollow glyph remains hollow, and exact source text is never interpreted as code.
glyph,w,h=contours('O');assert sum(len(p.interiors) for p in getattr(glyph,'geoms',[glyph]))>=1
bank=Materials(Path('work/material-assets'),{});stone,_=bank.get('stone');concrete,_=bank.get('concrete');assert stone is not concrete and stone.name!='concrete'
for name in ['brick_modular','brick_dark','terracotta_panel','terracotta_ribbed','stucco','panel_dark','panel_silver']:
 m,scale=bank.get(name);assert m.name==name and min(scale)>0
profiles=json.load(open('work/pipeline/facade-profiles.json'))['profiles'];fits=json.load(open('work/pipeline/facade-layouts.json'))['fits'];bf=json.load(open('work/pipeline/boundary-layouts.json'))['fits']
for f in bf:next(p for p in profiles if p['id']==f['profileId']).setdefault('openings',[]).extend(f['openings'])
for p in profiles:
 ops=openings(p,fits)
 for op in ops:
  a,b,c,d=op['rect'];assert 0<=a<c<=1.001 and 0<=b<d<=1.001,(p['id'],op)
  assert admissible(op,p,explicit=True)
  for other in ops:
   if op is other:continue
   aa,bb,cc,dd=other['rect'];area=max(0,min(c,cc)-max(a,aa))*max(0,min(d,dd)-max(b,bb));assert area<1e-5,(p['id'],op,other)
result={'solidEvidenceOverridesWindows':True,'unknownNotMisclassifiedAsSolid':True,'explicitBoundsOverrideRepeat':True,'contradictoryMaterialsResolvedForUnfamiliarSurfaces':True,'detachedSignsReattachedForUnfamiliarNames':True,'syntheticProjectiveEdgeRecovery':True,'hollowLetterContoursPreserved':True,'stoneDistinctFromConcrete':True,'realOpeningBoundsValidAndDisjoint':True,'realBoundaryFitCount':sum(len(f['openings']) for f in bf),'realSameImageMeanCornerResidualPx':bf[0]['meanCornerResidualPx'],'independentPhotographicAccuracy':False}
Path('outputs/king-street-surface-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
