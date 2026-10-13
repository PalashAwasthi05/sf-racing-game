from shapely.geometry import box,mapping
import pytest
from pipeline.ground_paint import admitted_geometry

def test_multiview_patches_deduplicate_and_respect_sidewalk():
 p=box(0,0,3,3)
 f={'geometry':mapping(p),'properties':{'status':'coarse-pixel-component-supported','imageIds':['a','b'],'agreementDistanceM':.2}}
 packet={'origin':{'lat':1,'lon':2},'features':[f,f]}
 plan={'road':box(-10,-10,10,10),'cycle':box(-10,-1,10,3),'sidewalk':box(-10,2,10,8)}
 g=admitted_geometry(packet,plan,packet['origin'])
 assert g.area==6
 assert g.intersection(plan['sidewalk']).area==0
 packet['features'][0]['properties']['imageIds']=['a','a']
 assert admitted_geometry(packet,plan,packet['origin']).is_empty

def test_rejects_evidence_for_another_origin():
 with pytest.raises(ValueError):admitted_geometry({'origin':{'lat':1,'lon':2}}, {}, {'lat':3,'lon':4})
