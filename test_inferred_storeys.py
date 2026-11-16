import pytest
from pipeline.facades import inferred_storeys,inferred_shopfront

def test_one_storey_volume_never_gets_two_squashed_floors():
    for height in (3.,3.6,4.,4.7):assert inferred_storeys(height,3.65,4.3,True)==(height,0,0.)

def test_host_above_ground_has_no_fictitious_ground_or_extra_recorded_floor():
    assert inferred_storeys(7.4,3.7,4.1,False)==(0,2,3.7)
    for height in (6.5,8.2,12.3,17,65):
        g,n,h=inferred_storeys(height,3.6,4.1,True)
        assert g+n*h==pytest.approx(height) and g>=2.4 and (not n or h>=2.4)

def test_donor_retail_grid_cannot_place_a_shopfront_on_unobserved_house():
    retail_donor={'ground':'shopfront','donorId':'foreign'}
    assert not inferred_shopfront({'building':'house'},None,None,retail_donor)
    assert inferred_shopfront({'building':'retail'},None,None,retail_donor)
    assert inferred_shopfront({},None,{'reading':{'shopfront':True}},retail_donor)
