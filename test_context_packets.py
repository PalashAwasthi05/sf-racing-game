import numpy as np
import pytest
from regen.context_packets import context_bounds,pixel_to_wall


def test_sloping_ground_and_roof_context_keep_original_datum():
    b=context_bounds(8,11,[-4.7,-2,0,1])
    assert b['zMinM']==-5.2 and b['zMaxM']==14
    ppm=40
    for x,z in [(0,0),(8,11),(2,-4.7)]:
        px=(x-b['xMinM'])*ppm-.5;py=(b['zMaxM']-z)*ppm-.5
        assert np.allclose(pixel_to_wall(px,py,b,ppm),(x,z))


def test_unbounded_ground_gap_is_reported_not_silently_complete():
    b=context_bounds(12,7,[-25,float('nan')])
    assert b['groundCoverageClipped'] and b['zMinM']==-10
    assert not context_bounds(12,7,[])['groundSamplesAvailable']


def test_invalid_frame_is_rejected():
    for dimensions in [(0,8),(10,float('nan')),(-1,4)]:
        with pytest.raises(ValueError):context_bounds(*dimensions,[])


def test_native_resolution_tracks_camera_distance_and_pixel_budget():
    from regen.context_packets import native_rect_ppm
    wall={'midpoint':[0,0],'normal':[0,-1]}
    b=context_bounds(8,10,[])
    near,_=native_rect_ppm(wall,[0,-12,2.4],0,b,(7680,3840))
    far,_=native_rect_ppm(wall,[0,-60,2.4],0,b,(7680,3840))
    assert near>far*1.5
    assert near*(b['zMaxM']-b['zMinM'])<=1600+1e-6
    low,_=native_rect_ppm(wall,[0,-60,2.4],0,b,(1920,960))
    assert np.isclose(low,far/4)
