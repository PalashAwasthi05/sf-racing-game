import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
from regen.constrain_front_yards import guarded_local_extent, constrain


def test_road_between_building_and_far_side_garden_removes_it_without_relocation():
    assert guarded_local_extent(0, 9, 8, 2, box(-2, 2, 11, 6)) is None
    assert guarded_local_extent(0, 9, .5, 1, box(-2, 2, 11, 6)) == (0, 9, .5, 1)


def test_shrub_can_thin_but_fence_keeps_thickness_and_shortens():
    road = box(2.5, 1, 20, 20)
    shrink = guarded_local_extent(0, 9, .5, 1, road)
    fence = guarded_local_extent(0, 9, .5, 1, road, fixed_depth=True)
    assert shrink[1] > 8.9 and .3 < shrink[3] < .5
    assert 2 < fence[1] < 2.5 and fence[3] == 1


def test_rotated_translated_facade_changes_only_low_private_planting():
    p = {'normal':[1,0], 'frameU':[10,18], 'planeV':25,
         'streetEdge':[{'kind':'driveway','x':[0,1],'offsetM':5}],
         'planting':[{'kind':'shrub','x':[0,1],'offsetM':6},
                     {'kind':'hedge','x':[0,1],'offsetM':6,'baseM':3.0}]}
    obstacles = [box(27,9,30,19)]
    out, events = constrain(p,STRtree(obstacles),obstacles)
    assert out['streetEdge'] == p['streetEdge']
    assert out['planting'] == p['planting'][1:] and len(events) == 1 and len(p['planting']) == 2


def test_already_clipped_extent_is_stable():
    road=box(-1,2,10,7)
    a=guarded_local_extent(0,8,1,3,road)
    assert np.allclose(a,(0,8,1,.85))
    assert np.allclose(guarded_local_extent(*a,road),a)
