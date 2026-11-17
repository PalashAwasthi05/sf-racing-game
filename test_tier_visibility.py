import numpy as np
from regen.tier_visibility import seen_share
from regen.side_grouping import sides_of


def test_tier_keeps_its_actual_base_and_separate_storeys():
    def wall(base,start):
        return dict(midpoint=[start+5,0],normal=[0,-1],lengthM=10,heightM=5,zBaseM=base)
    groups=sides_of([wall(10,0),wall(11,10)],base_tolerance=.15)
    assert len(groups)==2
    assert sorted(g['zb'] for g in groups)==[10,11]
    groups=sides_of([wall(10,0),wall(10.1,10)],base_tolerance=.15)
    assert len(groups)==1
    assert abs(groups[0]['h']-5.1)<1e-8


def test_camera_terrain_changes_which_upper_wall_is_visible():
    grid=np.full((40,40),-np.inf,dtype=np.float32)
    grid[8:13,:]=9 # intervening podium from x=4 to 6.5 m
    piece=dict(midpoint=[10,5],normal=[-1,0],lengthM=3,heightM=4)
    args=(grid,0,0,[0,5],piece,10,0)
    low=seen_share(*args,0)
    high=seen_share(*args,8)
    assert low<.5
    assert high==1
    assert seen_share(*args,None)==0


def test_common_vertical_datum_shift_does_not_change_visibility():
    grid=np.full((40,40),-np.inf,dtype=np.float32);grid[8:13,:]=9
    piece=dict(midpoint=[10,5],normal=[-1,0],lengthM=3,heightM=4)
    old=seen_share(grid,0,0,[0,5],piece,10,0,0)
    shifted=seen_share(grid+37,0,0,[0,5],piece,10,37,37)
    assert old==shifted
