from pipeline.window_panes import inner_bars
from shapely.geometry import box
def test_casement_not_paired_by_width():
    assert not inner_bars((0,0,1.8,2),{'sash':'casement','lites':[1,1]})
def test_six_panes_from_source_on_casement_and_fixed_bay():
    for sash in ('casement','fixed'):
        bars=inner_bars((0,0,1.8,2),{'sash':sash,'lites':[3,2]})
        assert len(bars)==3
        assert sorted(round(g.centroid.x,3) for g,_ in bars)==[.6,.9,1.2]
def test_meeting_rail_is_not_an_extra_transom_and_single_leaf_stays_single():
    bars=inner_bars((0,0,1,2),{'sash':'double-hung','meetingRail':.6,'lites':[1,2],'transomFractions':[.6]})
    assert sorted(round(p.centroid.y,3) for p,_ in bars)==[.6,1.6]
def test_segment_of_window_clips_original_grid_not_new_one():
    bars=inner_bars((0,0,2,1),{'sash':'casement','lites':[4,1]})
    crop=box(1.1,0,2,1);parts=[p.intersection(crop) for p,_ in bars if p.intersects(crop)]
    assert len(parts)==1 and abs(parts[0].centroid.x-1.5)<1e-8
