from shapely.geometry import Point, box
from pipeline.street_space import place


def plan():
    return dict(forbidden=box(-20,-4,20,4).union(box(5,4,7,12)),
                sidewalk=box(-20,4,20,8).difference(box(5,4,7,12)),
                occupied=[],records=[])


def test_preserves_a_supported_mapped_location():
    p=plan()
    q=place(p,(0,6),.35,box(-30,-30,30,30),kind='bin',source_id='a',require_walk=True)
    assert tuple(q)==(0.,6.)
    assert p['records'][-1]['status']=='mapped'


def test_road_control_node_is_not_rendered_as_a_mapped_pole():
    p=plan()
    assert place(p,(0,0),.1,box(-30,-30,30,30),kind='lamp',source_id='a',max_shift=1.5) is None
    q=place(p,(0,0),.14,box(-30,-30,30,30),kind='signal',source_id='b',max_shift=12,require_walk=True)
    assert q is not None
    assert Point(q).buffer(.14).intersection(p['forbidden']).area<1e-6
    assert p['records'][-1]['status']=='inferred-bounded-shift'


def test_excludes_crossings_buildings_and_other_furniture():
    p=plan();bounds=box(-30,-30,30,30)
    assert place(p,(6,6),.1,bounds,kind='tree',source_id='a') is None
    q=place(p,(0,6),.35,bounds,kind='bin',source_id='b',require_walk=True)
    assert q is not None
    assert place(p,(0,6),.2,bounds,kind='tree',source_id='c') is None


def test_bin_cannot_be_placed_on_unmapped_open_ground():
    p=plan()
    assert place(p,(0,18),.35,box(-30,-30,30,30),kind='bin',source_id='a',max_shift=1.5,require_walk=True) is None


def test_large_can_cannot_fit_in_a_narrow_sidewalk_sliver():
    p=plan();p['sidewalk']=box(-5,4,5,4.5)
    assert place(p,(0,4.25),.36,box(-30,-30,30,30),kind='bin',source_id='a',max_shift=2,require_walk=True) is None
