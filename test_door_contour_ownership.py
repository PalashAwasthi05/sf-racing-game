from shapely.geometry import Polygon,box
from shapely.ops import unary_union
import pytest
from pipeline.door_contours import draw_observed_garage
from pipeline.door_leaves import draw_garage_leaf


def draw(original,segment,style):
    pieces=[]
    draw_observed_garage(draw_garage_leaf,lambda p,z,d,m:pieces.append((p,z,d,m)),original,segment.intersection(original),style,.5,'wall',lambda x,d:0)
    return pieces


def test_oblique_head_and_split_preserve_only_original_casing_and_panels():
    outline=Polygon([(0,0),(4,0),(4,1.7),(3,2.4),(2,2.8),(1,2.4),(0,1.7)])
    st={'doorKind':'panel','material':'leaf','frameMaterial':'frame','doorCells':[{'kind':'panel','rect':[.50,.15,.80,.76]}]}
    full=draw(outline,outline,st)
    parts=draw(outline,box(-1,-1,2.5,4),st)+draw(outline,box(2.5,-1,5,4),st)
    for region,_,_,_ in full+parts:assert region.difference(outline).area<1e-8
    for _,z,d,m in full:
        def layer(q):return unary_union([p for p,h,w,k in q if (h,w,k)==(z,d,m)])
        assert layer(full).symmetric_difference(layer(parts)).area<1e-7
    assert not any(p.contains(box(2.45,.2,2.55,1.4)) for p,h,d,m in parts if m=='frame')


def test_open_garage_has_no_leaf_or_photo_marks():
    o=box(0,0,3,3);assert not any(m=='leaf' for p,h,d,m in draw(o,o,{'doorKind':'open','material':'leaf'}))

def test_three_correlated_runs_need_spatially_unique_registration():
    import numpy as np
    from regen.compile_garage_marks import agree_runs
    rng=np.random.default_rng(3);a=rng.normal(0,.6,(4,100))
    for row in a:
        for x,y in [(7,22),(37,64),(73,94)]:row[x:y]+=rng.uniform(8,20)
    assert len(agree_runs(a,np.roll(a,5,axis=1),45))>=8
    assert not agree_runs(a,rng.normal(0,2,a.shape),45)
