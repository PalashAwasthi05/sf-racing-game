import cv2
import numpy as np
from refine_paired_openings import paired_panes


def wall():
    rng=np.random.default_rng(4)
    a=np.clip(rng.normal(174,12,(185,290,1)),0,255).astype(np.uint8)
    a=np.repeat(a,3,axis=-1)
    for row in range(3):
        for col in range(5):
            x=14+col*53;y=15+row*55
            cv2.rectangle(a,(x,y),(x+22,y+29),(66,70,76),-1)
    return a


def test_two_views_and_occlusion():
    a=wall()
    b=cv2.warpAffine(a,np.float32([[1,0,3],[0,1,-2]]),(a.shape[1],a.shape[0]))
    cv2.rectangle(b,(70,63),(101,108),(173,172,168),-1)
    panes=paired_panes(a,b)
    assert panes and len(panes)==14
    assert not any(x<75<r and y<85<t for x,y,r,t in panes)


def test_no_windows_without_registered_support():
    assert paired_panes(wall(),np.full_like(wall(),170)) is None


def test_dividers_require_two_sided_evidence_in_both_photos():
    from refine_paired_openings import paired_style
    a=np.full((30,46),60,dtype=np.uint8)
    a[:,22:24]=170
    b=a.copy();a[7:9,:]=170
    style=paired_style(a,b)
    assert abs(style['mullionFractions'][0]-.5)<.03
    assert 'transomFractions' not in style


def test_whole_hole_owns_nested_sash_without_dropping_neighbor():
    from refine_paired_openings import disjoint_observed_openings
    outer={'rect':[.1,.2,.2,.4]};neighbor={'rect':[.201,.2,.31,.40]}
    rects=[outer,{'rect':[.102,.21,.199,.26]},neighbor,{'rect':[.101,.202,.21,.37]}]
    assert disjoint_observed_openings(rects)==[neighbor,outer]
