import numpy as np
import trimesh
from shapely.geometry import box
from pipeline.crown_clearance import CrownClearance

def test_low_cards_trimmed_as_whole_high_overhang_and_yard_untouched():
    c=CrownClearance({'road':box(0,0,10,20)},lambda x,y:.1*x)
    points=np.array([[-2.,1,2.],[-.2,1,1.],[5,10,4.],[5,10,8.]])
    assert c.allows(points).tolist()==[True,False,False,True]
    fans=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]]);verts=[p+np.array([[.1,0,0],[0,.1,0],[-.1,0,0],[0,-.1,0],[0,0,.02]]) for p in points]
    mesh=trimesh.Trimesh(np.vstack(verts),np.vstack([fans+5*i for i in range(4)]),process=False);out=c.leaves(mesh)
    assert len(out.faces)==8 and c.card_count==2

def test_low_branch_clips_at_curb_on_slope_vertical_trunk_preserved():
    c=CrownClearance({'road':box(0,0,10,20)},lambda x,y:x*.3)
    assert c.limb([-3,6,2],[3,6,3])[0]<=0
    assert np.allclose(c.limb([2,6,0],[2,6,5]),[2,6,5])
    assert np.allclose(c.limb([-3,6,8],[3,6,10]),[3,6,10])
