import numpy as np
import trimesh

from regen.check_roof_sky_projection import compare, raster


def projection():
    return {'cameraXYZ':[0,0,0], 'basisRightDownForward':np.eye(3).tolist(),
        'focalPx':50, 'xRangePx':[10,90],
        'facadeCornersPx':[[10,80],[10,30],[90,80],[90,30]]}


def test_raster_nearest_triangle_wins_independent_of_face_order():
    vertices=np.array([[-1,-1,5],[1,-1,5],[0,1,5],[-1,-1,4],[1,-1,4],[0,1,4]])
    for faces in ([[0,1,2],[3,4,5]],[[3,4,5],[0,1,2]]):
        depth,face=raster(trimesh.Trimesh(vertices=vertices,faces=faces,process=False),projection(),(100,100))
        assert abs(depth[50,50]-4)<1e-5
        assert face[50,50]>=0


def test_sky_margin_and_known_foreground_occluder():
    groups={'sky':np.zeros((100,100),np.float32),'structure':np.zeros((100,100),np.float32),'occluder':np.zeros((100,100),np.float32)}
    groups['sky'][:30]=1;groups['structure'][30:80]=1
    depth=np.full((100,100),np.inf,np.float32);depth[20:30,30:70]=5
    face=np.full((100,100),-1,np.int32);face[20:30,30:70]=0
    other=np.full_like(depth,np.inf)
    conflict,_,metrics=compare(groups,depth,face,other,projection(),5)
    assert conflict[20,40] and not conflict[28,40]
    assert metrics['conflictPixels']>0
    other[20:30,30:70]=4
    conflict,_,metrics=compare(groups,depth,face,other,projection(),5)
    assert not conflict.any()
    assert metrics['skyConflictPixelsBeforeKnownOcclusion']>0


def test_distant_background_does_not_supply_target_wall_support():
    groups={'sky':np.ones((100,100),np.float32),'structure':np.zeros((100,100),np.float32),'occluder':np.zeros((100,100),np.float32)}
    groups['sky'][70:]=0;groups['structure'][70:]=1
    depth=np.full((100,100),np.inf,np.float32);depth[20:30,30:70]=5
    face=np.zeros((100,100),np.int32)
    conflict,_,metrics=compare(groups,depth,face,np.full_like(depth,np.inf),projection(),5)
    assert not conflict.any()
    assert metrics['skyPixelsAbstainedForNoUpperFacadeSupport']>0
