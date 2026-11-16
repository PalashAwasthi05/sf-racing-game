import numpy as np
import pytest
from shapely.geometry import Polygon
from pipeline.facade_grammar import compile_wall


def fixture():
    meshes=[]
    ctx={'polys':lambda g:[g], 'put':lambda mesh,key,material:meshes.append((mesh,key,material)),
         'cube':lambda *args:None}
    building={'id':'synthetic-host','g':Polygon([(0,0),(10,0),(10,10),(7,10),(7,11),(3,11),(3,10),(0,10)])}
    profile={'id':'observed-front','normal':[0,1],'planeV':10.,'frameU':[-10.,0.],
             'heightRefM':5.,'material':'host','sourceIds':['source-photo'],
             'bays':[{'rect':[.4,.2,.6,.9],'projectionM':.7,'shape':'square','cladding':'bay','cap':'none'}],
             'openings':[{'rect':[.45,.3,.55,.8], 'style':{'kind':'window','frameMaterial':'frame'}}]}
    return ctx,building,profile,meshes


def test_bay_emits_once_even_when_host_edge_has_no_bay_center():
    ctx,building,profile,meshes=fixture()
    # The true host edge ends before the bay. A footprint already contains
    # the projecting front, so no host edge passes beneath the bay centre.
    compile_wall(ctx,building,np.array([10.,10.]),np.array([7.,10.]),0.,5.,profile,[])
    first=[mesh for mesh,_,material in meshes if material=='bay']
    assert first
    bay_count=len(first)
    compile_wall(ctx,building,np.array([7.,11.]),np.array([3.,11.]),0.,5.,profile,[])
    all_bay=[mesh for mesh,_,material in meshes if material=='bay']
    assert len(all_bay)==bay_count
    # Depth is measured from y=10, not added to the mapped y=11 projection.
    assert max(mesh.bounds[1,1] for mesh in all_bay)==pytest.approx(10.7,abs=.01)


def test_mapped_front_cannot_cover_bay_glazing_with_an_opaque_host_sheet():
    ctx,building,profile,meshes=fixture()
    compile_wall(ctx,building,np.array([7.,11.]),np.array([3.,11.]),0.,5.,profile,[])
    # Every host-wall triangle intersecting the middle of the bay elevation
    # must be behind the measured bay, not at the old outer footprint face.
    for mesh,_,material in meshes:
        if material!='host':continue
        for tri in mesh.triangles:
            projected=Polygon(tri[:,[0,2]])
            from shapely.geometry import Point
            if projected.is_valid and projected.area>1e-8 and projected.covers(Point(5.,2.5)):
                assert np.max(tri[:,1])<=10.71
