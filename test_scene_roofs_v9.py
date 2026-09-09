import numpy as np
import trimesh
from shapely.geometry import box
from trimesh.visual.texture import TextureVisuals
from general_world.scene_roofs import clip_under_roof,repair_scene

def test_clipping_preserves_uv_and_shared_roof_line():
    m=trimesh.Trimesh([[0,0,0],[10,0,0],[10,0,12],[0,0,12]],[[0,1,2],[0,2,3]],process=False)
    m.visual=TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],material=trimesh.visual.material.PBRMaterial())
    clipped,r=clip_under_roof(m,box(0,-1,10,1),{'planes':[[.2,0,8]],'operation':'min'})
    assert r['clippedTriangles']==2
    assert np.all(clipped.vertices[:,2]<=8+.2*clipped.vertices[:,0]+1e-8)
    assert np.allclose(clipped.visual.uv[:,0],clipped.vertices[:,0]/10)
    assert np.allclose(clipped.visual.uv[:,1],clipped.vertices[:,2]/12)

def test_roof_repair_is_uniform_and_does_not_touch_passage_geometry():
    s=trimesh.Scene();roof=trimesh.creation.box([20,14,.1]);roof.apply_translation([10,7,12]);roof.metadata={'layer':'roofs','featureId':'opaque-key'}
    roof.visual=TextureVisuals(uv=np.zeros((len(roof.vertices),2)),material=trimesh.visual.material.PBRMaterial())
    s.add_geometry(roof,geom_name='roof',node_name='roof')
    x,y=np.meshgrid(np.arange(.1,20,.3),np.arange(.1,14,.3));p=np.c_[x.ravel(),y.ravel(),10+.2*x.ravel()]
    r=repair_scene(s,p)
    assert r['capsReplaced']==1 and r['buildingsChanged']==1
    assert all(m.metadata.get('layer')=='roofs' for m in s.geometry.values())
    assert all(rec['replacementWatertight'] for rec in r['records'] if rec['applied'])

def test_wall_crossing_cap_is_split_without_closing_unowned_space():
    m=trimesh.Trimesh([[0,0,0],[10,0,0],[10,0,12],[0,0,12]],[[0,1,2],[0,2,3]],process=False)
    m.visual=TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],material=trimesh.visual.material.PBRMaterial())
    clipped,r=clip_under_roof(m,box(4,-1,6,1),{'planes':[[0,0,8]],'operation':'min'})
    assert r['ambiguousTriangles']==0
    assert abs(clipped.area-(120-2.004*4))<1e-5
    assert clipped.bounds[1,2]==12
    assert np.allclose(clipped.visual.uv[:,0],clipped.vertices[:,0]/10)

if __name__=='__main__':
    test_clipping_preserves_uv_and_shared_roof_line();test_roof_repair_is_uniform_and_does_not_touch_passage_geometry();print('2 scene roof tests passed')
