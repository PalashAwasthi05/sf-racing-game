import numpy as np
import trimesh
from general_world.mesh_hygiene import remove_empty_faces

def test_keeps_real_thin_faces_and_open_boundaries():
 m=trimesh.Trimesh([[0,0,0],[1,0,0],[0,1e-8,0],[2,0,0]],[[0,1,2],[0,1,3]],process=False)
 original=m.vertices.copy();r=remove_empty_faces(m)
 assert r['emptyFacesRemoved']==1 and len(m.faces)==1
 assert np.array_equal(original,m.vertices)
 assert not m.is_watertight
