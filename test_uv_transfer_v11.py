import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial
from general_world.texture_selection import metric_uv
from general_world.uv_transfer import infer_repeat_m,apply_source_material

def test_metric_repeat_survives_clipping():
 m=trimesh.creation.box([3,4,5]);m.unmerge_vertices();m.visual=TextureVisuals(uv=metric_uv(m,[2,3]),material=PBRMaterial())
 assert np.allclose(infer_repeat_m(m),[2,3])
 n=apply_source_material(trimesh.creation.box([1,2,3]),m)
 assert np.allclose(infer_repeat_m(n),[2,3])
 assert n.visual.material is m.visual.material
