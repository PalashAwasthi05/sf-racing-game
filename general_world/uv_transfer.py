"""Preserve physical texture repeat for newly clipped surfaces."""
import numpy as np
from trimesh.visual import TextureVisuals
from .texture_selection import metric_uv

def infer_repeat_m(mesh):
 uv=getattr(mesh.visual,'uv',None)
 if uv is None:return None
 f=mesh.faces;delta_uv=uv[f[:,1:]]-uv[f[:,0,None]]
 det=np.linalg.det(delta_uv);good=np.abs(det)>1e-10
 if not good.any():return None
 edges=mesh.vertices[f[good,1:]]-mesh.vertices[f[good,0,None]]
 derivatives=np.linalg.inv(delta_uv[good])@edges
 lengths=np.linalg.norm(derivatives,axis=2)
 valid=np.isfinite(lengths).all(1)&(lengths.min(1)>1e-5)
 return np.median(lengths[valid],axis=0) if valid.any() else None

def apply_source_material(replacement,source):
 scale=infer_repeat_m(source)
 if scale is None:
  replacement.visual=TextureVisuals(material=source.visual.material)
 else:
  replacement.unmerge_vertices()
  replacement.visual=TextureVisuals(uv=metric_uv(replacement,scale),material=source.visual.material)
 replacement.metadata=dict(source.metadata)
 return replacement
