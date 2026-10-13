"""Constrained, densified terrain surfaces preserve holes and positive winding."""
import numpy as np,trimesh
from shapely import segmentize
from .geometry import polygons

def drape(g,terrain,offset=.03):
 parts=[]
 for p in polygons(g):
  if p.area<.001:continue
  p=segmentize(p,.75);v,f=trimesh.creation.triangulate_polygon(p,engine='earcut')
  m=trimesh.Trimesh(np.c_[v,terrain(v[:,0],v[:,1])+offset],f,process=False)
  if np.median(m.face_normals[:,2])<0:m.invert()
  parts.append(m)
 return trimesh.util.concatenate(parts) if parts else None
