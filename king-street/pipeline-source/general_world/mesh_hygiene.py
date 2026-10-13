"""Remove mathematically empty faces without filling openings or changing silhouettes."""
import numpy as np

def remove_empty_faces(mesh):
 count=len(mesh.faces);areas=mesh.area_faces
 if not np.isfinite(mesh.vertices).all() or not np.isfinite(areas).all():raise ValueError('nonfinite geometry')
 keep=areas>0
 if not keep.all():mesh.update_faces(keep)
 return {'emptyFacesRemoved':int(count-len(mesh.faces)),'filledHoles':0,'movedVertices':0}
