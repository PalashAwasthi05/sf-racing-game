import trimesh,numpy as np,json
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
s=trimesh.load('work/coverage-area-v12b/candidate.glb',force='scene',process=False); polys=[];refs=[]
for name,m in s.geometry.items():
 if name.startswith('coverage_repair') or m.metadata.get('layer') not in ('buildings','roofs','facades'):continue
 ids=np.where(m.face_normals[:,2]>.5)[0]
 for i in ids:
  p=Polygon(m.triangles[i,:,:2])
  if p.area>1e-8:polys.append(p);refs.append((m.metadata.get('featureId'),m.triangles_center[i,2]))
tree=STRtree(polys)
for name,m in s.geometry.items():
 if not name.startswith('coverage_repair'):continue
 p=unary_union([Polygon(t[:,:2]) for t in m.triangles[m.face_normals[:,2]>.5]]);ids=tree.query(p,predicate='intersects');sameheight=[polys[i] for i in ids if abs(refs[i][1]-m.bounds[1,2])<.25];above=[polys[i] for i in ids if refs[i][1]>=m.bounds[1,2]-.25];print(name,'area',round(p.area,3),'coplanarOverlap',round(p.intersection(unary_union(sameheight)).area,3),'obscuredAbove',round(p.intersection(unary_union(above)).area,3))
