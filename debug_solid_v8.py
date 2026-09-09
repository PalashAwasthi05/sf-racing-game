import sys
sys.path.insert(0,'work')
from general_world import run,geometry
real=geometry.union_solids
def inspect(meshes):
 bad=[m for m in meshes if not m.is_volume]
 if bad:
  for m in bad:print('BAD_SOLID',{'watertight':m.is_watertight,'winding':m.is_winding_consistent,'volume':m.volume,'faces':len(m.faces),'unique':len(m.edges_unique),'zeroarea':int((m.area_faces<1e-10).sum()),'euler':m.euler_number,'bounds':m.extents.tolist()},flush=True)
  bad[0].export('work/bad-solid-v8.ply')
 return real(meshes)
geometry.union_solids=inspect
run.execute('work/normalized-v8/input.json','work/debug-candidate-v8')
