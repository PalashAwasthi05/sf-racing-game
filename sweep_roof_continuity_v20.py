"""Roof continuity sweep, memory-light: reads only mesh names, positions and indices straight from the GLB (no
trimesh, no textures; ~150 MB peak) and counts roof cap fragments, cap fragments under 4 m2 in plan and box-like
pole parts (both plan extents 0.3-3 m, 3 m or taller) in the massing layer. Holes are read from the v12 roof
coverage step (`unresolvedAreaM2` in the chain log). Usage: sweep_roof_continuity_v20.py <candidate.glb>
Baselines 2026-09-21: v28 18223 fragments / 3395 small / 236 poles; v30 12035 / 1523 / 69."""
import sys,json,struct,numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
path=sys.argv[1];b=open(path,'rb').read()
ln=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+ln]);bin_off=20+ln+8;buf=b[bin_off:]
CT={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32};NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
def acc(i):
    a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];dt=CT[a['componentType']];n=NC[a['type']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0);arr=np.frombuffer(buf,dt,count=a['count']*n,offset=off)
    return arr.reshape(a['count'],n) if n>1 else arr
def components(pos,idx):
    tri=idx.reshape(-1,3);rows=np.concatenate([tri[:,0],tri[:,1],tri[:,2]]);cols=np.concatenate([tri[:,1],tri[:,2],tri[:,0]])
    g=coo_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(len(pos),len(pos)));n,lab=connected_components(g,directed=False);return n,lab
# world matrices: walk the node tree so positions are compared in the scene's z-up frame, not the file's y-up frame
def mat(node):
    if 'matrix' in node:return np.array(node['matrix'],float).reshape(4,4).T
    T=np.eye(4)
    if 'rotation' in node:
        x,y,z,w=node['rotation'];T[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    if 'scale' in node:T[:3,:3]=T[:3,:3]@np.diag(node['scale'])
    if 'translation' in node:T[:3,3]=node['translation']
    return T
parent={c:i for i,n in enumerate(j['nodes']) for c in n.get('children',[])}
def world(i):
    M=mat(j['nodes'][i]);p=parent.get(i)
    return (world(p)@M) if p is not None else M
mesh_world={}
for i,n in enumerate(j['nodes']):
    if 'mesh' in n:mesh_world.setdefault(n['mesh'],world(i))
caps=small=thin=tri_total=0;names=0
for mi,mesh in enumerate(j['meshes']):
    name=mesh.get('name','');layer=name.split('___')[0].split('__')[0]
    if layer not in('roofs','buildings'):continue
    M=mesh_world.get(mi,np.eye(4))
    for prim in mesh['primitives']:
        pos=acc(prim['attributes']['POSITION']).astype(np.float64);pos=pos@M[:3,:3].T+M[:3,3];idx=acc(prim['indices']) if 'indices' in prim else np.arange(len(pos))
        tri_total+=len(idx)//3;names+=1
        n,lab=components(pos,idx)
        for k in range(n):
            v=pos[lab==k];bb=v.max(0)-v.min(0)
            if layer=='roofs':
                caps+=1
                if bb[0]*bb[1]<4 and bb[2]<.5:small+=1     # scene frame is z-up after the node transform
            elif bb[2]>=3 and .3<=min(bb[0],bb[1]) and max(bb[0],bb[1])<=3:thin+=1
print(json.dumps({'file':path.split('/')[-3],'roofAndMassingPrimitives':names,'triangles':tri_total,'roofCapFragments':caps,'capFragmentsUnder4m2':small,'thinTallMassingParts':thin}))
