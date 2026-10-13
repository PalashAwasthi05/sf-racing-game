"""Blank towers, memory-light: connected massing parts (buildings layer, not observed skins) at least 6 m tall whose
top stands at least 4 m above the highest glazing of the same building within 3 m of the part in plan. These read as
windowless towers. Usage: sweep_blank_towers_v23.py <glb> [<glb> ...]"""
import sys,json,struct,collections,numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
CT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32};NC={'SCALAR':1,'VEC3':3}
for path in sys.argv[1:]:
    b=open(path,'rb').read();ln=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+ln]);buf=b[20+ln+8:]
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];n=NC[a['type']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
        arr=np.frombuffer(buf,CT[a['componentType']],count=a['count']*n,offset=off);return arr.reshape(a['count'],n) if n>1 else arr
    parts=collections.defaultdict(list);glass=collections.defaultdict(list)
    for mesh in j['meshes']:
        name=mesh.get('name','');pp=name.split('__')
        if len(pp)<2:continue
        layer=pp[0].split('___')[0];bid=pp[1].split('___')[0];cls=name.split('___')[-1].split('#')[0]
        for prim in mesh['primitives']:
            pos=acc(prim['attributes']['POSITION']).astype(np.float64);idx=acc(prim['indices'])
            if layer=='facades' and cls.startswith(('glass','curtain')):glass[bid].append(pos[::3])
            elif layer=='buildings' and 'observed' not in name:
                tri=idx.reshape(-1,3);r=np.r_[tri[:,0],tri[:,1],tri[:,2]];c=np.r_[tri[:,1],tri[:,2],tri[:,0]]
                n,lab=connected_components(coo_matrix((np.ones(len(r),np.int8),(r,c)),shape=(len(pos),len(pos))),directed=False)
                for k in range(n):
                    v=pos[lab==k]
                    if v[:,2].max()-v[:,2].min()>=6:parts[bid].append(v)
    towers=[];area=0
    for bid,vs in parts.items():
        g=np.concatenate(glass[bid]) if glass.get(bid) else np.zeros((0,3));tree=cKDTree(g[:,:2]) if len(g) else None
        for v in vs:
            top=v[:,2].max();near=[]
            if tree is not None:
                for q in tree.query_ball_point(v[:,:2],3.0):near.extend(q)
            gtop=g[list(set(near)),2].max() if near else v[:,2].min()
            if top-gtop>=4:
                ext=v[:,:2].max(0)-v[:,:2].min(0);towers.append((round(top-gtop,1),bid,round(top,1),round(gtop,1),ext.round(1).tolist(),v[:,:2].mean(0).round(1).tolist()))
    towers.sort(reverse=True)
    print(path.split('/')[0],'blank tower parts:',len(towers),'in',len({t[1] for t in towers}),'buildings')
    for t in towers[:10]:print('  blank height',t[0],t[1],'top',t[2],'glazing top',t[3],'plan',t[4],'at',t[5])
