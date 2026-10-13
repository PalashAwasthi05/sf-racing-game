"""Coplanar overlap sweep (z-fighting), memory-light: reads positions/indices straight from the GLB, groups the
building, facade and roof triangles by plane (normal to .02, offset within TOL (default 3 cm)), and measures the area where
triangles of two different meshes overlap in the same plane and face the same way. Usage: sweep_coplanar_v23.py <glb>"""
import sys,json,struct,collections,numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
path=sys.argv[1];b=open(path,'rb').read();ln=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+ln]);buf=b[20+ln+8:]
CT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32};NC={'SCALAR':1,'VEC3':3}
def acc(i):
    a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];n=NC[a['type']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    arr=np.frombuffer(buf,CT[a['componentType']],count=a['count']*n,offset=off);return arr.reshape(a['count'],n) if n>1 else arr
TOL=float(sys.argv[2]) if len(sys.argv)>2 else .03
groups=collections.defaultdict(list)
for mi,mesh in enumerate(j['meshes']):
    name=mesh.get('name','');layer=name.split('___')[0].split('__')[0]
    if layer not in('buildings','facades','roofs'):continue
    for prim in mesh['primitives']:
        p=acc(prim['attributes']['POSITION']).astype(np.float64);idx=acc(prim['indices']).reshape(-1,3);t=p[idx]
        n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);a=np.linalg.norm(n,axis=1)/2;ok=a>.05;t,n,a=t[ok],n[ok]/(2*a[ok,None]),a[ok]
        d=np.einsum('ij,ij->i',n,t[:,0])
        keys=np.c_[np.round(n/.02).astype(int),np.floor(d/TOL).astype(int)]
        for k,tri,nn in zip(map(tuple,keys),t,n):groups[k].append((mi,tri,nn))
def basis(nn):
    u=np.cross(nn,[0,0,1] if abs(nn[2])<.9 else [1,0,0]);u/=np.linalg.norm(u);return u,np.cross(nn,u)
pairs=collections.Counter();examples={}
for k,own in list(groups.items()):
    items=own+groups.get(k[:-1]+(k[-1]+1,),[])   # the next offset bin too: surfaces up to TOL apart
    meshes={m for m,_,_ in items}
    if len(meshes)<2:continue
    u,v=basis(items[0][2]);per=collections.defaultdict(list);orient='wall' if abs(items[0][2][2])<.3 else 'roof'   # the generator writes z-up positions
    for m,tri,_ in items:per[m].append(Polygon(np.c_[tri@u,tri@v]).buffer(0))
    polys={m:unary_union(ps) for m,ps in per.items()};ms=sorted(polys)
    for x in range(len(ms)):
        for y in range(x+1,len(ms)):
            ov=polys[ms[x]].intersection(polys[ms[y]]).area
            if ov>.05:
                na=j['meshes'][ms[x]]['name'];nb=j['meshes'][ms[y]]['name']
                key=tuple(sorted([na.split('___')[0].split('__')[0]+('/obs' if 'observed' in na else '')+':'+na.split('___')[-1].split('#')[0],nb.split('___')[0].split('__')[0]+('/obs' if 'observed' in nb else '')+':'+nb.split('___')[-1].split('#')[0]]))
                key=(orient,)+key;pairs[key]+=ov
                if key not in examples or ov>examples[key][0]:examples[key]=(round(ov,1),na[:70],nb[:70])
tot=sum(pairs.values())
print(json.dumps({'file':path,'coplanarOverlapM2':round(tot)}))
print('walls',round(sum(v for k,v in pairs.items() if k[0]=='wall')),'roofs',round(sum(v for k,v in pairs.items() if k[0]=='roof')))
for o in ('wall','roof'):
    print('--',o)
    for k,v in [kv for kv in pairs.most_common() if kv[0][0]==o][:12]:print(round(v),k[1:],examples[k])
