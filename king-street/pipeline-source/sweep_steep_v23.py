"""Steep roof faces, memory-light: per building, area of massing/roof triangles that are neither walls nor near-flat
(slope 25-80 degrees) and their vertical extent. Usage: sweep_steep_v23.py <glb> [<glb> ...]"""
import sys,json,struct,collections,numpy as np
CT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32};NC={'SCALAR':1,'VEC3':3}
for path in sys.argv[1:]:
    b=open(path,'rb').read();ln=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+ln]);buf=b[20+ln+8:]
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];n=NC[a['type']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
        arr=np.frombuffer(buf,CT[a['componentType']],count=a['count']*n,offset=off);return arr.reshape(a['count'],n) if n>1 else arr
    res=collections.defaultdict(lambda:[0.,1e9,-1e9,None])
    for mesh in j['meshes']:
        name=mesh.get('name','');parts=name.split('__')
        if len(parts)<2 or parts[0].split('___')[0] not in('buildings','roofs') or 'observed' in name:continue
        bid=parts[1].split('___')[0]
        for prim in mesh['primitives']:
            t=acc(prim['attributes']['POSITION'])[acc(prim['indices']).reshape(-1,3)].astype(np.float64)
            n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);L=np.linalg.norm(n,axis=1);ok=L>1e-6;cz=np.abs(n[ok,2])/L[ok];t=t[ok];A=L[ok]/2
            steep=(cz<np.cos(np.radians(25)))&(cz>np.cos(np.radians(80)))
            if steep.any():
                r=res[bid];r[0]+=A[steep].sum();r[1]=min(r[1],t[steep][:,:,2].min());r[2]=max(r[2],t[steep][:,:,2].max());r[3]=t[steep][:,:,:2].reshape(-1,2).mean(0).round(1).tolist()
    rows=sorted(((round(v[0]),k,round(v[2]-v[1],1),round(v[2],1),v[3]) for k,v in res.items()),reverse=True)
    print(path.split('/')[0],'steep roof area total',sum(r[0] for r in rows),'| worst (area, id, vertical extent, top z, centre):')
    for r in rows[:6]:print('  ',r)
