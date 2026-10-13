"""Per-building massing comparison of two GLBs, memory-light (positions/indices only). For each building id and
layer (buildings, facades, roofs) reports the top height, and for the massing layer the vertical wall area standing
above the reference build's massing top. Usage: sweep_mass_compare_v23.py <reference.glb> <candidate.glb>"""
import sys,json,struct,collections,numpy as np
CT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32};NC={'SCALAR':1,'VEC3':3}
def load(path):
    b=open(path,'rb').read();ln=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+ln]);buf=b[20+ln+8:]
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];n=NC[a['type']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
        arr=np.frombuffer(buf,CT[a['componentType']],count=a['count']*n,offset=off);return arr.reshape(a['count'],n) if n>1 else arr
    out=collections.defaultdict(list)
    for mesh in j['meshes']:
        name=mesh.get('name','');parts=name.split('__')
        if len(parts)<2:continue
        layer=parts[0].split('___')[0]
        if layer not in('buildings','facades','roofs'):continue
        bid=parts[1].split('___')[0]
        for prim in mesh['primitives']:
            p=acc(prim['attributes']['POSITION']).astype(np.float32);idx=acc(prim['indices']).reshape(-1,3);out[(bid,layer)].append(p[idx])
    return {k:np.concatenate(v) for k,v in out.items()}
ref=load(sys.argv[1]);cand=load(sys.argv[2])
rows=[]
for bid in sorted({k[0] for k in cand}):
    c=cand.get((bid,'buildings'));r=ref.get((bid,'buildings'))
    if c is None:continue
    top_r=float(r[:,:,2].max()) if r is not None else np.nan;top_c=float(c[:,:,2].max())
    n=np.cross(c[:,1]-c[:,0],c[:,2]-c[:,0]);area=np.linalg.norm(n,axis=1)/2;vert=np.abs(n[:,2])<.3*np.linalg.norm(n,axis=1)
    above=c[:,:,2].min(1)>=top_r-.01 if r is not None else np.zeros(len(c),bool)
    rows.append((round(float(area[vert&above].sum())),bid,round(top_r,1),round(top_c,1),round(float(ref[(bid,'facades')][:,:,2].max()),1) if (bid,'facades') in ref else None,round(float(cand[(bid,'facades')][:,:,2].max()),1) if (bid,'facades') in cand else None))
rows.sort(reverse=True)
print('wall area above the reference massing top (m2), id, ref top, cand top, ref facade top, cand facade top')
for r in rows[:14]:print(' ',r)
print('buildings grown by >3 m:',sum(1 for r in rows if r[3]-r[2]>3),'of',len(rows),'| total new wall area above old tops:',sum(r[0] for r in rows))
