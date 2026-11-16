"""Reindex display surfaces by full vertex attributes and simplify dense connected shells."""
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np


def compact_surfaces(model, binary, added):
    """Keep the archive, scene identity, metre coordinates, material boundaries and open edges."""
    types = {5121: np.uint8, 5123: np.dtype('<u2'), 5125: np.dtype('<u4'), 5126: np.dtype('<f4')}
    widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    def load(i):
        a = model['accessors'][i]
        v = model['bufferViews'][a['bufferView']]
        local = a['bufferView'] in added
        data = added[a['bufferView']] if local else binary
        offset = a.get('byteOffset', 0) + (0 if local else v.get('byteOffset', 0))
        return np.frombuffer(data, dtype=types[a['componentType']], count=a['count']*widths[a['type']],
                             offset=offset).reshape(a['count'], -1)

    def store(array, template, component=None):
        raw = np.ascontiguousarray(array).tobytes()
        k = len(model['bufferViews'])
        model['bufferViews'].append({'buffer': 0, 'byteLength': len(raw)})
        added[k] = raw
        updated = dict(template, bufferView=k, count=len(array), byteOffset=0)
        if component is not None:
            updated['componentType'] = component
        if 'min' in updated:
            updated['min'] = array.reshape(len(array), -1).min(axis=0).tolist()
        if 'max' in updated:
            updated['max'] = array.reshape(len(array), -1).max(axis=0).tolist()
        model['accessors'].append(updated)
        return len(model['accessors'])-1

    stats = [0, 0, 0, 0]
    process = None
    node = shutil.which('node')
    surface_layers = {'roofs', 'buildings', 'terrain', 'sidewalks', 'roads', 'foundations', 'landscape'}
    try:
        for mesh in model['meshes']:
            layer = mesh.get('name', '').split('___')[0].split('__')[0]
            if layer not in surface_layers:
                continue
            for p in mesh['primitives']:
                attrs = p['attributes']
                if p.get('mode', 4) != 4 or 'indices' not in p or set(attrs) - {'POSITION','NORMAL','TEXCOORD_0','COLOR_0'}:
                    continue
                records = [model['accessors'][i] for i in [*attrs.values(), p['indices']]]
                if any(a.get('normalized') or 'byteStride' in model['bufferViews'][a['bufferView']] or
                       a['componentType'] not in types or a['type'] not in widths for a in records):
                    continue
                old = load(p['indices']).ravel()
                if len(old) < 192 or len(old) % 3:
                    continue
                arrays = {k: load(i) for k, i in attrs.items()}
                pos = arrays['POSITION']
                if pos.dtype != np.float32 or any(len(a)!=len(pos) for a in arrays.values()):
                    continue
                # A position alone isn't an identity. Preserve UV boundaries, colours,
                # and sharp edges even when the source relies on computed normals.
                normal = arrays.get('NORMAL')
                if normal is None:
                    faces = old.reshape(-1,3)
                    weighted = np.cross(pos[faces[:,1]]-pos[faces[:,0]], pos[faces[:,2]]-pos[faces[:,0]])
                    normal = np.zeros_like(pos)
                    for corner in faces.T:
                        np.add.at(normal, corner, weighted)
                    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True),1.e-20)
                attributes = [normal, *[v for k,v in arrays.items() if k not in ('POSITION','NORMAL')]]
                shading = np.column_stack(attributes).astype('<f4',copy=False)
                keys = np.ascontiguousarray(np.column_stack((pos, shading)))
                keys[keys==0] = 0  # +0 and -0 should compare equal
                rows = keys.view(np.dtype((np.void,keys.strides[0]))).ravel()
                _, uniq, inv = np.unique(rows,return_index=True,return_inverse=True)
                mapped = inv.ravel()[old].astype('<u4')
                if node and len(mapped)>=384:
                    if process is None:
                        process = subprocess.Popen([node, str(Path(__file__).with_name('simplify_surface.mjs'))],
                                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE)
                    process.stdin.write(struct.pack('<III',len(uniq),len(mapped),shading.shape[1]))
                    process.stdin.write(np.ascontiguousarray(pos[uniq]).tobytes())
                    process.stdin.write(np.ascontiguousarray(shading[uniq]).tobytes())
                    process.stdin.write(mapped.tobytes());process.stdin.flush()
                    (count,) = struct.unpack('<I',process.stdout.read(4))
                    if count > len(mapped) or count % 3:
                        raise ValueError('Invalid simplified surface')
                    raw = process.stdout.read(count*4)
                    if len(raw)!=count*4:
                        raise ValueError('Incomplete simplified surface')
                    if count:
                        mapped=np.frombuffer(raw,dtype='<u4')
                used, compact = np.unique(mapped, return_inverse=True)
                if len(used)>=len(pos) and len(mapped)>=len(old) and old.dtype.itemsize<=2:
                    continue
                ids=uniq[used]
                for k,v in arrays.items():
                    attrs[k]=store(v[ids],model['accessors'][attrs[k]])
                small=len(used)<=65535
                index=compact.ravel().astype('<u2' if small else '<u4')
                p['indices']=store(index,model['accessors'][p['indices']],5123 if small else 5125)
                stats = [a+b for a,b in zip(stats,(len(pos),len(ids),len(old)//3,len(index)//3))]
    finally:
        if process is not None:
            process.stdin.close()
            if process.wait():
                raise RuntimeError('Surface simplifier failed')
    print('Optimized surfaces (vertices before/after, triangles before/after):', *stats,flush=True)
