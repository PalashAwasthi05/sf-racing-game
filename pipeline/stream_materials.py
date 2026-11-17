"""Apply material decisions to static GLBs without expanding city geometry into Python objects."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'king-street/pipeline-source'))
from package_route_block import Source
from pipeline.merge_static_glbs import remap_material


def append_data(model, binary, data):
    size = len(data)
    i = len(model['bufferViews'])
    model['bufferViews'].append({'buffer': 0, 'byteOffset': binary.tell(), 'byteLength': size})
    binary.write(data)
    binary.write(bytes(-size % 4))
    return i


def save(model, binary, path):
    size = binary.tell()
    model['buffers'] = [{'byteLength': size}]
    text = json.dumps(model, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    text += b' ' * (-len(text) % 4)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        tmp = Path(out.name)
        try:
            out.write(struct.pack('<5I',0x46546c67,2,28+size+len(text),len(text),0x4e4f534a))
            out.write(text);out.write(struct.pack('<II',size,0x004e4942))
            binary.seek(0);shutil.copyfileobj(binary,out,1<<20)
            out.close();tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def copy_image(src, image, model, binary):
    im = dict(image)
    offset, size = src.view_range(im['bufferView'])
    src.stream.seek(src.offset+offset)
    im['bufferView'] = append_data(model,binary,src.stream.read(size))
    return im


def palette(source_path, path):
    """One UV-mapped triangle per original material so the normal selector sees its true maps."""
    with source_path.open('rb') as stream, tempfile.TemporaryFile() as binary:
        src = Source(stream)
        m = src.model
        model = {k:copy.deepcopy(m[k]) for k in ('asset','materials','textures','samplers','extensionsUsed','extensionsRequired') if k in m}
        model.update(bufferViews=[], meshes=[], nodes=[], scene=0)
        model['images']=[copy_image(src,im,model,binary) for im in m.get('images',[])]
        pos=append_data(model,binary,np.array([[0,0,0],[1,0,0],[0,1,0]],'<f4').tobytes())
        uv=append_data(model,binary,np.array([[0,0],[1,0],[0,1]],'<f4').tobytes())
        model['accessors']=[{'bufferView':pos,'componentType':5126,'count':3,'type':'VEC3','min':[0,0,0],'max':[1,1,0]},
                            {'bufferView':uv,'componentType':5126,'count':3,'type':'VEC2'}]
        for i in range(len(model['materials'])):
            name=f'material-{i}'
            model['meshes'].append({'name':name,'primitives':[{'material':i,'attributes':{'POSITION':0,'TEXCOORD_0':1}}]})
            model['nodes'].append({'name':name,'mesh':i})
        model['scenes']=[{'nodes':list(range(len(model['nodes'])))}]
        save(model,binary,path)


def apply(source_path, palette_path, destination, decisions):
    """Copy source BIN exactly, then append selected scanned maps and metric UVs."""
    with source_path.open('rb') as original, palette_path.open('rb') as colors, tempfile.TemporaryFile(dir=destination.parent) as binary:
        src, pal = Source(original), Source(colors)
        m, painted = src.model, pal.model
        original.seek(src.offset)
        shutil.copyfileobj(original,binary,1<<20)
        old_size=binary.tell(); original_count=len(m['materials'])
        image_index=[len(m.get('images',[]))+i for i in range(len(painted.get('images',[])))]
        for im in painted.get('images',[]):m.setdefault('images',[]).append(copy_image(pal,im,m,binary))
        sampler_offset=len(m.get('samplers',[]));m.setdefault('samplers',[]).extend(painted.get('samplers',[]))
        tex_index=[]
        for texture in painted.get('textures',[]):
            texture['source']=image_index[texture['source']]
            if 'sampler' in texture:texture['sampler']+=sampler_offset
            tex_index.append(len(m['textures']));m['textures'].append(texture)
        byname={mat['name']:mat for mat in painted['materials']}
        replacement={}
        for i, mat in enumerate(m['materials'][:original_count]):
            name=mat['name'];d=decisions.get(name)
            if not d or d['action'] not in ('recolor','reuse'):continue
            new=copy.deepcopy(byname['auto_'+name+'_'+d['asset_id']]);remap_material(new,tex_index)
            replacement[i]=(len(m['materials']),d);m['materials'].append(new)
        uv_cache={};updated=0;missing_uv=0
        for mesh in m['meshes']:
            for prim in mesh['primitives']:
                old=prim.get('material')
                if old not in replacement:continue
                ref=prim['attributes'].get('TEXCOORD_0')
                if ref is None:missing_uv+=1;continue
                new,d=replacement[old];pair=(ref,old)
                if pair not in uv_cache:
                    coords=src.array(ref).astype('float64')
                    coords[:,1]=1-coords[:,1]
                    coords*=np.array(d['target']['old_scale'])
                    if d.get('rotate90'):coords=coords[:,::-1]*[1.,-1.]
                    coords/=np.array(d['tile_size_m']);coords[:,1]=1-coords[:,1]
                    if not np.isfinite(coords).all():raise ValueError('Nonfinite material coordinates')
                    coords=coords.astype('<f4')
                    vi=append_data(m,binary,coords.tobytes())
                    uv_cache[pair]=len(m['accessors'])
                    m['accessors'].append({'bufferView':vi,'componentType':5126,'count':len(coords),'type':'VEC2',
                                           'min':coords.min(0).tolist(),'max':coords.max(0).tolist()})
                prim['attributes']['TEXCOORD_0']=uv_cache[pair];prim['material']=new;updated+=1
        if not m.get('samplers'):m.pop('samplers',None)
        save(m,binary,destination)
        # Verify the entire original geometry payload survived the disk handoff byte-for-byte.
        original.seek(src.offset)
        with destination.open('rb') as check:
            dst=Source(check);check.seek(dst.offset);expected=hashlib.sha256();actual=hashlib.sha256()
            for offset in range(0,old_size,1<<20):
                n=min(1<<20,old_size-offset);expected.update(original.read(n));actual.update(check.read(n))
            if expected.digest()!=actual.digest():raise ValueError('Source geometry payload changed')
        return len(m['meshes']),updated,missing_uv
