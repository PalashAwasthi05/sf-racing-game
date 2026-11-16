"""Small mmap-backed GLB reader for selective geometry inspection.

It reads the GLB JSON once, uses accessor bounds for broad phase, and decodes
positions/indices only for requested primitives. Images and materials are not
loaded. The original BIN chunk stays memory mapped for the reader lifetime.
"""
from __future__ import annotations

import json
import mmap
import copy
import os
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import trimesh

from regen.host_shell_apertures import GLB_TO_SOURCE, segment_boxes


_DTYPES={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,
         5125:np.uint32,5126:np.float32}
_COMPONENTS={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}


def _local_matrix(node):
    if 'matrix' in node:return np.asarray(node['matrix'],float).reshape((4,4),order='F')
    t=np.asarray(node.get('translation',[0,0,0]),float)
    s=np.asarray(node.get('scale',[1,1,1]),float)
    x,y,z,w=map(float,node.get('rotation',[0,0,0,1]))
    r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    m=np.eye(4);m[:3,:3]=r@np.diag(s);m[:3,3]=t
    return m


class GraphView:
    def __init__(self,entries):
        self._entries=entries
        self.nodes_geometry=list(entries)
    def get(self,node):
        mesh_name,transform=self._entries[node]
        return transform,mesh_name


class SceneView:
    """Minimal scene protocol used by the geometry ray audit."""
    def __init__(self,entries,meshes):
        self.geometry=meshes
        self.graph=GraphView(entries)


class GltfStream:
    def __init__(self,path,cache_size=96,chunk_cache_size=32):
        self.path=Path(path);self.file=open(self.path,'rb')
        self.mapping=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        magic,version,total=struct.unpack_from('<4sII',self.mapping,0)
        if magic!=b'glTF' or version!=2:raise ValueError('expected GLB v2')
        offset=12;self.document=None;self.bin_offset=None;self.bin_length=None
        while offset<total:
            size,kind=struct.unpack_from('<I4s',self.mapping,offset);offset+=8
            if kind==b'JSON':self.document=json.loads(self.mapping[offset:offset+size])
            elif kind==b'BIN\x00':self.bin_offset=offset;self.bin_length=size
            offset+=size
        if self.document is None or self.bin_offset is None:
            raise ValueError('GLB must contain JSON and BIN chunks')
        self.cache_size=int(cache_size);self.cache=OrderedDict()
        self.chunk_cache_size=int(chunk_cache_size);self.chunk_cache=OrderedDict()
        self.ray_blocks=OrderedDict()
        self._prepare_nodes()

    def close(self):
        self.cache.clear();self.chunk_cache.clear();self.ray_blocks.clear();self.mapping.close();self.file.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

    def _prepare_nodes(self):
        doc=self.document;nodes=doc.get('nodes',[])
        # A GLB may retain detached nodes after section splices. They are not
        # rendered by the active scene and must not become phantom occluders.
        scenes=doc.get('scenes') or []
        scene_index=int(doc.get('scene',0))
        if scenes and 0<=scene_index<len(scenes):
            roots=scenes[scene_index].get('nodes',[])
        else:
            children={int(c) for node in nodes for c in node.get('children',[])}
            roots=[i for i in range(len(nodes)) if i not in children]
        active=set()
        def visit(i):
            if i in active or not 0<=int(i)<len(nodes):return
            active.add(int(i))
            for child in nodes[int(i)].get('children',[]):visit(int(child))
        for root in roots:visit(int(root))
        parents={}
        for i in active:
            for child in nodes[i].get('children',[]):
                if int(child) in active:parents[int(child)]=i
        world_matrices={}
        def world(i):
            if i not in world_matrices:
                local=_local_matrix(nodes[i])
                world_matrices[i]=world(parents[i])@local if i in parents else local
            return world_matrices[i]
        rows=[];by_feature={}
        for i,node in enumerate(nodes):
            if i not in active:continue
            if 'mesh' not in node:continue
            mesh_index=int(node['mesh']);mesh=doc['meshes'][mesh_index]
            transform=world(i);node_name=node.get('name',f'node_{i}')
            for pi,primitive in enumerate(mesh.get('primitives',[])):
                if int(primitive.get('mode',4))!=4 or 'POSITION' not in primitive.get('attributes',{}):continue
                accessor=doc['accessors'][primitive['attributes']['POSITION']]
                if 'min' not in accessor or 'max' not in accessor:
                    positions=self._accessor(primitive['attributes']['POSITION'])
                    lo=positions.min(axis=0);hi=positions.max(axis=0)
                else:
                    lo=np.asarray(accessor['min'],float);hi=np.asarray(accessor['max'],float)
                corners=np.array([[x,y,z,1.] for x in (lo[0],hi[0])
                                  for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                source=(GLB_TO_SOURCE@(corners@transform.T).T).T[:,:3]
                row={'node':node_name if len(mesh.get('primitives',[]))==1 else f'{node_name}#primitive-{pi}',
                     'nodeIndex':i,'meshIndex':mesh_index,'primitiveIndex':pi,
                     'transform':transform,'primitive':primitive,
                     'minSource':source.min(axis=0),'maxSource':source.max(axis=0),
                     'material':primitive.get('material')}
                rows.append(row)
                parts=node_name.split('__',2)
                if len(parts)==3 and parts[0] in ('buildings','facades'):
                    by_feature.setdefault(parts[1],[]).append(row)
        self.rows=rows;self.by_node={r['node']:r for r in rows}
        self.by_feature=by_feature
        self.node_index_by_name={node.get('name',f'node_{i}'):i
                                 for i,node in enumerate(nodes)
                                 if i in active and 'mesh' in node}
        self.node_bounds=[(r['node'],r['meshIndex'],r['primitiveIndex'],r['transform'],
                           r['minSource'],r['maxSource']) for r in rows]

    def _accessor(self,index):
        accessor=self.document['accessors'][int(index)]
        if accessor.get('sparse'):raise ValueError('sparse accessors are unsupported by audit reader')
        view=self.document['bufferViews'][accessor['bufferView']]
        dtype=np.dtype(_DTYPES[int(accessor['componentType'])])
        width=_COMPONENTS[accessor['type']];count=int(accessor['count'])
        offset=self.bin_offset+int(view.get('byteOffset',0))+int(accessor.get('byteOffset',0))
        stride=int(view.get('byteStride',dtype.itemsize*width))
        if stride==dtype.itemsize*width:
            array=np.frombuffer(self.mapping,dtype=dtype,count=count*width,offset=offset).reshape(count,width)
        else:
            array=np.ndarray((count,width),dtype=dtype,buffer=self.mapping,offset=offset,
                             strides=(stride,dtype.itemsize))
        return array

    def mesh(self,row):
        key=(row['meshIndex'],row['primitiveIndex'])
        cached=self.cache.pop(key,None)
        if cached is not None:
            self.cache[key]=cached;return cached
        primitive=row['primitive'];attributes=primitive['attributes']
        vertices=np.asarray(self._accessor(attributes['POSITION']),dtype=np.float64)
        if 'indices' in primitive:
            indices=np.asarray(self._accessor(primitive['indices'])).reshape(-1)
        else:indices=np.arange(len(vertices),dtype=np.uint32)
        if len(indices)%3:raise ValueError('triangle primitive has non-triangle index count')
        mesh=trimesh.Trimesh(vertices=vertices,faces=indices.reshape(-1,3),process=False)
        normal_accessor=attributes.get('NORMAL')
        if normal_accessor is not None:
            normals=np.asarray(self._accessor(normal_accessor),dtype=np.float64)
            if normals.shape==vertices.shape:
                # Preserve authored smooth/weighted normals for mesh edits.
                mesh.vertex_normals=normals
        uv_accessor=attributes.get('TEXCOORD_0')
        if uv_accessor is not None:
            from trimesh.visual.texture import TextureVisuals
            uv=np.asarray(self._accessor(uv_accessor),dtype=np.float64)
            if len(uv)==len(vertices):mesh.visual=TextureVisuals(uv=uv)
        self.cache[key]=mesh
        while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return mesh

    def scene(self,rows):
        entries={};meshes={}
        for i,row in enumerate(rows):
            geom=f'lazy_{i}_{row["meshIndex"]}_{row["primitiveIndex"]}'
            entries[row['node']]=(geom,row['transform'])
            meshes[geom]=self.mesh(row)
        return SceneView(entries,meshes)

    def _chunk_bounds(self,row,chunk_faces=512):
        # Bounds are in world coordinates. Instanced meshes share vertices but
        # each node can stand at a different position or have a different scale.
        key=(row['nodeIndex'],row['primitiveIndex'],chunk_faces)
        cached=self.chunk_cache.pop(key,None)
        if cached is not None:
            self.chunk_cache[key]=cached;return cached
        primitive=row['primitive'];attrs=primitive['attributes']
        positions=self._accessor(attrs['POSITION'])
        indices=(self._accessor(primitive['indices']).reshape(-1) if 'indices' in primitive
                 else np.arange(len(positions),dtype=np.uint32))
        if len(indices)%3:raise ValueError('triangle primitive has non-triangle index count')
        chunks=[];transform=row['transform']
        for start in range(0,len(indices),chunk_faces*3):
            end=min(len(indices),start+chunk_faces*3)
            faces=indices[start:end].reshape(-1,3)
            points=np.asarray(positions[faces.reshape(-1)],dtype=np.float64).reshape(-1,3,3)
            lo=points.min(axis=(0,1));hi=points.max(axis=(0,1))
            corners=np.array([[x,y,z,1.] for x in (lo[0],hi[0])
                              for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            source=(GLB_TO_SOURCE@(corners@transform.T).T).T[:,:3]
            chunks.append((start,end,source.min(axis=0),source.max(axis=0)))
        arrays=(chunks,np.asarray([part[2] for part in chunks]).reshape(-1,3),
                np.asarray([part[3] for part in chunks]).reshape(-1,3))
        self.chunk_cache[key]=arrays
        while len(self.chunk_cache)>self.chunk_cache_size:self.chunk_cache.popitem(last=False)
        return arrays

    def _ray_block(self,row,start,end,positions,indices):
        """World-space bounds for repeated nearby rays through an immutable mesh."""
        from regen.host_shell_apertures import GLB_TO_SOURCE
        key=(row['nodeIndex'],row['primitiveIndex'],start,end)
        cached=self.ray_blocks.pop(key,None)
        if cached is None:
            faces=indices[start:end].reshape(-1,3)
            local=np.asarray(positions[faces],dtype=np.float64)
            world=local@row['transform'][:3,:3].T+row['transform'][:3,3]
            triangles=world@GLB_TO_SOURCE[:3,:3].T+GLB_TO_SOURCE[:3,3]
            low=triangles.min(axis=1)-1e-5;high=triangles.max(axis=1)+1e-5
            starts=np.arange(0,len(faces),16)
            cached=(triangles,low,high,np.minimum.reduceat(low,starts,axis=0),
                    np.maximum.reduceat(high,starts,axis=0))
        self.ray_blocks[key]=cached
        # 512 faces per block, roughly 62 KiB each. Evict distant neighborhoods.
        if len(self.ray_blocks)>1536:self.ray_blocks.popitem(last=False)
        return cached

    def segment_mesh_hits(self,row,origin,target):
        """Triangle-test only small indexed chunks crossed by one camera ray."""
        chunks,low,high=self._chunk_bounds(row)
        selected=[chunks[i] for i in np.flatnonzero(segment_boxes(origin,target,low-.02,high+.02))]
        if not selected:return []
        primitive=row['primitive'];attrs=primitive['attributes']
        positions=self._accessor(attrs['POSITION'])
        indices=(self._accessor(primitive['indices']).reshape(-1) if 'indices' in primitive
                 else np.arange(len(positions),dtype=np.uint32))
        origin=np.asarray(origin,float);delta=np.asarray(target,float)-origin
        length=float(np.linalg.norm(delta))
        if length<1e-8:return []
        direction=delta/length
        blocks=[self._ray_block(row,start,end,positions,indices) for start,end,_,_ in selected]
        # Check sixteen-face boxes together. Only the few sub-blocks touched by
        # this ray need individual triangle bounds or triangle coordinates.
        group_lows=np.concatenate([block[3] for block in blocks])
        group_highs=np.concatenate([block[4] for block in blocks])
        groups=np.flatnonzero(segment_boxes(origin,target,group_lows,group_highs))
        if not len(groups):return []
        ends=np.cumsum([len(block[3]) for block in blocks])
        owners=np.searchsorted(ends,groups,side='right')
        offsets=(groups-np.r_[0,ends][owners])*16
        spans=[(blocks[owner],slice(start,start+16)) for owner,start in zip(owners,offsets)]
        low=np.concatenate([b[1][s] for b,s in spans])
        high=np.concatenate([b[2][s] for b,s in spans])
        crossed=segment_boxes(origin,target,low,high)
        tri=np.concatenate([b[0][s] for b,s in spans])[crossed]
        if not len(tri):return []
        face_ids=np.concatenate([selected[owner][0]//3+
            np.arange(start,min(start+16,len(blocks[owner][0])))
            for owner,start in zip(owners,offsets)])[crossed]
        e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
        h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
        valid=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[valid]=1./det[valid]
        s=origin-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h)
        valid&=(u>=-1e-8)&(u<=1.+1e-8);q=np.cross(s,e1)
        v=inv*np.einsum('ij,ij->i',np.broadcast_to(direction,q.shape),q)
        valid&=(v>=-1e-8)&(u+v<=1.+1e-8)
        dist=inv*np.einsum('ij,ij->i',e2,q);valid&=(dist>1e-5)&(dist<=length+.02)
        hits=[]
        for ix in np.flatnonzero(valid)[np.argsort(dist[valid])]:
            normal=np.cross(e1[ix],e2[ix]);normal/=max(1e-12,np.linalg.norm(normal))
            hits.append({'distanceM':float(dist[ix]),'node':row['node'],
                         'pointSource':origin+direction*dist[ix],
                         'triangleIndex':int(face_ids[ix]),'normalSource':normal})
        return hits

    def rows_for_feature(self,feature_id):
        return self.by_feature.get(feature_id,[])

    def document_node_index(self,node_name):
        return self.node_index_by_name.get(node_name)


class GlbAppendWriter:
    """Append mesh accessors and replace selected nodes without rewriting old BIN bytes."""
    def __init__(self,stream):
        self.stream=stream;self.document=copy.deepcopy(stream.document)
        self.binary_offset=int(stream.bin_length);self.temp=tempfile.TemporaryFile()
        self.material_by_name={m.get('name'):i for i,m in enumerate(self.document.get('materials',[]))
                               if m.get('name')}
        self.changed_nodes=set();self.added_nodes=[]

    def close(self):self.temp.close()

    def _append_array(self,array,component_type,target,accessor_type):
        dtype=np.dtype({5125:'<u4',5126:'<f4'}[component_type])
        values=np.ascontiguousarray(array,dtype=dtype)
        pad=(-self.binary_offset)%4
        if pad:self.temp.write(b'\x00'*pad);self.binary_offset+=pad
        offset=self.binary_offset;data=values.tobytes(order='C')
        self.temp.write(data);self.binary_offset+=len(data)
        view_index=len(self.document.setdefault('bufferViews',[]))
        view={'buffer':0,'byteOffset':offset,'byteLength':len(data),'target':target}
        self.document['bufferViews'].append(view)
        accessor_index=len(self.document.setdefault('accessors',[]))
        accessor={'bufferView':view_index,'byteOffset':0,'componentType':component_type,
                  'count':int(values.shape[0]),'type':accessor_type}
        if accessor_type!='SCALAR' and values.size:
            accessor['min']=values.min(axis=0).astype(float).tolist()
            accessor['max']=values.max(axis=0).astype(float).tolist()
        self.document['accessors'].append(accessor)
        return accessor_index

    def material_index(self,material):
        name=str(getattr(material,'name','material') or 'material')
        if name in self.material_by_name:return self.material_by_name[name]
        # Preserve explicit typed material classes and their PBR factors.
        factor=np.asarray(getattr(material,'baseColorFactor',[160,160,160,255]),dtype=float).reshape(-1)
        if factor.max(initial=0)>1.0:factor=factor/255.
        factor=np.clip(factor,0.,1.)
        if len(factor)==3:factor=np.r_[factor,1.]
        pbr={'baseColorFactor':factor.tolist(),
             'metallicFactor':float(getattr(material,'metallicFactor',0.)),
             'roughnessFactor':float(getattr(material,'roughnessFactor',.7))}
        docmat={'name':name,'pbrMetallicRoughness':pbr}
        alpha_mode=getattr(material,'alphaMode',None)
        if alpha_mode in ('BLEND','MASK'):docmat['alphaMode']=alpha_mode
        index=len(self.document.setdefault('materials',[]));self.document['materials'].append(docmat)
        self.material_by_name[name]=index
        return index

    def material_for_name(self,name,visual_material=None):
        name=str(name)
        if visual_material is not None:
            index=self.material_index(visual_material)
            if name==getattr(visual_material,'name',None):return index
        if name in self.material_by_name:return self.material_by_name[name]
        # Existing generator materials commonly suffix their colour recipe.
        matches=[(k,v) for k,v in self.material_by_name.items() if k.endswith(name)]
        if matches:return sorted(matches,key=lambda q:(len(q[0]),q[0]))[0][1]
        from trimesh.visual.material import PBRMaterial
        material=PBRMaterial(name=name)
        if len(name)==7 and name.startswith('#'):
            try:material.baseColorFactor=np.asarray([int(name[i:i+2],16) for i in (1,3,5)]+[255],dtype=np.uint8)
            except ValueError:pass
        elif name=='glass':
            material.baseColorFactor=np.asarray([90,115,133,255],dtype=np.uint8)
            material.metallicFactor=.05;material.roughnessFactor=.35
        return self.material_index(material)

    def add_mesh(self,mesh,name,material_index,extras=None):
        vertices=np.asarray(mesh.vertices,dtype=np.float32)
        faces=np.asarray(mesh.faces,dtype=np.uint32).reshape(-1)
        attributes={'POSITION':self._append_array(vertices,5126,34962,'VEC3')}
        normals=np.asarray(mesh.vertex_normals,dtype=np.float32)
        if normals.shape==vertices.shape:
            attributes['NORMAL']=self._append_array(normals,5126,34962,'VEC3')
        uv=getattr(mesh.visual,'uv',None)
        if uv is not None and len(uv)==len(vertices):
            attributes['TEXCOORD_0']=self._append_array(np.asarray(uv,dtype=np.float32),5126,34962,'VEC2')
        indices=self._append_array(faces,5125,34963,'SCALAR')
        mesh_index=len(self.document.setdefault('meshes',[]))
        mesh_extras={'units':'meters','layer':'facade','featureId':None,
                     'evidence':'source-measured-aperture-repair'}
        if extras:mesh_extras.update(extras)
        primitive={'attributes':attributes,'indices':indices,'mode':4}
        # Omitting material is valid glTF and means the renderer's default
        # material. Preserve that behavior for replaced untyped shell meshes.
        if material_index is not None:primitive['material']=int(material_index)
        self.document['meshes'].append({'name':name,'extras':mesh_extras,
            'primitives':[primitive]})
        return mesh_index

    def replace_node_mesh(self,node_name,mesh,material_index,extras=None):
        row=self.stream.by_node.get(node_name)
        node_index=(row.get('nodeIndex') if row is not None
                    else self.stream.document_node_index(node_name))
        if node_index is None:raise KeyError(f'GLB node missing: {node_name}')
        node=self.document['nodes'][node_index]
        new_primitive_mesh=self.add_mesh(
            mesh,node.get('name',node_name),material_index,extras)
        primitive=self.document['meshes'][new_primitive_mesh]['primitives'][0]
        old_mesh=self.document['meshes'][int(node['mesh'])]
        primitive_index=int(row.get('primitiveIndex',0)) if row is not None else 0
        if not 0<=primitive_index<len(old_mesh.get('primitives',[])):
            raise IndexError(f'primitive index out of range for {node_name}')
        # Keep sibling primitives on a multi-primitive node and keep any other
        # nodes sharing the original mesh untouched. The replacement mesh is a
        # private clone attached only to the selected node.
        replacement=copy.deepcopy(old_mesh)
        replacement['primitives'][primitive_index]=primitive
        replacement['name']=node.get('name',node_name)
        mesh_index=len(self.document['meshes'])
        self.document['meshes'].append(replacement)
        node['mesh']=mesh_index
        self.changed_nodes.add(node_name)
        return mesh_index

    def add_node(self,node_name,mesh,material_index,transform,extras=None):
        mesh_index=self.add_mesh(mesh,node_name,material_index,extras)
        matrix=np.asarray(transform,dtype=float).reshape(4,4)
        node={'name':node_name,'mesh':mesh_index,'matrix':matrix.flatten(order='F').tolist(),
              'extras':{'featureId':(extras or {}).get('featureId'),
                        'evidence':'source-measured-aperture-repair'}}
        self.document['nodes'].append(node)
        scene_index=int(self.document.get('scene',0))
        self.document['scenes'][scene_index].setdefault('nodes',[]).append(len(self.document['nodes'])-1)
        self.added_nodes.append(node_name)
        return mesh_index

    def write(self,path,chunk_size=8<<20):
        self.document['buffers'][0]['byteLength']=self.binary_offset
        json_data=json.dumps(self.document,separators=(',',':'),ensure_ascii=False).encode('utf-8')
        json_data+=b' '*((-len(json_data))%4)
        bin_length=self.binary_offset
        total=12+8+len(json_data)+8+bin_length
        out_path=Path(path);out_path.parent.mkdir(parents=True,exist_ok=True)
        with open(out_path,'wb') as out:
            out.write(struct.pack('<4sII',b'glTF',2,total))
            out.write(struct.pack('<I4s',len(json_data),b'JSON'));out.write(json_data)
            out.write(struct.pack('<I4s',bin_length,b'BIN\x00'))
            pos=self.stream.bin_offset;remain=self.stream.bin_length
            while remain:
                size=min(chunk_size,remain);out.write(self.stream.mapping[pos:pos+size])
                pos+=size;remain-=size
            self.temp.seek(0)
            while True:
                chunk=self.temp.read(chunk_size)
                if not chunk:break
                out.write(chunk)
        return out_path


def bounds_zbase(rows,feature_id):
    lows=[r['minSource'][2] for r in rows
          if r['node'].startswith(f'buildings__{feature_id}___') and '__observed_' not in r['node']]
    return float(min(lows)) if lows else None

