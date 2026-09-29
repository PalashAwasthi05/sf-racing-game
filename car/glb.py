"""A small glTF 2.0 binary writer: named nodes, one primitive per node, positions, normals, optional uv and colour."""
import json
import struct
import numpy as np


class GLB:
    def __init__(self):
        self.nodes, self.meshes, self.materials, self.accessors, self.views = [], [], [], [], []
        self.blob = bytearray()
        self.material_index = {}

    def _view(self, data, target):
        while len(self.blob) % 4:
            self.blob.append(0)
        off = len(self.blob)
        self.blob += data
        self.views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(data), 'target': target})
        return len(self.views) - 1

    def _accessor(self, arr, kind, target, minmax=False):
        comp = {np.float32: 5126, np.uint32: 5125}[arr.dtype.type]
        view = self._view(arr.tobytes(), target)
        acc = {'bufferView': view, 'componentType': comp, 'count': len(arr), 'type': kind}
        if minmax:
            acc['min'] = arr.min(0).tolist()
            acc['max'] = arr.max(0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def material(self, name, color=(.8, .8, .8, 1), metallic=0., roughness=.5, double=False):
        if name not in self.material_index:
            m = {'name': name, 'pbrMetallicRoughness': {'baseColorFactor': list(color), 'metallicFactor': metallic, 'roughnessFactor': roughness}}
            if double:
                m['doubleSided'] = True
            self.materials.append(m)
            self.material_index[name] = len(self.materials) - 1
        return self.material_index[name]

    def add(self, name, mesh, material, translation=None, extras=None, uv=None, color=None):
        if not len(mesh.T):
            return None
        V = mesh.V.astype(np.float32)
        attrs = {'POSITION': self._accessor(V, 'VEC3', 34962, True)}
        if mesh.N is not None:
            N = np.asarray(mesh.N, np.float64)
            ln = np.linalg.norm(N, axis=1)
            bad = ~np.isfinite(ln) | (ln < .5)
            if bad.any():
                # a zero or NaN normal turns into NaN in the shader and the bloom blur spreads it into black blocks
                raise ValueError('%s: %d vertices without a usable normal' % (name, bad.sum()))
            attrs['NORMAL'] = self._accessor((N / ln[:, None]).astype(np.float32), 'VEC3', 34962)
        if uv is not None:
            attrs['TEXCOORD_0'] = self._accessor(uv.astype(np.float32), 'VEC2', 34962)
        if color is not None:
            attrs['COLOR_0'] = self._accessor(color.astype(np.float32), 'VEC4' if color.shape[1] == 4 else 'VEC3', 34962)
        idx = self._accessor(mesh.T.astype(np.uint32).reshape(-1), 'SCALAR', 34963)
        mat = self.material_index[material] if isinstance(material, str) else material
        self.meshes.append({'name': name, 'primitives': [{'attributes': attrs, 'indices': idx, 'material': mat}]})
        node = {'name': name, 'mesh': len(self.meshes) - 1}
        if translation is not None:
            node['translation'] = [float(v) for v in translation]
        if extras:
            node['extras'] = extras
        self.nodes.append(node)
        return len(self.nodes) - 1

    def write(self, path, extras=None):
        gltf = {'asset': {'version': '2.0', 'generator': 'vehicle generator'}, 'scene': 0,
                'scenes': [{'nodes': list(range(len(self.nodes)))}], 'nodes': self.nodes, 'meshes': self.meshes,
                'materials': self.materials, 'accessors': self.accessors, 'bufferViews': self.views,
                'buffers': [{'byteLength': len(self.blob)}]}
        if extras:
            gltf['scenes'][0]['extras'] = extras
        js = json.dumps(gltf, separators=(',', ':')).encode()
        js += b' ' * ((4 - len(js) % 4) % 4)
        bin_ = bytes(self.blob) + b'\0' * ((4 - len(self.blob) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(bin_)
        with open(path, 'wb') as fh:
            fh.write(struct.pack('<III', 0x46546C67, 2, total))
            fh.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
            fh.write(struct.pack('<II', len(bin_), 0x004E4942) + bin_)
        return total
