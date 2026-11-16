"""Compile a pinned portable world to a route extent without mapping the source BIN.

Usage: python3 package_route_block.py PUBLIC block-trial/trial.json --margin 42
Only /compare/route-block.json is mutable; the frozen input is never rewritten.
Memory scales with JSON metadata plus one source primitive (not the whole GLB).
"""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from package_startup import emit_chunks, prune_material_resources, supported

DTYPES = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
COPY_BYTES = 1024 * 1024


def local_path(public, url):
    path = (public / str(url).lstrip('/')).resolve()
    if not path.is_relative_to(public.resolve()):
        raise ValueError('Resource escapes public directory')
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_json(public, stem, value):
    data = json_bytes(value)
    url = '/compare/' + stem + '-' + hashlib.sha256(data).hexdigest()[:16] + '.json'
    target = local_path(public, url)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.route-json-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return url


def route_box(route, margin):
    if len(route) < 2 or not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid route or margin')
    if any(len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) for p in route):
        raise ValueError('Route must contain finite east/north pairs')
    return [min(p[0] for p in route)-margin, min(p[1] for p in route)-margin,
            max(p[0] for p in route)+margin, max(p[1] for p in route)+margin]


def touches(lo, hi, box):
    return lo[0] <= box[2] and hi[0] >= box[0] and lo[1] <= box[3] and hi[1] >= box[1]


def points_touch(points, box, pad=0):
    if not points:
        return False
    return touches([min(p[0] for p in points)-pad, min(p[1] for p in points)-pad],
                   [max(p[0] for p in points)+pad, max(p[1] for p in points)+pad], box)


def unpack_portable(public, descriptor, output):
    digest = hashlib.sha256()
    size = 0
    for i, url in enumerate(descriptor['chunks']):
        if url != f"/world-{descriptor['contentHash']}-{i:02}.bin":
            raise ValueError('Unpinned portable chunk URL')
        path = local_path(public, url)
        with path.open('rb') as stream:
            while data := stream.read(COPY_BYTES):
                digest.update(data)
        with gzip.open(path, 'rb') as stream:
            while data := stream.read(COPY_BYTES):
                size += len(data)
                if size > descriptor['uncompressedBytes']:
                    raise ValueError('Portable exceeds declared length')
                output.write(data)
    if size != descriptor['uncompressedBytes'] or digest.hexdigest()[:12] != descriptor['contentHash']:
        raise ValueError('Portable length/hash mismatch')
    output.flush()


class Source:
    def __init__(self, stream):
        self.stream = stream
        stream.seek(0)
        magic, version, total, length, kind = struct.unpack('<5I', stream.read(20))
        if (magic, version, kind) != (0x46546c67, 2, 0x4e4f534a) or length > 64*1024*1024:
            raise ValueError('Invalid or oversized GLB JSON')
        self.model = json.loads(stream.read(length))
        size, kind = struct.unpack('<II', stream.read(8))
        self.offset, self.size = stream.tell(), size
        stream.seek(0, 2)
        if kind != 0x004e4942 or total != stream.tell() or total != self.offset+size:
            raise ValueError('Invalid GLB BIN length')
        m = self.model
        if (not supported(m) or m.get('cameras') or any(mesh.get('weights') for mesh in m['meshes'])
                or any(n.get('skin') is not None or n.get('weights') for n in m['nodes'])):
            raise ValueError('Unsupported glTF: expected static portable mesh layout')
        for a in m['accessors']:
            if a['type'] not in WIDTHS or a['componentType'] not in DTYPES:
                raise ValueError('Unsupported accessor layout')
        for mesh in m['meshes']:
            for p in mesh['primitives']:
                if p.get('mode', 4) not in (0, 4) or 'POSITION' not in p['attributes']:
                    raise ValueError('Only points and triangle primitives are supported')

    def view_range(self, index):
        v = self.model['bufferViews'][index]
        offset, length = v.get('byteOffset', 0), v['byteLength']
        if offset < 0 or length < 0 or offset+length > self.size:
            raise ValueError('Buffer view outside BIN')
        return offset, length

    def array(self, index):
        a = self.model['accessors'][index]
        v = self.model['bufferViews'][a['bufferView']]
        start, length = self.view_range(a['bufferView'])
        dtype, width = np.dtype(DTYPES[a['componentType']]), WIDTHS[a['type']]
        stride = v.get('byteStride', dtype.itemsize*width)
        offset, count = a.get('byteOffset', 0), a['count']
        span = (count-1)*stride + dtype.itemsize*width if count else 0
        if count < 0 or stride < dtype.itemsize*width or offset < 0 or offset+span > length:
            raise ValueError('Accessor exceeds buffer view')
        self.stream.seek(self.offset+start+offset)
        raw = self.stream.read(span)
        if len(raw) != span:
            raise ValueError('Truncated accessor')
        return np.ndarray((count, width), dtype, buffer=raw, strides=(stride, dtype.itemsize)).copy()


def matrix_of(node):
    if 'matrix' in node:
        if any(k in node for k in ('translation', 'rotation', 'scale')):
            raise ValueError('Node mixes matrix and TRS')
        matrix = np.array(node['matrix'], dtype=np.float64).reshape(4, 4, order='F')
    else:
        x, y, z, w = node.get('rotation', [0, 0, 0, 1])
        matrix = np.eye(4)
        matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        matrix[:3, :3] *= node.get('scale', [1, 1, 1])
        matrix[:3, 3] = node.get('translation', [0, 0, 0])
    if not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Unsupported projective/nonfinite node transform')
    return matrix


def flat_nodes(model):
    result, visited = [], set()
    def walk(index, parent):
        if index in visited:
            raise ValueError('Repeated/cyclic scene node is unsupported')
        visited.add(index)
        node = model['nodes'][index]
        world = parent @ matrix_of(node)
        if 'mesh' in node:
            result.append((node, world))
        for child in node.get('children', []):
            walk(child, world)
    scene = model['scenes'][model.get('scene', 0)]
    for index in scene['nodes']:
        walk(index, np.eye(4))
    return result


def en_positions(positions, matrix):
    world = positions @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(world).all():
        raise ValueError('Nonfinite positions')
    return world[:, [0, 2]] * [1, -1]


def node_bounds(source, node, matrix):
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for p in source.model['meshes'][node['mesh']]['primitives']:
        a = source.model['accessors'][p['attributes']['POSITION']]
        if a.get('normalized') or a['type'] != 'VEC3' or a['componentType'] != 5126:
            raise ValueError('Expected unnormalized float VEC3 position')
        if 'min' in a and 'max' in a:
            positions = np.array(list(itertools.product(*zip(a['min'], a['max']))))
        else:
            positions = source.array(p['attributes']['POSITION'])
        en = en_positions(positions, matrix)
        lo = np.minimum(lo, en.min(axis=0)); hi = np.maximum(hi, en.max(axis=0))
    return lo, hi


BUILDING_LAYERS = {'buildings', 'facades', 'facade', 'roofs', 'foundations'}


def building_id(node, mesh):
    # Exported extras are canonical; display names can append observed-wall,
    # component and material suffixes to the same building's identity.
    metadata = {**mesh.get('extras', {}), **node.get('extras', {})}
    names = [node.get('name', ''), mesh.get('name', '')]
    parsed = [name.partition('___')[0].split('__') for name in names if isinstance(name, str)]
    layer = metadata.get('layer') or next((parts[0] for parts in parsed if len(parts) > 1), None)
    if layer not in BUILDING_LAYERS:
        return None
    feature = metadata.get('featureId')
    if isinstance(feature, (str, int)) and not isinstance(feature, bool) and str(feature):
        return str(feature)
    # Supported legacy names are layer__ID[__observed_...][___material].
    return next((parts[1] for parts in parsed if len(parts) > 1 and parts[0] in BUILDING_LAYERS and parts[1]), None)


def scope_world(data, building_ids, box):
    """Scope metadata only where route membership is evidenced by the output.

    Grid/ground and coordinates remain in the original absolute EN frame.
    Source generation/evidence counts have no mesh-level counting rules; keep
    them under sourceCounts instead of presenting them as route counts.
    """
    if 'buildings' in data:
        data['buildings'] = [feature for feature in data['buildings'] if str(feature.get('id')) in building_ids]
    if 'streetViews' in data:
        data['streetViews'] = [view for view in data['streetViews'] if points_touch([view['position']], box)]
    if 'counts' in data:
        data['sourceCounts'] = data['counts']
    data['counts'] = {'buildings': len(building_ids)}
    data.update(scope='route-block', boundsEN=box)
    return data


class Writer:
    def __init__(self, stream):
        self.stream, self.views, self.accessors = stream, [], []

    def append(self, data, metadata=None):
        offset = self.stream.tell()
        self.stream.write(data)
        length = len(data)
        self.stream.write(bytes(-length % 4))
        self.views.append({**(metadata or {}), 'buffer': 0, 'byteOffset': offset, 'byteLength': length})
        return len(self.views)-1

    def accessor(self, array, original):
        a = {k: copy.deepcopy(v) for k, v in original.items() if k not in ('bufferView', 'byteOffset', 'count', 'min', 'max')}
        a.update(bufferView=self.append(array.tobytes()), count=len(array))
        if 'min' in original or 'max' in original:
            a.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        self.accessors.append(a)
        return len(self.accessors)-1

    def copy_view(self, source, index):
        start, length = source.view_range(index)
        offset = self.stream.tell()
        source.stream.seek(source.offset+start)
        remaining = length
        while remaining:
            data = source.stream.read(min(COPY_BYTES, remaining))
            if not data:
                raise ValueError('Truncated image')
            self.stream.write(data); remaining -= len(data)
        self.stream.write(bytes(-length % 4))
        old = source.model['bufferViews'][index]
        self.views.append({**{k:v for k,v in old.items() if k not in ('byteStride', 'target')},
                           'buffer':0, 'byteOffset':offset, 'byteLength':length})
        return len(self.views)-1


def compile_glb(source_stream, bin_stream, packed_stream, box):
    source, writer = Source(source_stream), Writer(bin_stream)
    model = source.model
    nodes = flat_nodes(model)
    selected_buildings = set()
    for node, matrix in nodes:
        bid = building_id(node, model['meshes'][node['mesh']])
        if bid and touches(*node_bounds(source, node, matrix), box):
            selected_buildings.add(bid)
    out_nodes, out_meshes = [], []
    included_buildings = set()
    vertices = triangles = 0
    for node, matrix in nodes:
        mesh = model['meshes'][node['mesh']]
        bid = building_id(node, mesh)
        whole = bool(bid and bid in selected_buildings)
        if (bid and not whole) or (not whole and not touches(*node_bounds(source, node, matrix), box)):
            continue
        primitives = []
        for primitive in mesh['primitives']:
            attrs = primitive['attributes']
            positions = source.array(attrs['POSITION'])
            count = len(positions)
            fan = bool(node.get('extras', {}).get('foliageFanCentres') or mesh.get('extras', {}).get('foliageFanCentres'))
            indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(count, dtype=np.uint32)
            if indices.dtype.kind != 'u' or (len(indices) and int(indices.max()) >= count):
                raise ValueError('Invalid mesh indices')
            mode = primitive.get('mode', 4)
            if mode == 4 and len(indices) % 3:
                raise ValueError('Incomplete triangle')
            if whole:
                used = np.arange(count)
                order = indices
            elif fan or mode == 0:
                en = en_positions(positions, matrix)
                used = np.flatnonzero((en[:,0] >= box[0]) & (en[:,0] <= box[2]) & (en[:,1] >= box[1]) & (en[:,1] <= box[3]))
                order = np.array([0, 0, 0], dtype=np.uint32) if fan else np.arange(len(used), dtype=np.uint32)
                del en
            else:
                en = en_positions(positions, matrix)[indices.reshape(-1, 3)]
                lo, hi = en.min(axis=1), en.max(axis=1)
                keep = (lo[:,0] <= box[2]) & (hi[:,0] >= box[0]) & (lo[:,1] <= box[3]) & (hi[:,1] >= box[1])
                selected = indices.reshape(-1, 3)[keep].reshape(-1)
                used, order = np.unique(selected, return_inverse=True)
                del en, lo, hi, keep, selected
            if not len(used):
                del positions, indices, used, order
                continue
            out = {k:copy.deepcopy(v) for k,v in primitive.items() if k not in ('attributes', 'indices')}
            out['attributes'] = {}
            for name, aid in attrs.items():
                array = positions if name == 'POSITION' else source.array(aid)
                if len(array) != count:
                    raise ValueError('Mismatched vertex attribute counts')
                out['attributes'][name] = writer.accessor(array[used], model['accessors'][aid])
                del array
            dtype, component = ('<u2', 5123) if len(used) <= 65536 else ('<u4', 5125)
            out['indices'] = writer.accessor(np.asarray(order, dtype=dtype).reshape(-1,1), {'componentType':component, 'type':'SCALAR'})
            primitives.append(out)
            vertices += len(used); triangles += len(order)//3 if mode == 4 else 0
            del positions, indices, used, order
        if primitives:
            if bid:
                included_buildings.add(bid)
            out_node = {k:copy.deepcopy(v) for k,v in node.items() if k not in ('mesh', 'children', 'matrix', 'translation', 'rotation', 'scale')}
            out_node.update(mesh=len(out_meshes), matrix=matrix.reshape(-1, order='F').tolist())
            out_nodes.append(out_node)
            out_meshes.append({**{k:copy.deepcopy(v) for k,v in mesh.items() if k != 'primitives'}, 'primitives':primitives})
    if not out_nodes:
        raise ValueError('Route extent selected no meshes')
    result = copy.deepcopy(model)
    result.update(nodes=out_nodes, meshes=out_meshes, scenes=[{'nodes':list(range(len(out_nodes)))}], scene=0)
    prune_material_resources(result)
    copied_images = {}
    for image in result.get('images', []):
        old = image['bufferView']
        if old not in copied_images:
            copied_images[old] = writer.copy_view(source, old)
        image['bufferView'] = copied_images[old]
    result.update(accessors=writer.accessors, bufferViews=writer.views, buffers=[{'byteLength':bin_stream.tell()}])
    text = json_bytes(result).rstrip(b'\n'); text += b' ' * (-len(text)%4)
    packed_stream.write(struct.pack('<5I', 0x46546c67, 2, 28+len(text)+bin_stream.tell(), len(text), 0x4e4f534a))
    packed_stream.write(text); packed_stream.write(struct.pack('<II', bin_stream.tell(), 0x004e4942))
    bin_stream.seek(0); shutil.copyfileobj(bin_stream, packed_stream, COPY_BYTES)
    return {'meshes':len(out_meshes), 'images':len(result.get('images', [])), 'vertices':vertices, 'triangles':triangles,
            'buildings':len(included_buildings), 'buildingIds':sorted(included_buildings)}


def package(public, trial_path, margin=42):
    public = Path(public).resolve()
    trial_file = local_path(public, trial_path)
    trial_bytes = trial_file.read_bytes(); trial = json.loads(trial_bytes)
    manifest_file = local_path(public, trial['manifest'])
    original_bytes = manifest_file.read_bytes(); original = json.loads(original_bytes)
    source_hash = original['contentHash']
    portable = original.get('startup')
    if not portable or portable.get('version') != 3 or portable.get('sourceHash') != source_hash:
        portable = json.loads(local_path(public, f'/startup-v3-{source_hash}.json').read_text())
    if portable.get('version') != 3 or portable.get('sourceHash') != source_hash or portable.get('format') != 'gzip-chunks':
        raise ValueError('Pinned portable startup-v3 package is required')
    box = route_box(trial['routeEN'], margin)
    (public/'compare').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.route-block-', dir=public) as scratch:
        temp = Path(scratch)
        with (temp/'source.glb').open('w+b') as source, (temp/'bin').open('w+b') as binary, (temp/'block.glb').open('wb') as packed:
            unpack_portable(public, portable, source)
            stats = compile_glb(source, binary, packed, box)
        building_ids = stats.pop('buildingIds')
        descriptor = emit_chunks(temp/'block.glb', temp, public)
        descriptor.update(scope='route-block', routeEN=trial['routeEN'], boundsEN=box, marginM=margin,
                          sourceWorldHash=source_hash, sourcePortableHash=portable['contentHash'], compilerVersion=2, buildingIds=building_ids)
        counts = {}
        for key in ('world', 'cars', 'overhead'):
            if key not in original:
                continue
            data = json.loads(local_path(public, original[key]).read_text())
            if key == 'cars':
                data['cars'] = [c for c in data.get('cars', []) if points_touch([[c['x'],c['y']]], box, 4)]
                counts['cars'] = len(data['cars'])
            elif key == 'overhead':
                data['poles'] = [p for p in data.get('poles', []) if points_touch([[p['x'],p['y']]], box, 3)]
                data['wires'] = [w for w in data.get('wires', []) if points_touch(w['p'], box, w.get('r', 0)+2)]
                counts.update(poles=len(data['poles']), wires=len(data['wires']))
            elif key == 'world':
                scope_world(data, set(building_ids), box)
            descriptor[key] = write_json(public, 'route-block-'+key, data)
        descriptor['stats'] = {**stats, **counts}
        url = write_json(public, 'route-block', descriptor)
        if trial_file.read_bytes() != trial_bytes or manifest_file.read_bytes() != original_bytes:
            raise RuntimeError('Pinned trial or manifest changed while compiling; alias not published')
        alias = temp/'alias.json'; alias.write_bytes(json_bytes(descriptor)); alias.replace(public/'compare/route-block.json')
    return {'manifest':'/compare/route-block.json', 'immutableManifest':url, **stats, **counts,
            'compressedBytes':descriptor['compressedBytes'], 'uncompressedBytes':descriptor['uncompressedBytes']}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('public', type=Path)
    cli.add_argument('trial', help='Public-relative trial JSON path')
    cli.add_argument('--margin', type=float, default=42)
    args = cli.parse_args()
    print(json.dumps(package(args.public, args.trial, args.margin), indent=2))
