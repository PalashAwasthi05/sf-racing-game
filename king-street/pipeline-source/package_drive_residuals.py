"""Bounded source-derived distant exteriors; no collision/source package changes.

Read one detail piece into a temporary file at a time. Preserve source envelopes,
roofs, courtyard topology and all significant facade faces. Tiny trim triangles
are omitted without joining faces or filling openings. Textures are reduced to linear-light diffuse means one image at a
time. Source leaf centers and their material tints remain resident across detail switches.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from package_route_block import Source, flat_nodes, building_id, unpack_portable, json_bytes

VERSION = 6
MAX_OUTPUT = 24 * 1024 * 1024


def binding(manifest):
    return {'version': VERSION, 'sourceWorldHash': manifest['sourceWorldHash'],
            'pieces': [{'id': p['id'], 'contentHash': p['contentHash']} for p in manifest.get('pieces', manifest['cells'])]}


def linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def material_color(source, index, cache):
    if index in cache:
        return cache[index]
    material = source.model['materials'][index]
    pbr = material.get('pbrMetallicRoughness', {})
    color = np.array(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3], dtype=float)
    texture = pbr.get('baseColorTexture')
    if texture:
        image = source.model['images'][source.model['textures'][texture['index']]['source']]
        start, count = source.view_range(image['bufferView'])
        source.stream.seek(source.offset + start)
        with Image.open(io.BytesIO(source.stream.read(count))) as im:
            # A single image decode, no parallel image pool or full world decode.
            rgb = np.asarray(im.convert('RGB').resize((32, 32), Image.Resampling.BOX)) / 255.
            color *= linear(rgb).mean(axis=(0, 1))
    cache[index] = np.clip(color, 0, 1)
    return cache[index]


def vertex_colors(source, primitive):
    index = primitive['attributes'].get('COLOR_0')
    if index is None:
        return None
    accessor = source.model['accessors'][index]
    raw = source.array(index)
    values = raw[:, :3].astype(float)
    if accessor.get('normalized') and accessor['componentType'] != 5126:
        values /= np.iinfo(raw.dtype).max
    # The detail renderer multiplies a linear tint by the material. A gain
    # above one lightens a dark material; quantize only the combined albedo.
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid source vertex tint')
    return values


def immutable_write(path, data):
    """Never replace an already published immutable URL, even after a rebuild."""
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError('Immutable output already exists with different bytes: ' + str(path))


def triangles(source, primitive, matrix):
    positions = source.array(primitive['attributes']['POSITION'])
    world = positions @ matrix[:3, :3].T + matrix[:3, 3]
    indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(world))
    return world[indices.reshape(-1, 3)]


def reduce_faces(faces, facade):
    """Discard only tiny facade trim; never convexify an envelope or roof."""
    cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    return faces[area2 >= (.16 if facade else 1e-7)]


def package(public):
    public = Path(public).resolve()
    index_path = public / 'compare/drive-stream.json'
    before = index_path.read_bytes()
    manifest = json.loads(before)
    bound = binding(manifest)
    # JS constructs this same ASCII JSON field order before SHA-256.
    fingerprint = hashlib.sha256(json.dumps(bound, separators=(',', ':')).encode()).hexdigest()[:16]
    entries = []
    stats = {'triangles': 0, 'sourceTriangles': 0, 'leafCenters': 0, 'buildings': 0}
    with tempfile.TemporaryDirectory(prefix='drive-residual-') as scratch:
        packed = Path(scratch) / 'residual.bin'
        with packed.open('wb') as output:
            for piece in manifest.get('pieces', manifest['cells']):
                layers = {}
                with tempfile.TemporaryFile() as stream:
                    unpack_portable(public, piece, stream)
                    source = Source(stream)
                    cache = {}
                    buildings = set()
                    foliage = []
                    for node, matrix in flat_nodes(source.model):
                        mesh = source.model['meshes'][node['mesh']]
                        extra = {**mesh.get('extras', {}), **node.get('extras', {})}
                        layer = extra.get('layer', node.get('name', '').split('__')[0])
                        bid = building_id(node, mesh)
                        if not bid and not (layer == 'vegetation' and extra.get('foliageFanCentres')):
                            continue
                        if bid:
                            buildings.add(bid)
                        for primitive in mesh['primitives']:
                            if not bid:
                                points = source.array(primitive['attributes']['POSITION']) @ matrix[:3, :3].T + matrix[:3, 3]
                                if len(points):
                                    # buildFoliage uses the source material color for each run of leaf cards.
                                    pbr = source.model['materials'][primitive['material']].get('pbrMetallicRoughness', {})
                                    foliage.append((points.astype('<f4'), pbr.get('baseColorFactor', [1, 1, 1, 1])[:3]))
                                continue
                            color = material_color(source, primitive['material'], cache)
                            tint = vertex_colors(source, primitive)
                            faces = triangles(source, primitive, matrix)
                            if tint is not None:
                                indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(tint))
                                color = color * tint[indices].mean(axis=0)
                            stats['sourceTriangles'] += len(faces)
                            faces = reduce_faces(faces, layer in ('facades', 'facade'))
                            if not len(faces):
                                continue
                            residual_layer = 'facades' if layer in ('facades', 'facade') else 'roofs' if layer == 'roofs' else 'buildings'
                            material_name = source.model['materials'][primitive['material']].get('name', '').lower()
                            glazing = any(word in material_name for word in ('glass', 'glazing'))
                            layers.setdefault(residual_layer + ('-glazing' if glazing else ''), []).append((faces, color))
                    stats['buildings'] += len(buildings)
                    for layer_key, parts in layers.items():
                        layer = layer_key.removesuffix('-glazing')
                        appearance = 'glazing' if layer_key.endswith('-glazing') else 'matte'
                        start = output.tell()
                        # Weld only vertices identical in position, surface normal and color.
                        # Keeping the normal in the key preserves every hard edge.
                        positions = np.concatenate([faces.reshape(-1, 3) for faces, _ in parts])
                        positions = np.round(positions, 3).astype('<f4')
                        tri = positions.reshape(-1, 3, 3)
                        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-14)
                        normals = np.repeat(np.rint(normals*127).astype('i1'), 3, axis=0)
                        colors = np.concatenate([np.tile(np.rint(np.clip(c, 0, 1)*255).astype('u1'), (len(f)*3, 1)) for f, c in parts])
                        keys = np.ascontiguousarray(np.column_stack((positions.view('u1').reshape(-1, 12), normals.view('u1'), colors)))
                        _, first, indices = np.unique(keys.view('V18').reshape(-1), return_index=True, return_inverse=True)
                        output.write(positions[first].tobytes())
                        output.write(normals[first].tobytes())
                        output.write(colors[first].tobytes())
                        output.write(bytes((-output.tell()) % 4))
                        index_bytes = 2 if len(first) <= 65535 else 4
                        output.write(indices.astype('<u2' if index_bytes == 2 else '<u4').tobytes())
                        output.write(bytes((-output.tell()) % 4))
                        if output.tell() > MAX_OUTPUT:
                            raise ValueError('Residual exceeds 24 MiB hard output bound')
                        entries.append({'id': piece['id'], 'layer': layer, 'appearance': appearance, 'offset': start,
                                        'vertices': len(first), 'indices': len(indices), 'indexBytes': index_bytes})
                        stats['triangles'] += len(indices) // 3
                    if foliage:
                        start = output.tell()
                        runs = []
                        for points, tint in foliage:
                            output.write(points.tobytes())
                            runs.append({'count': len(points), 'color': tint})
                        count = sum(run['count'] for run in runs)
                        entries.append({'id': piece['id'], 'layer': 'vegetation', 'appearance': 'foliage',
                                        'offset': start, 'vertices': count, 'runs': runs})
                        stats['leafCenters'] += count
                        if output.tell() > MAX_OUTPUT:
                            raise ValueError('Residual exceeds 24 MiB hard output bound')
                print(json.dumps({'piece': piece['id'], 'outputBytes': output.tell()}), flush=True)
        blob = packed.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        url = '/compare/drive-residual-' + digest[:16] + '.bin'
        immutable_write(public / url.lstrip('/'), blob)
    result = {**bound, 'binary': url, 'binaryHash': digest, 'byteLength': len(blob), 'entries': entries,
              'stats': stats, 'policy': 'Exact source envelope and roof topology; facade triangles >=0.08m2; source linear diffuse means; persistent source leaf centers; display only.'}
    target = public / 'compare' / ('drive-residuals-' + fingerprint + '.json')
    immutable_write(target, json_bytes(result))
    residual_url = '/' + str(target.relative_to(public))
    # All sidecar data is complete before publishing its reference. A concurrent
    # source/index change invalidates this attempt instead of moving its alias.
    if index_path.read_bytes() != before:
        raise RuntimeError('Driving index changed during distant exterior packaging')
    updated = {**manifest, 'distantExteriors': residual_url}
    index_data = json_bytes(updated)
    immutable_url = '/compare/drive-stream-' + hashlib.sha256(index_data).hexdigest()[:16] + '.json'
    immutable_write(public / immutable_url.lstrip('/'), index_data)
    with tempfile.NamedTemporaryFile(dir=index_path.parent, prefix='.drive-residual-index-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(index_data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if index_path.read_bytes() != before:
            raise RuntimeError('Driving index changed before distant exterior publication')
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({'manifest': '/' + str(target.relative_to(public)), 'byteLength': len(blob), **stats}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('public', type=Path)
    package(parser.parse_args().public)
