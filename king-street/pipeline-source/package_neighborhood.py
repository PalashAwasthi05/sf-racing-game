"""Compile a full pinned source neighborhood to permanent support + spatial detail.

Offline only. The source BIN is unpacked to disk; each source primitive is decoded
separately, never the full world. No GPU or parallel workers are used. Usage:
  python package_neighborhood.py PUBLIC --source /compare/current.json --tile 55
Only /compare/neighborhood-stream.json is mutable. Source inputs are never edited.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np

from package_drive_stream import compile_part, emit_sequential, convex_hull, layer_of
from package_drive_residuals import material_color
from package_route_block import (Source, flat_nodes, node_bounds, building_id,
    en_positions, local_path, json_bytes, write_json, unpack_portable)

VERSION = 5
SUPPORT = {'terrain', 'roads', 'sidewalks', 'markings', 'landscape', 'paths', 'foundations'}


class ProxySimplifier:
    """Weld geometry across UV/shading seams; preserve actual open boundaries."""
    def __init__(self):
        self.process = None

    def simplify(self, positions, normals, colors, indices):
        # The source UV seams are irrelevant to diffuse-only exterior proxies.
        shading = np.column_stack((normals, colors)).astype('<f4')
        keys = np.ascontiguousarray(positions, dtype='<f4')
        keys[keys == 0] = 0
        rows = keys.view(np.dtype((np.void, keys.strides[0]))).ravel()
        _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        mapped = inverse[indices].astype('<u4')
        if len(mapped) >= 384:
            if self.process is None:
                node = shutil.which('node')
                if not node:
                    raise ValueError('Node is required for bounded exterior simplification')
                self.process = subprocess.Popen([node, str(Path(__file__).with_name('simplify_neighborhood_proxy.mjs'))],
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            process = self.process
            process.stdin.write(struct.pack('<III', len(first), len(mapped), shading.shape[1]))
            process.stdin.write(np.ascontiguousarray(positions[first], dtype='<f4').tobytes())
            process.stdin.write(np.ascontiguousarray(shading[first], dtype='<f4').tobytes())
            process.stdin.write(mapped.tobytes()); process.stdin.flush()
            header = process.stdout.read(4)
            if len(header) != 4:
                raise ValueError('Exterior simplifier returned no result')
            count, = struct.unpack('<I', header)
            if count > len(mapped) or count % 3 or count == 0:
                raise ValueError('Invalid simplified exterior indices')
            data = process.stdout.read(count * 4)
            if len(data) != count * 4:
                raise ValueError('Truncated simplified exterior')
            mapped = np.frombuffer(data, dtype='<u4')
        used, compact = np.unique(mapped, return_inverse=True)
        used = first[used]
        p = positions[used]
        faces = compact.reshape(-1, 3)
        normal = np.cross(p[faces[:, 1]]-p[faces[:, 0]], p[faces[:, 2]]-p[faces[:, 0]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
        # Keep wall/roof hard edges. Averaging all shared-position normals made
        # broad planar walls shade as giant triangular prisms at their corners.
        # Byte-normal identity also welds coplanar faces without smoothing creases.
        corner_normals = np.repeat(np.rint(np.clip(normal, -1, 1)*127).astype('i1'), 3, axis=0)
        corner_positions = p[compact]
        corner_colors = colors[used][compact]
        keys = np.ascontiguousarray(np.column_stack((corner_positions, corner_normals.astype('<f4'), corner_colors)), dtype='<f4')
        rows = keys.view(np.dtype((np.void, keys.strides[0]))).ravel()
        _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        count = len(first)
        return (corner_positions[first], corner_normals[first].astype('<f4')/127,
                corner_colors[first], inverse.astype('<u2' if count <= 65535 else '<u4'))

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.stdout.close()
            if self.process.wait():
                raise RuntimeError('Exterior simplifier failed')
            self.process = None


def tile_of(points, size):
    """Origin-anchored half-open grid; negative coordinates use floor, never truncation."""
    return np.floor(np.asarray(points, dtype=float) / size).astype(np.int64)


def cell_id(tile):
    return 'grid-' + '-'.join(str(int(v)) for v in tile)


def base_resource_stats(packed):
    """Account authored support images instead of rejecting all startup textures."""
    with Path(packed).open('rb') as stream:
        model = Source(stream).model
    image_views = {image['bufferView'] for image in model.get('images', [])}
    return {'images': len(model.get('images', [])),
            'encodedImageBytes': sum(model['bufferViews'][i]['byteLength'] for i in image_views),
            'texturePolicy': 'original support PBR names/maps/settings retained; exterior proxies diffuse-only'}


def vertical_bounds(source, node, matrix):
    lo, hi = math.inf, -math.inf
    for primitive in source.model['meshes'][node['mesh']]['primitives']:
        a = source.model['accessors'][primitive['attributes']['POSITION']]
        if 'min' in a and 'max' in a:
            p = np.array(list(itertools.product(*zip(a['min'], a['max']))))
        else:
            p = source.array(primitive['attributes']['POSITION'])
        y = p @ matrix[1, :3] + matrix[1, 3]
        lo, hi = min(lo, float(y.min())), max(hi, float(y.max()))
    return lo, hi


class NeighborhoodPlan:
    weld_full_support = True
    """Adapter for the established sequential package writer.

    Buildings have one owner across every source layer. Anonymous batched props
    and foliage are partitioned exclusively by triangle center / leaf center.
    Explicit nonbuilding feature IDs have one owner across their source nodes.
    """
    def __init__(self, source, tile_size=55):
        if not math.isfinite(tile_size) or tile_size < 10:
            raise ValueError('Tile size must be finite and at least 10 metres')
        self.tile_size = tile_size
        self.nodes = flat_nodes(source.model)
        self.materials = copy.deepcopy(source.model.get('materials', []))
        self.bounds, self.groups, self.node_groups = {}, {}, {}
        self.hulls, self.owners, self.owner_tiles = {}, {}, {}
        self.base_materials = None
        self.active_whole = False
        self.world, self.overhead = None, None
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for node, matrix in self.nodes:
            mesh = source.model['meshes'][node['mesh']]
            a, b = node_bounds(source, node, matrix)
            self.bounds[id(node)] = (a, b)
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
            bid = building_id(node, mesh)
            extra = {**mesh.get('extras', {}), **node.get('extras', {})}
            feature = extra.get('featureId')
            key = ('building', bid) if bid else (('feature', layer_of(node, mesh), str(feature)) if feature is not None else None)
            self.node_groups[id(node)] = key
            if key is not None:
                if key not in self.groups:
                    self.groups[key] = [a.copy(), b.copy(), []]
                group = self.groups[key]
                group[0], group[1] = np.minimum(group[0], a), np.maximum(group[1], b)
                group[2].append((node, matrix))
        if not np.isfinite([*lo, *hi]).all():
            raise ValueError('Source neighborhood has no finite geometry')
        self.box = [*lo.tolist(), *hi.tolist()]
        first, last = tile_of(lo, tile_size), tile_of(hi, tile_size)
        self.tiles = [(e, n) for n in range(first[1], last[1]+1) for e in range(first[0], last[0]+1)]
        if len(self.tiles) > 4096:
            raise ValueError('Grid exceeds 4096 cells; check source coordinates or tile size')
        self.tile_index = {tile: i for i, tile in enumerate(self.tiles)}
        self.cells = [{'id': cell_id(tile), 'tileEN': [tile[0]*tile_size, tile[1]*tile_size,
                       (tile[0]+1)*tile_size, (tile[1]+1)*tile_size]} for tile in self.tiles]
        for key, (a, b, nodes) in self.groups.items():
            owner = self.tile_index[tuple(tile_of((a+b)/2, tile_size))]
            self.owner_tiles[key] = owner
            if key[0] != 'building':
                continue
            bid = key[1]
            self.owners[bid] = owner
            points, bottom, top = [], math.inf, -math.inf
            for node, matrix in nodes:
                mesh = source.model['meshes'][node['mesh']]
                low, high = vertical_bounds(source, node, matrix)
                bottom, top = min(bottom, low), max(top, high)
                if layer_of(node, mesh) == 'foundations':
                    for primitive in mesh['primitives']:
                        p = source.array(primitive['attributes']['POSITION'])
                        p = p @ matrix[:3, :3].T + matrix[:3, 3]
                        points.extend(p[:, [0, 2]].tolist())
            if not points:
                points = [[a[0], -a[1]], [a[0], -b[1]], [b[0], -a[1]], [b[0], -b[1]]]
            self.hulls[bid] = (convex_hull(points), bottom, top)
        # This exact naming is consumed by the established stream replacement path.
        self.hull_materials = []
        for cell in self.cells:
            self.hull_materials.append(len(self.materials))
            self.materials.append({'name': 'drive_stream_collision_hull_' + cell['id'],
                'pbrMetallicRoughness': {'baseColorFactor': [.5, .5, .5, 1], 'roughnessFactor': 1},
                'extras': {'class': 'collision_hull', 'streamCell': cell['id']}})

    def keep_node(self, node, mesh, bid, cell):
        layer = layer_of(node, mesh)
        self.active_whole = False
        if cell is None:
            self.active_whole = True
            return layer in SUPPORT
        if bid:
            return layer != 'foundations' and self.owners.get(bid) == cell
        if layer in SUPPORT:
            return False
        key = self.node_groups[id(node)]
        if key is not None:
            self.active_whole = True
            return self.owner_tiles[key] == cell
        a, b = self.bounds[id(node)]
        # Reject before decoding any primitive that cannot contribute to this tile.
        tile = self.cells[cell]['tileEN']
        return a[0] < tile[2] and b[0] >= tile[0] and a[1] < tile[3] and b[1] >= tile[1]

    def select(self, positions, matrix, indices, fan, mode, cell):
        if self.active_whole or cell is None:
            return np.arange(len(positions)), indices
        en = en_positions(positions, matrix)
        tile = np.asarray(self.tiles[cell])
        if fan or mode == 0:
            used = np.flatnonzero(np.all(tile_of(en, self.tile_size) == tile, axis=1))
            return used, np.array([0, 0, 0], dtype=np.uint32) if fan else np.arange(len(used), dtype=np.uint32)
        tris = indices.reshape(-1, 3)
        # Work in bounded blocks: no (entire triangle count, 3, 3) temporary.
        selected = []
        for start in range(0, len(tris), 65536):
            block = tris[start:start+65536]
            centres = (en[block[:, 0]] + en[block[:, 1]] + en[block[:, 2]]) / 3
            keep = np.all(tile_of(centres, self.tile_size) == tile, axis=1)
            if keep.any():
                selected.append(block[keep].reshape(-1))
        if not selected:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.unique(np.concatenate(selected), return_inverse=True)

    def prepare_base(self, source):
        """Retain authored support PBR; only distant exteriors become tint-only."""
        self.detail_materials = copy.deepcopy(self.materials)
        self.color_cache = {}

    def diffuse_material(self, source, index):
        original = source.model['materials'][index]
        pbr = original.get('pbrMetallicRoughness', {})
        color = material_color(source, index, self.color_cache).tolist()
        # Source vertex COLOR_0 is retained on proxies and support.
        result = {'name': original.get('name', 'material') + '__diffuse',
                  'pbrMetallicRoughness': {'baseColorFactor': color + [pbr.get('baseColorFactor', [1, 1, 1, 1])[3]],
                    'roughnessFactor': pbr.get('roughnessFactor', .8),
                    'metallicFactor': pbr.get('metallicFactor', 0)}}
        for key in ('alphaMode', 'alphaCutoff', 'doubleSided'):
            if key in original:
                result[key] = copy.deepcopy(original[key])
        return result

    def append_hulls(self, source, writer, nodes, meshes):
        # Reuse the established footprint extrusion and material conventions.
        from package_drive_stream import Plan
        Plan.append_hulls(self, source, writer, nodes, meshes)
        self.append_static_colliders(source, writer, nodes, meshes)
        from neighborhood_canopies import append_canopies
        self.canopy_stats = append_canopies(source, writer, nodes, meshes, self)
        print(json.dumps({'phase': 'distant-canopies', **self.canopy_stats}), flush=True)
        from neighborhood_impostors import build_facade_impostors
        self.atlas_stats = build_facade_impostors(source, self, writer, nodes, meshes)
        print(json.dumps({'phase': 'facade-impostors', **self.atlas_stats}), flush=True)
        by_cell = {}
        for node, matrix in self.nodes:
            mesh = source.model['meshes'][node['mesh']]
            bid = building_id(node, mesh)
            layer = layer_of(node, mesh)
            if not bid or layer != 'roofs':
                continue
            by_cell.setdefault((self.owners[bid], bid), []).append((node, matrix))
        simplifier = ProxySimplifier()
        self.proxy_stats = {'sourceVertices': 0, 'vertices': 0, 'sourceTriangles': 0, 'triangles': 0, 'meshes': 0}
        self.proxy_bounds = [np.full(2, np.inf), np.full(2, -np.inf)]
        roof_materials = {}
        try:
            for (owner, bid), members in sorted(by_cell.items()):
                cell = self.cells[owner]['id']
                groups = {}
                # Reduce one building only; never collapse adjacent buildings together.
                for node, matrix in members:
                    mesh = source.model['meshes'][node['mesh']]
                    for primitive in mesh['primitives']:
                        old = primitive.get('material')
                        if old is None or primitive.get('mode', 4) != 4:
                            continue
                        material = source.model['materials'][old]
                        name = material.get('name', '').lower()
                        glazing = any(word in name for word in ('glass', 'glazing'))
                        if layer_of(node, mesh) in ('facades', 'facade') and not glazing:
                            continue
                        key = (layer_of(node, mesh), glazing, material.get('alphaMode', 'OPAQUE'))
                        group = groups.setdefault(key, [[], [], [], [], 0])
                        attrs = primitive['attributes']
                        p = source.array(attrs['POSITION'])
                        p = (p @ matrix[:3, :3].T + matrix[:3, 3]).astype('<f4')
                        indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(p))
                        if 'NORMAL' in attrs:
                            normal = source.array(attrs['NORMAL']).astype(float)
                            normal = normal @ np.linalg.inv(matrix[:3, :3])
                        else:
                            faces = indices.reshape(-1, 3)
                            weighted = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
                            normal = np.zeros_like(p)
                            for corner in faces.T:
                                np.add.at(normal, corner, weighted)
                        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
                        color = np.ones((len(p), 4), dtype='<f4')
                        color[:, :3] *= material_color(source, old, self.color_cache)
                        color[:, 3] = material.get('pbrMetallicRoughness', {}).get('baseColorFactor', [1, 1, 1, 1])[3]
                        if 'COLOR_0' in attrs:
                            aid = attrs['COLOR_0']; tint = source.array(aid)
                            meta = source.model['accessors'][aid]
                            values = tint.astype('<f4')
                            if meta.get('normalized') and tint.dtype.kind in 'ui':
                                values /= np.iinfo(tint.dtype).max
                            color[:, :values.shape[1]] *= values
                        group[0].append(p); group[1].append(normal.astype('<f4'))
                        group[2].append(color); group[3].append(indices.astype('<u4') + group[4]); group[4] += len(p)
                for (layer, glazing, alpha), (positions, normals, colors, indices, _) in groups.items():
                    p, normal, color, index = map(np.concatenate, (positions, normals, colors, indices))
                    self.proxy_stats['sourceVertices'] += len(p)
                    self.proxy_stats['sourceTriangles'] += len(index) // 3
                    p, normal, color, index = simplifier.simplify(p, normal, color, index)
                    en = p[:, [0, 2]] * [1, -1]
                    self.proxy_bounds[0] = np.minimum(self.proxy_bounds[0], en.min(axis=0))
                    self.proxy_bounds[1] = np.maximum(self.proxy_bounds[1], en.max(axis=0))
                    material_key = (layer, glazing, alpha)
                    if material_key not in roof_materials:
                        roof_materials[material_key] = len(self.materials)
                        self.materials.append({'name': 'neighborhood_stream_exterior__' + layer + ('_glass' if glazing else ''),
                            'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'roughnessFactor': .2 if glazing else .85, 'metallicFactor': .1 if glazing else 0},
                            'doubleSided': True, 'alphaMode': alpha,
                            'extras': {'class': 'distant_exterior', 'bakedAppearance': True, 'layer': layer}})
                    mid = roof_materials[material_key]
                    attrs = {'POSITION': writer.accessor(p, {'componentType': 5126, 'type': 'VEC3', 'min': [], 'max': []}),
                             'NORMAL': writer.accessor(np.rint(np.clip(normal, -1, 1)*127).astype('i1'), {'componentType': 5120, 'type': 'VEC3', 'normalized': True}),
                             'COLOR_0': writer.accessor(np.rint(np.clip(color, 0, 1)*255).astype('u1'), {'componentType': 5121, 'type': 'VEC4', 'normalized': True})}
                    primitive = {'attributes': attrs, 'material': mid,
                        'indices': writer.accessor(index.reshape(-1, 1), {'componentType': 5123 if index.dtype.itemsize == 2 else 5125, 'type': 'SCALAR'})}
                    extra = {'layer': layer, 'class': 'distant_exterior', 'streamCell': cell, 'featureId': bid, 'bakedAppearance': True}
                    nodes.append({'name': 'neighborhood_exterior__' + bid + '__' + layer, 'mesh': len(meshes), 'extras': extra})
                    meshes.append({'primitives': [primitive], 'extras': extra})
                    self.proxy_stats['vertices'] += len(p); self.proxy_stats['triangles'] += len(index)//3
                    self.proxy_stats['meshes'] += 1
        finally:
            simplifier.close()
        print(json.dumps({'phase': 'merged-exteriors', **self.proxy_stats}), flush=True)

    def append_static_colliders(self, source, writer, nodes, meshes):
        """Cheap permanent source-derived poles and ground-attached trunk bounds."""
        boxes = []
        poles = (self.overhead or {}).get('poles', [])
        for pole in poles:
            x, north, y, height = (float(pole[key]) for key in ('x', 'y', 'z', 'h'))
            if np.isfinite([x, north, y, height]).all() and height > 0:
                boxes.append(([x-.08, y, -north-.08], [x+.08, y+height, -north+.08]))
        pole_count = len(boxes)
        world = self.world or {}
        grid = world.get('grid', {})
        ground = np.asarray(world.get('ground', []), dtype=float)
        if grid.get('n') and ground.size == grid['n']**2:
            # Source bark is batched. Geometrically welded connected components
            # identify cylinders without mistaking high branches for ground trunks.
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            for node, matrix in self.nodes:
                mesh = source.model['meshes'][node['mesh']]
                extra = {**mesh.get('extras', {}), **node.get('extras', {})}
                if layer_of(node, mesh) != 'vegetation' or extra.get('material') not in ('bark', 'trunk', 'wood'):
                    continue
                for primitive in mesh['primitives']:
                    p = source.array(primitive['attributes']['POSITION'])
                    p = (p @ matrix[:3, :3].T + matrix[:3, 3]).astype('<f4')
                    index = source.array(primitive['indices']).reshape(-1, 3) if 'indices' in primitive else np.arange(len(p)).reshape(-1, 3)
                    unique, inverse = np.unique(p, axis=0, return_inverse=True)
                    tri = inverse[index]
                    row = np.concatenate((tri[:, 0], tri[:, 0]))
                    col = np.concatenate((tri[:, 1], tri[:, 2]))
                    adjacency = coo_matrix((np.ones(len(row), dtype='u1'), (row, col)), shape=(len(unique), len(unique))).tocsr()
                    count, labels = connected_components(adjacency, directed=False)
                    low, high = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
                    np.minimum.at(low, labels, unique); np.maximum.at(high, labels, unique)
                    centres = (low + high) / 2
                    en = centres[:, [0, 2]] * [1, -1]
                    sample = np.rint((en-grid['min'])/grid['step']).astype(int)
                    valid = np.all((sample >= 0) & (sample < grid['n']), axis=1)
                    sample = np.clip(sample, 0, grid['n']-1)
                    terrain = ground[sample[:, 1]*grid['n']+sample[:, 0]]
                    extent = high-low
                    keep = (valid & (extent[:, 1] >= 1.25) & (extent[:, 0] <= 1.4) & (extent[:, 2] <= 1.4)
                            & (low[:, 1] <= terrain+.65) & (low[:, 1] >= terrain-1.0))
                    for a, b in zip(low[keep], high[keep]):
                        b = b.copy(); b[1] = min(b[1], a[1]+2.8)
                        boxes.append((a, b))
                    del p, index, unique, inverse, tri, row, col, adjacency, labels
        self.collider_stats = {'poles': pole_count, 'trunks': len(boxes)-pole_count, 'boxes': len(boxes)}
        if not boxes:
            return
        positions, indices = [], []
        pattern = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
                            2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], dtype='<u4')
        for a, b in boxes:
            indices.append(pattern + len(positions)*8)
            positions.append(np.array(list(itertools.product(*zip(a, b))), dtype='<f4'))
        positions = np.concatenate(positions); index = np.concatenate(indices)
        material = len(self.materials)
        extra = {'layer': 'street_objects', 'class': 'static_collision_proxy', 'collisionOnly': True,
                 'permanent': True, 'sourcePoleCount': pole_count, 'sourceTrunkCount': len(boxes)-pole_count}
        self.materials.append({'name': 'neighborhood_static_collision_proxy', 'extras': extra,
                               'pbrMetallicRoughness': {'baseColorFactor': [.2, .2, .2, 1]}})
        attrs = {'POSITION': writer.accessor(positions, {'componentType': 5126, 'type': 'VEC3', 'min': [], 'max': []})}
        primitive = {'attributes': attrs, 'material': material,
                     'indices': writer.accessor(index.reshape(-1, 1), {'componentType': 5125, 'type': 'SCALAR'})}
        nodes.append({'name': 'neighborhood_static_collision_proxy', 'mesh': len(meshes), 'extras': extra})
        meshes.append({'primitives': [primitive], 'extras': extra})
        print(json.dumps({'phase': 'static-colliders', **self.collider_stats}), flush=True)


def package(public, source_url='/compare/current.json', seed_url='/compare/drive-stream.json', tile_size=55):
    public = Path(public).resolve()
    source_path, seed_path = local_path(public, source_url), local_path(public, seed_url)
    original_bytes, seed_bytes = source_path.read_bytes(), seed_path.read_bytes()
    original, seed = json.loads(original_bytes), json.loads(seed_bytes)
    source_hash = original['contentHash']
    portable = original.get('startup')
    portable_path = None
    if not portable:
        portable_path = local_path(public, '/startup-v3-' + source_hash + '.json')
        portable_bytes = portable_path.read_bytes()
        portable = json.loads(portable_bytes)
    if portable.get('format') != 'gzip-chunks' or portable.get('version') != 3 or portable.get('sourceHash') != source_hash:
        raise ValueError('Matching immutable portable startup-v3 source required')
    if seed.get('sourceWorldHash') != source_hash:
        raise ValueError('Seed route must be from this source world')
    route = seed.get('routeEN', [])
    if len(route) < 2 or not all(len(p) == 2 and np.isfinite(p).all() for p in route):
        raise ValueError('Valid existing driving seed route required')
    # Preserve a short real-road route without inventing a neighborhood-specific spawn.
    route = route[:2]
    world_path = local_path(public, original['world'])
    world_bytes = world_path.read_bytes()
    world = json.loads(world_bytes)
    (public / 'compare').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.neighborhood-stream-', dir=public) as scratch:
        temp = Path(scratch)
        with (temp / 'source.glb').open('w+b') as stream:
            unpack_portable(public, portable, stream)
            source = Source(stream)
            plan = NeighborhoodPlan(source, tile_size)
            plan.world = world
            if 'overhead' in original:
                plan.overhead = json.loads(local_path(public, original['overhead']).read_text())
            print(json.dumps({'phase': 'plan', 'cells': len(plan.cells), 'buildings': len(plan.owners), 'boundsEN': plan.box}), flush=True)
            outputs = []
            for cell in [None] + list(range(len(plan.cells))):
                name = 'base' if cell is None else plan.cells[cell]['id']
                if cell is None:
                    plan.prepare_base(source)
                elif cell == 0:
                    plan.materials = plan.detail_materials
                packed = temp / 'part.glb'
                try:
                    with (temp / 'binary').open('w+b') as binary, packed.open('wb') as output:
                        stats = compile_part(source, binary, output, plan, cell)
                    if cell is None:
                        stats['proxies'] = plan.proxy_stats
                        stats['facadeAtlas'] = plan.atlas_stats
                        stats['colliders'] = plan.collider_stats
                        stats['canopies'] = plan.canopy_stats
                        stats['boundsEN'] = [*np.minimum(stats['boundsEN'][:2], plan.proxy_bounds[0]).tolist(),
                                             *np.maximum(stats['boundsEN'][2:], plan.proxy_bounds[1]).tolist()]
                except ValueError as error:
                    if cell is not None and str(error) == 'Route extent selected no meshes':
                        continue
                    raise
                if cell is None:
                    stats['resources'] = base_resource_stats(packed)
                descriptor = emit_sequential(packed, temp, public)
                descriptor.update(stats=stats, boundsEN=stats['boundsEN'], sourceWorldHash=source_hash)
                if cell is None:
                    base = descriptor
                else:
                    outputs.append({**plan.cells[cell], **descriptor})
                packed.unlink()
                print(json.dumps({'phase': name, 'compressedBytes': descriptor['compressedBytes'], **stats}), flush=True)
        # Metadata is the complete current world, never corridor-scoped.
        world.update(scope='neighborhood-stream', boundsEN=plan.box)
        urls = {'world': write_json(public, 'neighborhood-stream-world', world)}
        metadata_inputs = []
        for key in ('cars', 'overhead'):
            if key in original:
                path = local_path(public, original[key]); data = path.read_bytes()
                metadata_inputs.append((path, data))
                urls[key] = write_json(public, 'neighborhood-stream-' + key, json.loads(data))
        descriptor = {'version': 1, 'scope': 'neighborhood-stream', 'compilerVersion': VERSION,
            'sourceWorldHash': source_hash, 'sourcePortableHash': portable['contentHash'],
            'routeEN': route, 'tileSizeM': tile_size, 'boundsEN': plan.box, 'base': base, 'cells': outputs, **urls,
            'ownership': {'buildings': 'all complete source layers owned by full feature bounds center',
                'features': 'explicit nonbuilding feature IDs owned whole by feature bounds center',
                'batches': 'anonymous triangle centers and source foliage centers owned exclusively by half-open grid',
                'support': 'complete original roads/ground/sidewalks/markings/paths/landscape/foundations in base',
                'proxies': 'source-foundation collision hulls and original building/roof exteriors; source diffuse means and vertex tints; hide both classes by streamCell when detail attaches'}}
        if source_path.read_bytes() != original_bytes or seed_path.read_bytes() != seed_bytes or world_path.read_bytes() != world_bytes:
            raise RuntimeError('Pinned source or driving seed changed; alias not published')
        if portable_path and portable_path.read_bytes() != portable_bytes:
            raise RuntimeError('Portable source descriptor changed; alias not published')
        if any(path.read_bytes() != data for path, data in metadata_inputs):
            raise RuntimeError('Full source metadata changed; alias not published')
        immutable = write_json(public, 'neighborhood-stream', descriptor)
        # Tree-only cells are compiled sequentially from the completed packed
        # cells before the new neighborhood alias is published. Their binding
        # follows source/cell hashes, so fresh areas get the same runtime detail.
        from package_tree_detail import ensure_package
        tree_detail = ensure_package(public, immutable)
        from scene_residency import annotate_manifest
        annotate_manifest(public, descriptor, tree_detail)
        immutable = write_json(public, 'neighborhood-stream', descriptor)
        alias = temp / 'alias.json'; alias.write_bytes(json_bytes(descriptor))
        alias.replace(public / 'compare/neighborhood-stream.json')
    return {'manifest': '/compare/neighborhood-stream.json', 'immutableManifest': immutable, 'treeDetail': tree_detail,
            'cells': len(outputs), 'buildings': len(plan.owners), 'baseCompressedBytes': base['compressedBytes'],
            'cellsCompressedBytes': sum(c['compressedBytes'] for c in outputs)}


def refresh_base(public, source_url='/compare/current.json', publish=True):
    """Replace only permanent support/proxies; keep completed detail packages byte-identical."""
    public = Path(public).resolve()
    index_path = public / 'compare/neighborhood-stream.json'
    before = index_path.read_bytes(); manifest = json.loads(before)
    if manifest.get('scope') != 'neighborhood-stream' or manifest.get('version') != 1:
        raise ValueError('Completed neighborhood-stream version 1 index required')
    source_path = local_path(public, source_url)
    source_bytes = source_path.read_bytes(); original = json.loads(source_bytes)
    source_hash = manifest['sourceWorldHash']
    if original.get('contentHash') != source_hash:
        raise ValueError('Base refresh source world differs from completed detail cells')
    portable_path = None
    portable = original.get('startup')
    if not portable:
        portable_path = local_path(public, '/startup-v3-' + source_hash + '.json')
        portable_bytes = portable_path.read_bytes(); portable = json.loads(portable_bytes)
    if (portable.get('format') != 'gzip-chunks' or portable.get('version') != 3
            or portable.get('sourceHash') != source_hash or portable.get('contentHash') != manifest['sourcePortableHash']):
        raise ValueError('Base refresh requires exactly the detail cells portable source')
    metadata_inputs = []
    for key in ('world', 'overhead'):
        if key in original:
            path = local_path(public, original[key]); data = path.read_bytes()
            metadata_inputs.append((key, path, data))
    with tempfile.TemporaryDirectory(prefix='.neighborhood-base-', dir=public) as scratch:
        temp = Path(scratch)
        with (temp / 'source.glb').open('w+b') as stream:
            unpack_portable(public, portable, stream); source = Source(stream)
            plan = NeighborhoodPlan(source, manifest['tileSizeM'])
            for key, _, data in metadata_inputs:
                setattr(plan, key, json.loads(data))
            cells = {cell['id']: cell for cell in plan.cells}
            for cell in manifest['cells']:
                if cell['id'] not in cells or cell['tileEN'] != cells[cell['id']]['tileEN']:
                    raise ValueError('Existing detail grid does not match reconstructed source plan')
                if cell.get('sourceWorldHash') != source_hash or not all(local_path(public, url).is_file() for url in cell['chunks']):
                    raise ValueError('Existing detail cell is missing or belongs to another source')
            plan.prepare_base(source)
            packed = temp / 'base.glb'
            with (temp / 'binary').open('w+b') as binary, packed.open('wb') as output:
                stats = compile_part(source, binary, output, plan, None)
            stats['proxies'] = plan.proxy_stats
            stats['facadeAtlas'] = plan.atlas_stats
            stats['colliders'] = plan.collider_stats
            stats['canopies'] = plan.canopy_stats
            stats['boundsEN'] = [*np.minimum(stats['boundsEN'][:2], plan.proxy_bounds[0]).tolist(),
                                 *np.maximum(stats['boundsEN'][2:], plan.proxy_bounds[1]).tolist()]
            stats['resources'] = base_resource_stats(packed)
            base = emit_sequential(packed, temp, public)
            base.update(stats=stats, boundsEN=stats['boundsEN'], sourceWorldHash=source_hash)
        updated = {k: v for k, v in manifest.items() if k != 'windingCorrections'}
        updated.update({'base': base, 'compilerVersion': VERSION,
                   'baseRefresh': {'version': 1, 'previousBaseHash': manifest['base']['contentHash'],
                       'detailCellsUnchanged': True, 'proxyReduction': 'original per-building roof surfaces; 0.15m geometric error limit with open boundaries locked; source-derived facade atlases, two 2048 square pages maximum; no cross-building clustering',
                       'appearance': 'full source envelope/facade orthographic RGBA atlases; source material maps and vertex colors; original support PBR; roof diffuse means',
                       'topology': 'original roofs reduced per building within 0.15m geometric error; real source facade projection and silhouette alpha on actual footprint wall planes; no cross-building clustering'}})
        if index_path.read_bytes() != before or source_path.read_bytes() != source_bytes:
            raise RuntimeError('Completed detail index or source changed during base refresh')
        if portable_path and portable_path.read_bytes() != portable_bytes:
            raise RuntimeError('Portable source changed during base refresh')
        if any(path.read_bytes() != data for _, path, data in metadata_inputs):
            raise RuntimeError('Source world/pole metadata changed during base refresh')
        immutable = write_json(public, 'neighborhood-stream', updated)
        from package_tree_detail import ensure_package
        tree_detail = ensure_package(public, immutable)
        from scene_residency import annotate_manifest
        annotate_manifest(public, updated, tree_detail)
        immutable = write_json(public, 'neighborhood-stream', updated)
        if index_path.read_bytes() != before:
            raise RuntimeError('Detail index changed during residency metadata pass')
        destination = index_path if publish else index_path.with_name('neighborhood-stream-candidate.json')
        alias = temp / 'alias.json'; alias.write_bytes(json_bytes(updated)); alias.replace(destination)
    return {'manifest': '/compare/'+destination.name, 'immutableManifest': immutable, 'treeDetail': tree_detail,
            'detailCellsUnchanged': True, 'cells': len(manifest['cells']), 'baseCompressedBytes': base['compressedBytes'], 'stats': stats}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('public', type=Path)
    cli.add_argument('--source', default='/compare/current.json')
    cli.add_argument('--seed', default='/compare/drive-stream.json')
    cli.add_argument('--tile', type=float, default=55)
    cli.add_argument('--refresh-base', action='store_true', help='Rebuild only base and preserve existing cell chunks')
    cli.add_argument('--candidate', action='store_true', help='Stage the refreshed base for a drive before publishing')
    cli.add_argument('--facade-middle', action='store_true', help='Build/publish bounded texture-free facade cells after this neighborhood build')
    args = cli.parse_args()
    if args.candidate and not args.refresh_base:
        cli.error('--candidate requires --refresh-base')
    result = refresh_base(args.public, args.source, not args.candidate) if args.refresh_base else package(args.public, args.source, args.seed, args.tile)
    if args.facade_middle:
        from package_facade_middle import publish
        result.update(publish(args.public, result['manifest']))
    print(json.dumps(result, indent=2))
