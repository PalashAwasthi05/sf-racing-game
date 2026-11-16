"""Generate conservative per-triangle winding corrections for neighborhood tiles.

The output is a small, optional sidecar. Source GLBs and published tile chunks
are never modified. Only single-sided exterior shell triangles that lie on an
actual foundation boundary segment and a convex-hull support plane are eligible.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from package_route_block import (Source, building_id, flat_nodes, json_bytes,
    local_path, write_json)

VERSION = 1
SCOPE = 'neighborhood-winding-corrections'
AREA_MIN_M2 = 0.10
VERTICAL_NORMAL_MAX_Y = 0.08
OUTWARD_DOT_MIN = 0.90
SUPPORT_PLANE_TOLERANCE_M = 0.04
BACKED_FACE_TOLERANCE_M = 0.20
SUPPORT_BOUNDARY_COVERAGE = 0.90
SUPPORT_INTERVAL_TOLERANCE_M = 0.025
COPLANAR_TOLERANCE_M = 0.005
OVERLAP_AREA_EPSILON_M2 = 1e-4
QUANTIZATION_M = 0.01
TOPOLOGY_WELD_TOLERANCE_M = 0.002
OVERLAP_BIN_M = 2.0
LOS_SAMPLE_WEIGHTS = np.array([[1/3, 1/3, 1/3], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
LOS_START_PAD_M = 1.0
LOS_MIN_START_M = 2.0
LOS_HIT_EPSILON_M = 0.005


def _point_key(point):
    return tuple(np.rint(np.asarray(point, dtype=float) / QUANTIZATION_M).astype(np.int64))


def _layer_of(node, mesh):
    return (node.get('extras', {}).get('layer') or mesh.get('extras', {}).get('layer')
            or node.get('name', '').split('__')[0])


def convex_hull(points):
    """Monotonic-chain hull for x/z pairs; returns a CCW ring without closure."""
    points = sorted(set(_point_key(p) for p in points))
    if len(points) < 3:
        return []

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [np.asarray(p, dtype=float) * QUANTIZATION_M for p in lower[:-1] + upper[:-1]]


def _merged_intervals(intervals, lo, hi, tolerance=SUPPORT_INTERVAL_TOLERANCE_M):
    pieces = []
    for a, b in intervals:
        a, b = sorted((float(a), float(b)))
        a, b = max(lo, a), min(hi, b)
        if b > a:
            pieces.append((a, b))
    if not pieces:
        return []
    pieces.sort()
    merged = [list(pieces[0])]
    for a, b in pieces[1:]:
        if a <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(p) for p in merged]


def build_support_edges(points, boundary_segments):
    """Return convex support planes clipped to actual foundation boundary spans.

    `boundary_segments` are exposed edges of the projected foundation triangle
    union. Restricting each convex-hull edge to these spans rejects concavities.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        return []
    result = []
    for edge_index, (a, b) in enumerate(zip(hull, hull[1:] + hull[:1])):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 0.05:
            continue
        tangent = delta / length
        outward = np.array([tangent[1], -tangent[0]], dtype=float)
        intervals = []
        for p, q in boundary_segments:
            if max(abs(float(np.dot(p-a, outward))), abs(float(np.dot(q-a, outward)))) > SUPPORT_INTERVAL_TOLERANCE_M:
                continue
            intervals.append((float(np.dot(p-a, tangent)), float(np.dot(q-a, tangent))))
        intervals = _merged_intervals(intervals, 0.0, length)
        if intervals:
            result.append({'edge': edge_index, 'a': a, 'tangent': tangent,
                           'outward': outward, 'length': length, 'intervals': intervals})
    return result


def _interval_coverage(lo, hi, intervals):
    if hi <= lo:
        return 0.0
    pieces = []
    for a, b in intervals:
        left, right = max(lo, a), min(hi, b)
        if right > left:
            pieces.append((left, right))
    if not pieces:
        return 0.0
    pieces.sort()
    start, end = pieces[0]
    total = 0.0
    for left, right in pieces[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end-start
            start, end = left, right
    return (total + end-start) / (hi-lo)


def _triangle_area(points):
    return float(np.linalg.norm(np.cross(points[1]-points[0], points[2]-points[0])) * 0.5)


def _polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly, dtype=float)
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))) * 0.5


def _triangle_overlap_area(subject, clip):
    """Convex triangle intersection area in (along-wall, height) coordinates."""
    subject = [np.asarray(p, dtype=float) for p in subject]
    clip = [np.asarray(p, dtype=float) for p in clip]
    signed = sum(clip[i][0]*clip[(i+1)%3][1] - clip[(i+1)%3][0]*clip[i][1] for i in range(3))
    orientation = 1.0 if signed >= 0 else -1.0
    poly = subject
    for i in range(3):
        a, b = clip[i], clip[(i+1)%3]
        edge = b-a

        def side(point):
            rel = point-a
            return orientation * (edge[0]*rel[1] - edge[1]*rel[0])

        output = []
        if not poly:
            break
        prev = poly[-1]
        prev_side = side(prev)
        for curr in poly:
            curr_side = side(curr)
            prev_in, curr_in = prev_side >= -1e-9, curr_side >= -1e-9
            if curr_in != prev_in:
                denom = prev_side-curr_side
                if abs(denom) > 1e-15:
                    output.append(prev + (curr-prev)*(prev_side/denom))
            if curr_in:
                output.append(curr)
            prev, prev_side = curr, curr_side
        poly = output
    return _polygon_area(poly)


def _world_positions(positions, matrix):
    return positions @ matrix[:3, :3].T + matrix[:3, 3]


def _foundation_supports(source):
    """Build per-owner support planes and exposed foundation boundary spans."""
    members = defaultdict(list)
    for node, matrix in flat_nodes(source.model):
        mesh = source.model['meshes'][node['mesh']]
        owner = building_id(node, mesh)
        if owner and _layer_of(node, mesh) == 'foundations':
            members[owner].append((node, matrix))

    support = {}
    for owner, group in members.items():
        points, seen_triangles = [], set()
        edge_counts = Counter()
        for node, matrix in group:
            mesh = source.model['meshes'][node['mesh']]
            for primitive in mesh.get('primitives', []):
                if primitive.get('mode', 4) != 4:
                    continue
                positions = _world_positions(source.array(primitive['attributes']['POSITION']), matrix)[:, [0, 2]]
                if 'indices' in primitive:
                    indices = source.array(primitive['indices']).reshape(-1)
                else:
                    indices = np.arange(len(positions), dtype=np.uint32)
                if len(indices) % 3:
                    continue
                for face in indices.reshape(-1, 3):
                    q = positions[face]
                    keys = tuple(sorted(_point_key(p) for p in q))
                    if keys in seen_triangles:
                        continue
                    seen_triangles.add(keys)
                    points.extend(q.tolist())
                    area2 = abs((q[1, 0]-q[0, 0])*(q[2, 1]-q[0, 1]) -
                                (q[1, 1]-q[0, 1])*(q[2, 0]-q[0, 0]))
                    if area2 < 1e-5:
                        continue
                    for p, qv in ((q[0], q[1]), (q[1], q[2]), (q[2], q[0])):
                        a, b = sorted((_point_key(p), _point_key(qv)))
                        if a != b:
                            edge_counts[(a, b)] += 1
        boundary = [(np.asarray(a, dtype=float)*QUANTIZATION_M,
                     np.asarray(b, dtype=float)*QUANTIZATION_M)
                    for (a, b), count in edge_counts.items() if count == 1]
        edges = build_support_edges(points, boundary)
        if edges:
            support[owner] = edges
    return support


def _support_match(points, normal, supports, max_plane_tolerance=SUPPORT_PLANE_TOLERANCE_M):
    """Return clear outward/inward shell side and its support edge, if any."""
    if abs(float(normal[1])) > VERTICAL_NORMAL_MAX_Y:
        return None
    projected = points[:, [0, 2]]
    for edge in supports:
        delta = projected-edge['a']
        distance = np.abs(delta @ edge['outward'])
        along = delta @ edge['tangent']
        if float(distance.max()) > max_plane_tolerance:
            continue
        if float(along.min()) < -SUPPORT_INTERVAL_TOLERANCE_M or float(along.max()) > edge['length'] + SUPPORT_INTERVAL_TOLERANCE_M:
            continue
        if _interval_coverage(float(along.min()), float(along.max()), edge['intervals']) < SUPPORT_BOUNDARY_COVERAGE:
            continue
        side = float(np.dot(normal[[0, 2]], edge['outward']))
        if abs(side) >= OUTWARD_DOT_MIN:
            return edge, side, float(np.mean(delta @ edge['outward']))
    return None


def _blocks_correction(wrong, other):
    """Whether an outward/backing surface already covers a proposed flip."""
    offset = other['plane_offset'] - wrong['plane_offset']
    coplanar = abs(offset) <= COPLANAR_TOLERANCE_M
    backed_outward = (other['side'] >= OUTWARD_DOT_MIN
        and COPLANAR_TOLERANCE_M < offset <= BACKED_FACE_TOLERANCE_M)
    if not coplanar and not backed_outward:
        return False
    if (wrong['bbox'][2] <= other['bbox'][0] or other['bbox'][2] <= wrong['bbox'][0]
            or wrong['bbox'][3] <= other['bbox'][1] or other['bbox'][3] <= wrong['bbox'][1]):
        return False
    return _triangle_overlap_area(wrong['polygon'], other['polygon']) > OVERLAP_AREA_EPSILON_M2


def _filter_overlapping(candidates):
    """Block only inward flips already covered by same-owner exterior faces.

    The blocker index includes small face fragments and near-support outward
    faces. Correctly oriented surfaces remain untouched.
    """
    bins = defaultdict(list)
    cell = OVERLAP_BIN_M
    for i, candidate in enumerate(candidates):
        x0, y0, x1, y1 = candidate['bbox']
        for x in range(math.floor(x0/cell), math.floor(x1/cell)+1):
            for y in range(math.floor(y0/cell), math.floor(y1/cell)+1):
                bins[(candidate['edge_id'], x, y)].append(i)
    excluded = set()
    for i, candidate in enumerate(candidates):
        if not candidate.get('eligible') or candidate['side'] >= -OUTWARD_DOT_MIN:
            continue
        x0, y0, x1, y1 = candidate['bbox']
        possible = set()
        for x in range(math.floor(x0/cell), math.floor(x1/cell)+1):
            for y in range(math.floor(y0/cell), math.floor(y1/cell)+1):
                possible.update(bins.get((candidate['edge_id'], x, y), ()))
        for j in possible:
            if i != j and _blocks_correction(candidate, candidates[j]):
                excluded.add(i)
    return excluded


def _closed_component_mask(triangles):
    """Mark triangles in closed, consistently wound shells after positional weld.

    UV/material seams commonly duplicate each triangle's vertices. Welding by
    world position restores edge incidence before connected closed-component
    detection. An enclosed-volume check prevents a doubled planar sheet from
    being mistaken for a solid.
    """
    triangles = np.asarray(triangles, dtype=float)
    count = len(triangles)
    if not count:
        return np.zeros(0, dtype=bool)
    quantized = np.rint(triangles.reshape(-1, 3) / TOPOLOGY_WELD_TOLERANCE_M).astype(np.int64)
    _, vertex_ids = np.unique(quantized, axis=0, return_inverse=True)
    faces = vertex_ids.reshape(-1, 3).astype(np.int64, copy=False)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edge_faces = np.tile(np.arange(count, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, edge_faces = edges[order], edge_faces[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1))+1]
    ends = np.r_[starts[1:], len(edges)]
    parent = np.arange(count, dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return int(value)

    for start, end in zip(starts, ends):
        faces_at_edge = edge_faces[start:end]
        if len(faces_at_edge) < 2:
            continue
        root = find(int(faces_at_edge[0]))
        for face in faces_at_edge[1:]:
            other = find(int(face))
            if other != root:
                parent[other] = root
    roots = np.fromiter((find(i) for i in range(count)), dtype=np.int64, count=count)
    closed_root = np.ones(count, dtype=bool)
    for start, end in zip(starts, ends):
        if end-start != 2:
            closed_root[find(int(edge_faces[start]))] = False

    root_ids, inverse = np.unique(roots, return_inverse=True)
    origins = np.zeros((len(root_ids), 3), dtype=float)
    np.add.at(origins, inverse, triangles.mean(axis=1))
    counts = np.bincount(inverse)
    origins /= counts[:, None]
    shifted = triangles-origins[inverse, None, :]
    signed_volume6 = np.einsum('ij,ij->i', shifted[:, 0],
                               np.cross(shifted[:, 1], shifted[:, 2]))
    volumes = np.zeros(len(root_ids), dtype=float)
    np.add.at(volumes, inverse, signed_volume6)
    good = np.array([closed_root[find(int(root))] for root in root_ids])
    good &= np.abs(volumes) > 1e-5
    return good[inverse]


def _closed_shell_face_keys(source, owner_nodes):
    """Return glTF mesh/primitive/triangle keys in closed owner shell components."""
    parts, keys = [], []
    model = source.model
    for node, matrix in owner_nodes:
        mesh_index = node['mesh']
        mesh = model['meshes'][mesh_index]
        layer = _layer_of(node, mesh)
        if layer not in ('facades', 'facade', 'buildings', 'roofs'):
            continue
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            if primitive.get('mode', 4) != 4:
                continue
            positions = _world_positions(source.array(primitive['attributes']['POSITION']), matrix)
            indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(positions), dtype=np.uint32)
            if len(indices) % 3:
                continue
            faces = indices.reshape(-1, 3)
            # Bound temporary triangles for very large source meshes.
            for start in range(0, len(faces), 65536):
                block = faces[start:start+65536]
                parts.append(positions[block].astype('<f4', copy=False))
                keys.extend((mesh_index, primitive_index, start+i) for i in range(len(block)))
    if not parts:
        return set()
    closed = _closed_component_mask(np.concatenate(parts, axis=0))
    return {key for key, is_closed in zip(keys, closed) if is_closed}


def _correction_map(candidates, blocked, closed_faces):
    result = defaultdict(set)
    for i, candidate in enumerate(candidates):
        face_key = (candidate['mesh'], candidate['primitive'], candidate['triangle'])
        if (i in blocked or not candidate.get('eligible')
                or candidate['side'] >= -OUTWARD_DOT_MIN or face_key in closed_faces
                or candidate.get('line_of_sight') is False):
            continue
        result[(candidate['mesh'], candidate['primitive'])].add(candidate['triangle'])
    return result


class TriangleBVH:
    """Low-overhead AABB tree for front-face ray blockers in one owner mesh set."""
    LEAF_SIZE = 16

    def __init__(self, triangles, double_sided, keys, owner=None):
        self.triangles = np.asarray(triangles, dtype=np.float32)
        self.double_sided = np.asarray(double_sided, dtype=bool)
        self.keys = keys
        self.owner = owner
        if not len(self.triangles):
            self.nodes, self.order = [], np.empty(0, dtype=np.int64)
            return
        self.minimum = self.triangles.min(axis=1)
        self.maximum = self.triangles.max(axis=1)
        self.centres = (self.minimum+self.maximum)*0.5
        self.order = np.arange(len(self.triangles), dtype=np.int64)
        self.nodes = []
        self._build(0, len(self.order))

    def _build(self, start, end):
        index = len(self.nodes)
        ids = self.order[start:end]
        low = self.minimum[ids].min(axis=0)
        high = self.maximum[ids].max(axis=0)
        self.nodes.append((low, high, -1, -1, start, end))
        if end-start <= self.LEAF_SIZE:
            return index
        points = self.centres[ids]
        axis = int(np.argmax(points.max(axis=0)-points.min(axis=0)))
        half = (end-start)//2
        order = np.argpartition(points[:, axis], half)
        self.order[start:end] = ids[order]
        mid = start+half
        left = self._build(start, mid)
        right = self._build(mid, end)
        self.nodes[index] = (low, high, left, right, -1, -1)
        return index

    @staticmethod
    def _ray_box(origin, direction, low, high, max_distance):
        near, far = 0.0, max_distance
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < low[axis]-1e-5 or origin[axis] > high[axis]+1e-5:
                    return False
                continue
            a = (low[axis]-origin[axis])/direction[axis]
            b = (high[axis]-origin[axis])/direction[axis]
            if a > b:
                a, b = b, a
            near, far = max(near, a), min(far, b)
            if far < near:
                return False
        return far >= 0 and near <= max_distance

    def ray_blocked(self, origin, direction, max_distance, skip_key=None):
        if not self.nodes:
            return False
        origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        stack = [0]
        while stack:
            node_index = stack.pop()
            low, high, left, right, start, end = self.nodes[node_index]
            if not self._ray_box(origin, direction, low, high, max_distance):
                continue
            if left >= 0:
                stack.extend((left, right))
                continue
            ids = self.order[start:end]
            triangles = self.triangles[ids]
            edge1 = triangles[:, 1]-triangles[:, 0]
            edge2 = triangles[:, 2]-triangles[:, 0]
            pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
            determinant = np.einsum('ij,ij->i', edge1, pvec)
            two_sided = self.double_sided[ids]
            valid = np.where(two_sided, np.abs(determinant)>1e-10, determinant>1e-10)
            if skip_key is not None:
                valid &= np.fromiter((self.keys[int(i)] != skip_key for i in ids), dtype=bool, count=len(ids))
            if not valid.any():
                continue
            inv = np.zeros_like(determinant)
            inv[valid] = 1.0/determinant[valid]
            tvec = origin-triangles[:, 0]
            u = np.einsum('ij,ij->i', tvec, pvec)*inv
            qvec = np.cross(tvec, edge1)
            v = qvec @ direction * inv
            distance = np.einsum('ij,ij->i', edge2, qvec)*inv
            hit = (valid & (u>=-1e-7) & (v>=-1e-7) & (u+v<=1+1e-7)
                   & (distance>1e-4) & (distance<=max_distance))
            if hit.any():
                return True
        return False


def _owner_occluder_bvh(source, owner, members):
    triangles, sidedness, keys = [], [], []
    model = source.model
    for node, matrix in members:
        mesh_index = node['mesh']
        mesh = model['meshes'][mesh_index]
        if _layer_of(node, mesh) not in ('facades', 'facade', 'buildings', 'roofs'):
            continue
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            if primitive.get('mode', 4) != 4:
                continue
            material = model.get('materials', [])[primitive.get('material', 0)]
            if material.get('alphaMode', 'OPAQUE') == 'BLEND':
                continue
            positions = _world_positions(source.array(primitive['attributes']['POSITION']), matrix)
            indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(positions), dtype=np.uint32)
            faces = indices.reshape(-1, 3)
            for start in range(0, len(faces), 65536):
                block = faces[start:start+65536]
                triangles.append(positions[block].astype('<f4', copy=False))
                sidedness.extend([bool(material.get('doubleSided', False))]*len(block))
                keys.extend((mesh_index, primitive_index, start+i) for i in range(len(block)))
    if not triangles:
        return TriangleBVH([], [], [], owner)
    return TriangleBVH(np.concatenate(triangles, axis=0), sidedness, keys, owner)


def _ray_start_distance(point, outward, bounds_en=None):
    if not bounds_en or len(bounds_en) != 4:
        return LOS_MIN_START_M
    x0, n0, x1, n1 = map(float, bounds_en)
    corners = np.array([[x0, -n0], [x0, -n1], [x1, -n0], [x1, -n1]])
    distance = float(np.max((corners-np.asarray(point)[[0, 2]]) @ outward))
    return max(LOS_MIN_START_M, distance+LOS_START_PAD_M)


def _candidate_visible(candidate, indices, bounds_en=None):
    """Require every inset sample to see the proposed front face before any owner."""
    points = candidate['points']
    outward = candidate['outward']
    key = (candidate['mesh'], candidate['primitive'], candidate['triangle'])
    for weights in LOS_SAMPLE_WEIGHTS:
        point = weights @ points
        start = _ray_start_distance(point, outward, bounds_en)
        origin = point.copy()
        origin[[0, 2]] += outward*start
        direction = np.array([-outward[0], 0.0, -outward[1]])
        target_distance = start+LOS_HIT_EPSILON_M
        for bvh in indices:
            skip = key if bvh.owner == candidate['owner'] else None
            if bvh.ray_blocked(origin, direction, target_distance, skip):
                return False
    return True


def _scan_cell(source, cell, support):
    model = source.model
    nodes = flat_nodes(model)
    mesh_uses = Counter(node.get('mesh') for node, _ in nodes)
    owner_nodes = defaultdict(list)
    for node, matrix in nodes:
        mesh = model['meshes'][node['mesh']]
        owner = building_id(node, mesh)
        if owner and _layer_of(node, mesh) in ('facades', 'facade', 'buildings', 'roofs'):
            owner_nodes[owner].append((node, matrix))
    closed_faces = set()
    for owner, members in owner_nodes.items():
        closed_faces.update(_closed_shell_face_keys(source, members))
    candidates = []
    stats = Counter()
    for node, matrix in nodes:
        mesh_index = node['mesh']
        mesh = model['meshes'][mesh_index]
        owner = building_id(node, mesh)
        layer = _layer_of(node, mesh)
        if not owner or layer not in ('facades', 'facade', 'buildings') or owner not in support:
            continue
        if mesh_uses[mesh_index] != 1:
            stats['shared_meshes_skipped'] += 1
            continue
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            if primitive.get('mode', 4) != 4:
                continue
            material_index = primitive.get('material', 0)
            material = model.get('materials', [])[material_index] if model.get('materials') else {}
            double_sided = material.get('doubleSided', False)
            if double_sided:
                stats['double_sided_primitives_excluded_from_correction'] += 1
            positions = _world_positions(source.array(primitive['attributes']['POSITION']), matrix)
            index_accessor = primitive.get('indices')
            indices = source.array(index_accessor).reshape(-1) if index_accessor is not None else np.arange(len(positions), dtype=np.uint32)
            if len(indices) % 3:
                continue
            faces = indices.reshape(-1, 3)
            for block_start in range(0, len(faces), 65536):
                block = faces[block_start:block_start+65536]
                xyz = positions[block].astype(float, copy=False)
                cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
                lengths = np.linalg.norm(cross, axis=1)
                areas = lengths * 0.5
                valid = (lengths > 1e-10) & (np.abs(cross[:, 1]) <= lengths*VERTICAL_NORMAL_MAX_Y)
                if not valid.any():
                    continue
                normals = np.zeros_like(cross)
                normals[valid] = cross[valid] / lengths[valid, None]
                assigned = np.zeros(len(block), dtype=bool)
                projected = xyz[:, :, [0, 2]]
                for edge in support[owner]:
                    delta = projected-edge['a']
                    distance = np.abs(delta @ edge['outward'])
                    along = delta @ edge['tangent']
                    near = (valid & ~assigned & (distance.max(axis=1) <= BACKED_FACE_TOLERANCE_M)
                            & (along.min(axis=1) >= -SUPPORT_INTERVAL_TOLERANCE_M)
                            & (along.max(axis=1) <= edge['length']+SUPPORT_INTERVAL_TOLERANCE_M))
                    side_values = normals[:, [0, 2]] @ edge['outward']
                    for local in np.flatnonzero(near):
                        coverage = _interval_coverage(float(along[local].min()),
                            float(along[local].max()), edge['intervals'])
                        if coverage < SUPPORT_BOUNDARY_COVERAGE:
                            continue
                        polygon = np.column_stack((along[local], xyz[local, :, 1]))
                        projected_area = _polygon_area(polygon)
                        if projected_area <= 1e-8:
                            continue
                        side = float(side_values[local])
                        if abs(side) < OUTWARD_DOT_MIN and areas[local] >= AREA_MIN_M2:
                            # Oblique, materialized vertical faces are usually
                            # reveals. They cannot be winding-corrected safely.
                            continue
                        triangle_index = block_start+int(local)
                        bbox = (float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                                float(polygon[:, 0].max()), float(polygon[:, 1].max()))
                        stats['support_shell_triangles'] += 1
                        candidates.append({'owner': owner, 'mesh': mesh_index, 'primitive': primitive_index,
                            'triangle': triangle_index, 'face': block[local].astype(int), 'layer': layer,
                            'edge_id': (owner, edge['edge']), 'side': side,
                            'plane_offset': float(np.mean(delta[local] @ edge['outward'])),
                            'polygon': polygon, 'bbox': bbox, 'area': float(areas[local]),
                            'points': xyz[local].astype(float), 'outward': edge['outward'].copy(),
                            'eligible': bool(not double_sided
                                             and distance[local].max() <= SUPPORT_PLANE_TOLERANCE_M
                                             and areas[local] >= AREA_MIN_M2
                                             and abs(side) >= OUTWARD_DOT_MIN)})
                        assigned[local] = True
                    if assigned.all():
                        break
    excluded = _filter_overlapping(candidates)
    stats['coplanar_overlap_triangles_excluded'] = len(excluded)
    stats['closed_shell_triangles_excluded'] = sum(
        candidate.get('eligible') and candidate['side'] < -OUTWARD_DOT_MIN
        and (candidate['mesh'], candidate['primitive'], candidate['triangle']) in closed_faces
        for candidate in candidates)
    potential = []
    for i, candidate in enumerate(candidates):
        face_key = (candidate['mesh'], candidate['primitive'], candidate['triangle'])
        if (i not in excluded and candidate.get('eligible')
                and candidate['side'] < -OUTWARD_DOT_MIN and face_key not in closed_faces):
            candidate['line_of_sight'] = True
            potential.append(candidate)
    stats['line_of_sight_candidates'] = len(potential)
    if potential:
        # Read and index one owner's geometry at a time. Every opaque neighbor
        # in this detail tile participates before a face can be retained.
        for owner, members in owner_nodes.items():
            bvh = _owner_occluder_bvh(source, owner, members)
            if bvh.nodes:
                stats['line_of_sight_owner_indexes'] += 1
                for candidate in potential:
                    if candidate.get('line_of_sight') and not _candidate_visible(candidate, [bvh], cell.get('boundsEN')):
                        candidate['line_of_sight'] = False
                        stats['line_of_sight_blocked_candidates'] += 1
            del bvh
    corrections = _correction_map(candidates, excluded, closed_faces)
    stats['corrected_triangles'] = sum(map(len, corrections.values()))
    mesh_records = []
    for (mesh_index, primitive_index), triangles in sorted(corrections.items()):
        primitive = model['meshes'][mesh_index]['primitives'][primitive_index]
        pos_accessor = model['accessors'][primitive['attributes']['POSITION']]
        count = model['accessors'][primitive['indices']]['count'] if 'indices' in primitive else pos_accessor['count']
        mesh_records.append({'mesh': int(mesh_index), 'primitive': int(primitive_index),
            'indexCount': int(count), 'vertexCount': int(pos_accessor['count']),
            'triangles': sorted(int(t) for t in triangles)})
    return {'id': cell['id'], 'contentHash': cell['contentHash'], 'records': mesh_records}, stats


def _unpack_checked(public, descriptor, output):
    digest = hashlib.sha256()
    size = 0
    for i, url in enumerate(descriptor['chunks']):
        if descriptor.get('format') != 'gzip-chunks' or url != f"/world-{descriptor['contentHash']}-{i:02}.bin":
            raise ValueError('Unpinned neighborhood chunk URL')
        path = local_path(public, url)
        with path.open('rb') as raw:
            while block := raw.read(1024*1024):
                digest.update(block)
        with gzip.open(path, 'rb') as source:
            while block := source.read(1024*1024):
                size += len(block)
                if size > descriptor['uncompressedBytes']:
                    raise ValueError('Neighborhood chunk exceeds declared length')
                output.write(block)
    output.flush()
    if size != descriptor['uncompressedBytes'] or digest.hexdigest()[:12] != descriptor['contentHash']:
        raise ValueError('Neighborhood chunk hash/length mismatch')


def _gpu_matches(public, cell, records, portable_source, scratch):
    gpu = cell.get('gpu')
    if not gpu or gpu.get('profile') != 's3tc-v1' or gpu.get('portableHash') != cell['contentHash']:
        return None
    path = scratch/'gpu.glb'
    with path.open('w+b') as stream:
        _unpack_checked(public, gpu, stream)
        source = Source(stream)
        for record in records:
            mesh_index = record['mesh']
            if mesh_index >= len(source.model.get('meshes', [])):
                return None
            mesh = source.model['meshes'][mesh_index]
            if record['primitive'] >= len(mesh.get('primitives', [])):
                return None
            primitive = mesh['primitives'][record['primitive']]
            source_mesh = portable_source.model['meshes'][mesh_index]
            source_primitive = source_mesh['primitives'][record['primitive']]
            gpu_indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else None
            source_indices = portable_source.array(source_primitive['indices']).reshape(-1) if 'indices' in source_primitive else None
            gpu_positions = source.array(primitive['attributes']['POSITION'])
            source_positions = portable_source.array(source_primitive['attributes']['POSITION'])
            if len(gpu_positions) != record['vertexCount'] or not np.array_equal(gpu_positions, source_positions):
                return None
            if ((gpu_indices is None) != (source_indices is None)
                    or (gpu_indices is not None and (len(gpu_indices) != record['indexCount']
                        or not np.array_equal(gpu_indices, source_indices)))):
                return None
            if ('NORMAL' in primitive['attributes']) != ('NORMAL' in source_primitive['attributes']):
                return None
            if 'NORMAL' in primitive['attributes']:
                if not np.array_equal(source.array(primitive['attributes']['NORMAL']),
                                      portable_source.array(source_primitive['attributes']['NORMAL'])):
                    return None
        return gpu['contentHash']


def generate(public, manifest_path='/compare/neighborhood-stream.json', verify_gpu=True,
             progress=True, manifest_override=None):
    """Return a complete per-cell sidecar for a verified neighborhood package."""
    public = Path(public).resolve()
    manifest_file = local_path(public, manifest_path) if manifest_override is None else None
    raw_manifest = manifest_file.read_bytes() if manifest_file is not None else None
    manifest = json.loads(raw_manifest) if manifest_override is None else manifest_override
    if manifest.get('scope') != 'neighborhood-stream' or manifest.get('version') != 1:
        raise ValueError('Completed neighborhood-stream version 1 manifest required')
    cells = manifest.get('cells', [])
    source_hash = manifest.get('sourceWorldHash')
    if not source_hash or not cells:
        raise ValueError('Neighborhood manifest lacks pinned source hash or detail cells')
    with tempfile.TemporaryDirectory(prefix='.winding-corrections-', dir=public) as td:
        scratch = Path(td)
        base_path = scratch/'base.glb'
        with base_path.open('w+b') as stream:
            _unpack_checked(public, manifest['base'], stream)
            base_source = Source(stream)
            support = _foundation_supports(base_source)
        out_cells = []
        total = Counter()
        for index, cell in enumerate(cells):
            if cell.get('sourceWorldHash') != source_hash:
                raise ValueError(f"Cell {cell.get('id')} belongs to a different source world")
            cell_path = scratch/'cell.glb'
            with cell_path.open('w+b') as stream:
                _unpack_checked(public, cell, stream)
                source = Source(stream)
                entry, stats = _scan_cell(source, cell, support)
                if verify_gpu and entry['records']:
                    gpu_hash = _gpu_matches(public, cell, entry['records'], source, scratch)
                    if gpu_hash:
                        entry['gpuContentHash'] = gpu_hash
            out_cells.append(entry)
            total.update(stats)
            total['cells'] += 1
            if progress:
                print(json.dumps({'phase':'winding-cell','cell':cell['id'],**stats}),flush=True)
        # Prevent publishing a sidecar based on a manifest changed during the scan.
        if manifest_file is not None and manifest_file.read_bytes() != raw_manifest:
            raise RuntimeError('Neighborhood manifest changed during winding analysis')
    return {'version': VERSION, 'scope': SCOPE, 'sourceWorldHash': source_hash,
            'cells': out_cells}, dict(total)


def build_candidate(public, manifest_path='/compare/neighborhood-stream.json', verify_gpu=True, progress=True):
    """Write an opt-in candidate archive without changing any active alias.

    Empty scans write no sidecar or descriptor, so normal viewers have nothing
    to fetch. Nonempty scans remain detached until a separate reviewed workflow
    explicitly opts into the candidate.
    """
    public = Path(public).resolve()
    manifest_file = local_path(public, manifest_path)
    before = manifest_file.read_bytes()
    manifest = json.loads(before)
    sidecar, stats = generate(public, manifest_path, verify_gpu, progress)
    triangle_count = sum(len(record['triangles']) for cell in sidecar['cells'] for record in cell['records'])
    if triangle_count == 0:
        return {'manifest': manifest_path, 'candidateManifest': None,
                'windingCorrections': None, 'cells': len(sidecar['cells']),
                'correctedTriangles': 0, **stats}
    sidecar_url = write_json(public, 'neighborhood-winding-corrections', sidecar)
    updated = {**manifest, 'windingCorrections': sidecar_url}
    if manifest_file.read_bytes() != before:
        raise RuntimeError('Neighborhood manifest changed before candidate publication')
    candidate_url = write_json(public, 'neighborhood-stream-winding-candidate', updated)
    return {'manifest': manifest_path, 'candidateManifest': candidate_url,
            'windingCorrections': sidecar_url, 'cells': len(sidecar['cells']),
            'correctedTriangles': triangle_count, **stats}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('public', type=Path)
    cli.add_argument('--manifest', default='/compare/neighborhood-stream.json')
    cli.add_argument('--skip-gpu-verification', action='store_true')
    args = cli.parse_args()
    print(json.dumps(build_candidate(args.public, args.manifest, not args.skip_gpu_verification), sort_keys=True))
