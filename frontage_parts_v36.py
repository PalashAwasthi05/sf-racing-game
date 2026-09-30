"""Building-part outlines of a built candidate, for the frontage readers' occlusion masks.

A split outline holds several buildings (building_parts.split): a low building on the corner and a taller one behind it
share one source footprint. The readers took every footprint holding a wall as the wall's own building, so the low
building standing in front of the tall one's face was never an occluder, and its brick was read as the tall block's
colour. The parts' own outlines are the plan of their foundation slabs (every part has one: `foundations__<part id>`
meshes), read here from the GLB without loading anything else. Carries no location.
Usage (inspection): frontage_parts_v36.py <glb> [part id ...]"""
import json, mmap, re, struct, sys
import numpy as np


def part_outlines(glb_path):
    """{building part id: shapely polygon} from the foundation meshes' plan (scene metres, x east, y north)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    f = open(glb_path, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    ln = struct.unpack('<I', mm[12:16])[0]; j = json.loads(mm[20:20 + ln]); off = 20 + ln + 8
    CT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}; NC = {'SCALAR': 1, 'VEC3': 3}
    def acc(i):
        a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]; n = NC[a['type']]
        arr = np.frombuffer(mm, CT[a['componentType']], count=a['count'] * n, offset=off + bv.get('byteOffset', 0) + a.get('byteOffset', 0))
        return arr.reshape(a['count'], n) if n > 1 else arr
    def mat(node):
        if 'matrix' in node: return np.array(node['matrix'], float).reshape(4, 4).T
        T = np.eye(4)
        if 'rotation' in node:
            x, y, z, w = node['rotation']; T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        if 'scale' in node: T[:3, :3] = T[:3, :3] @ np.diag(node['scale'])
        if 'translation' in node: T[:3, 3] = node['translation']
        return T
    parent = {c: i for i, n in enumerate(j['nodes']) for c in n.get('children', [])}
    def world(i):
        M = mat(j['nodes'][i]); p = parent.get(i)
        return world(p) @ M if p is not None else M
    mesh_world = {}
    for i, n in enumerate(j['nodes']):
        if 'mesh' in n: mesh_world.setdefault(n['mesh'], world(i))
    Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float); tris = {}
    for mi, mesh in enumerate(j['meshes']):
        m = re.match(r'foundations__(.+?)___', mesh.get('name', ''))
        if not m: continue
        M = mesh_world.get(mi, np.eye(4))
        for prim in mesh['primitives']:
            pos = (acc(prim['attributes']['POSITION']).astype(np.float64) @ M[:3, :3].T + M[:3, 3]) @ Y2Z.T
            idx = acc(prim['indices']).astype(np.int64) if 'indices' in prim else np.arange(len(pos))
            tris.setdefault(m.group(1), []).append(pos[idx.reshape(-1, 3)][:, :, :2])
    out = {}
    for pid, ts in tris.items():
        T = np.concatenate(ts); e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]; area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
        polys = [Polygon(t) for t in T[area > 1e-4]]
        if polys: out[pid] = unary_union([p.buffer(.01, join_style=2) for p in polys]).buffer(-.01, join_style=2)
    return out


if __name__ == '__main__':
    o = part_outlines(sys.argv[1])
    for pid in (sys.argv[2:] or sorted(o)[:10]):
        g = o.get(pid); print(pid, None if g is None else (g.geom_type, round(g.area, 1), [round(v, 1) for v in g.bounds]))


def own_mass(wall, occluders, parts, own_footprints):
    """The mass that counts as the wall's own building for the occlusion test: every footprint holding the wall (as
    frontage_evidence_v34.own_footprints), less the other parts of the same split outline, so a lower or taller part
    standing between camera and wall hides it. Map and city outlines disagree by metres, so outline area outside every
    part stays the wall's own (a strip 1-2 m in front of a low front is its own eaves, not another building). Other
    parts are shrunk by half a metre: a split boundary is uncertain by about a raster cell."""
    from shapely.ops import unary_union
    base = own_footprints(wall, occluders)
    pid = wall['buildingId']; src = re.sub(r'_p\d+$', '', pid)
    others = [g for q, g in parts.items() if q != pid and re.sub(r'_p\d+$', '', q) == src]
    if base is None or not others: return base
    return base.difference(unary_union(others).buffer(-.5, join_style=2))


_RAISED = {}


def raised_surface(surf, wall, parts, margin_m=1.):
    """The building surface raster with every other part of the wall's split outline raised to its own top (the 98th
    percentile of its returns, plus 1 m): as an occluder a part stands to its full height. Parapets, copings and ridges
    are thin, and the raster's cells read below them (with 0.3 m, a low building's stepped parapet still showed above
    the mask and its brick was read as the tall block behind). The wall's own part and every unsplit building keep the raster as measured."""
    import shapely
    pid = wall['buildingId']; src = re.sub(r'_p\d+$', '', pid)
    key = (id(surf), pid)
    if key in _RAISED: return _RAISED[key]
    top, (x0, y1, st) = surf; out = top
    for q, g in parts.items():
        if q == pid or re.sub(r'_p\d+$', '', q) != src: continue
        b = g.bounds; c0, c1 = max(0, int((b[0] - x0) // st)), min(top.shape[1], int((b[2] - x0) // st) + 1)
        r0, r1 = max(0, int((y1 - b[3]) // st)), min(top.shape[0], int((y1 - b[1]) // st) + 1)
        if c0 >= c1 or r0 >= r1: continue
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        inside = shapely.contains_xy(g, x0 + (cc + .5) * st, y1 - (rr + .5) * st)
        vals = top[r0:r1, c0:c1][inside]; vals = vals[np.isfinite(vals)]
        if len(vals) < 3: continue
        if out is top: out = top.copy()
        block = out[r0:r1, c0:c1]; block[inside] = np.fmax(block[inside], float(np.percentile(vals, 98)) + margin_m)
    _RAISED[key] = (out, (x0, y1, st)) if out is not top else surf
    return _RAISED[key]
