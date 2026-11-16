"""Trees the LiDAR measures off the streets that no street-tree inventory lists: in open ground (parks, gardens, lawns)
and in the unbuilt land between buildings (back gardens, setbacks, courtyards), where a city block's trees stand.

Every non-ground return outside the building outlines (kept 1.5 m clear of them) and inside the mapped open ground,
standing 2.5-45 m over the terrain, goes into a 1 m canopy-height raster. A crown is a local peak of the smoothed
raster at least 4 m tall, no nearer than 4 m to a taller peak; its spread is the area of the cells nearest to it that
stand at half its height or more. A peak within 3 m of a placed inventory tree is that tree. Each crown is drawn as the
inventory's broadleaf trees are (bark trunk and limbs, leaf cards), sized to the measured height and spread: the crown
top at the measured height. Nothing here refers to a place; the open ground comes from the area's mapped land use."""
import math
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter, maximum_filter
from shapely import contains_xy
from shapely.geometry import Point
from shapely.ops import unary_union

OPEN_GROUND = {'leisure': {'park', 'garden', 'dog_park', 'common', 'nature_reserve'},
               'landuse': {'grass', 'recreation_ground', 'meadow', 'village_green', 'forest'},
               'natural': {'grassland', 'scrub', 'wood', 'heath'}}

def open_ground(ways, boundary):
    """The mapped open ground (parks, gardens, lawns, woods) inside the boundary."""
    parts = [w['g'] for w in ways if w['g'].geom_type in ('Polygon', 'MultiPolygon')
             and any(w['tags'].get(k) in v for k, v in OPEN_GROUND.items())]
    return unary_union(parts).intersection(boundary) if parts else None

def yards(ctx):
    """The unbuilt land off the streets: the boundary less the building outlines (1.5 m clear) and the street plan."""
    plan = ctx.get('street_plan') or {}; y = ctx['boundary'].difference(ctx['all_buildings'].buffer(1.5))
    for k in ('road', 'sidewalk', 'cycle', 'paths'):
        if plan.get(k) is not None and not plan[k].is_empty: y = y.difference(plan[k])
    return y

def crowns(pts, elev, area, buildings, placed, cell=1., min_height=4., min_sep=4.):
    """(x, y, ground z, height, crown radius) of the crowns the returns over `area` show."""
    if area is None or area.is_empty: return []
    a, b, c, d = area.bounds
    m = (pts[:, 3] == 1) & (pts[:, 0] >= a) & (pts[:, 0] <= c) & (pts[:, 1] >= b) & (pts[:, 1] <= d)
    q = pts[m]
    if not len(q): return []
    clear = area.difference(buildings.buffer(1.5)) if buildings is not None else area
    q = q[contains_xy(clear, q[:, 0], q[:, 1])]
    if not len(q): return []
    g = np.asarray(elev(q[:, 0], q[:, 1]), float); h = q[:, 2] - g
    k = (h > 2.5) & (h < 45.); q, g, h = q[k], g[k], h[k]
    if not len(q): return []
    nx, ny = int((c - a) / cell) + 2, int((d - b) / cell) + 2
    ix, iy = ((q[:, 0] - a) / cell).astype(int), ((q[:, 1] - b) / cell).astype(int)
    chm = np.zeros((ny, nx), np.float32); np.maximum.at(chm, (iy, ix), h.astype(np.float32))
    sm = gaussian_filter(chm, 1.)
    peaks = (sm == maximum_filter(sm, size=int(2 * min_sep / cell) + 1)) & (sm >= min_height)
    py, px = np.nonzero(peaks)
    order = np.argsort(-sm[py, px]); kept = []
    placed = np.asarray(placed, float).reshape(-1, 2)
    for o in order:
        x, y = a + (px[o] + .5) * cell, b + (py[o] + .5) * cell
        if any(math.hypot(x - u, y - v) < min_sep for u, v, *_ in kept): continue
        if len(placed) and np.min(np.hypot(placed[:, 0] - x, placed[:, 1] - y)) < 3.: continue
        if not contains_xy(clear, x, y): continue
        kept.append((x, y, float(sm[py[o], px[o]])))
    if not kept: return []
    # spread: the cells at half a crown's height or more that lie nearest to it
    K = np.array([(x, y) for x, y, _ in kept]); H = np.array([t for _, _, t in kept])
    cy, cx = np.nonzero(chm > 0); cxy = np.c_[a + (cx + .5) * cell, b + (cy + .5) * cell]
    from scipy.spatial import cKDTree
    _, owner = cKDTree(K).query(cxy); tall = chm[cy, cx] >= .5 * H[owner]
    counts = np.bincount(owner[tall], minlength=len(K))
    out = []
    for i, (x, y, t) in enumerate(kept):
        r = min(8., max(1.5, math.sqrt(counts[i] * cell * cell / math.pi)))
        # a crown never reaches through a wall: it is held 0.3 m short of the nearest building (its returns over the roof
        # were never this tree's, and leaves drawn through the wall read as decals stuck on it)
        if buildings is not None and not buildings.is_empty: r = min(r, max(1., float(buildings.distance(Point(x, y))) - .3))
        out.append((x, y, float(elev(x, y)), t, r))
    return out

_TREE_INDEX = None


def measure_tree(ctx, x, y, others=None):
    """(height, crown radius) the LiDAR measures for a tree standing at (x, y): the non-ground returns 2.5-30 m over the
    terrain near it, off the buildings. Height: the 95th percentile within 3 m; spread: how far returns at half that
    height or more reach (90th percentile, within 7 m), held 0.3 m short of the nearest building. None when the returns
    show no tree there (an inventory point the LiDAR predates, or a sapling). `others`: the other trees' (x, y)."""
    global _TREE_INDEX
    from scipy.spatial import cKDTree
    if _TREE_INDEX is None:
        pts = ctx['pts']; q = pts[pts[:, 3] == 1]
        bld = ctx.get('all_buildings')
        if bld is not None and not bld.is_empty: q = q[~contains_xy(bld.buffer(1.), q[:, 0], q[:, 1])]
        h = q[:, 2] - np.asarray(ctx['elev'](q[:, 0], q[:, 1]), float); k = (h > 2.5) & (h < 30.)
        _TREE_INDEX = (cKDTree(q[k, :2]), q[k, :2], h[k])
    tree, xy, h = _TREE_INDEX
    near = tree.query_ball_point([x, y], 7.)
    if not near: return None
    near = np.asarray(near); sample_xy = xy[near]
    d = np.hypot(sample_xy[:, 0] - x, sample_xy[:, 1] - y); hh = h[near]
    # a street's trees stand a few metres apart: a return belongs to the planted tree nearest it, not to every tree
    # whose 7 m reach covers it (else each crown in a row spans its neighbours)
    others = [p for p in (others or []) if 0 < math.hypot(p[0] - x, p[1] - y) < 14.]
    if others:
        o = np.asarray(others, float); do = np.min(np.hypot(xy[near, 0, None] - o[:, 0], xy[near, 1, None] - o[:, 1]), axis=1)
        mine = d <= do; d, hh, sample_xy = d[mine], hh[mine], sample_xy[mine]
    core = hh[d <= 3.]
    if len(core) < 5: return None
    height = float(np.percentile(core, 95))
    wide = d[hh >= .5 * height]
    radius = float(np.clip(np.percentile(wide, 90) if len(wide) >= 5 else 2., 1.5, 7.))
    bld = ctx.get('all_buildings')
    if bld is not None and not bld.is_empty: radius = min(radius, max(1., float(bld.distance(Point(x, y))) - .3))
    # Tall returns concentrated on the wall-facing side can be facade returns
    # that survived the building buffer. Keep the normal LiDAR size unless
    # this source-neutral directional check rejects its upper tail.
    if bld is not None and not bld.is_empty:
        from shapely.ops import nearest_points
        from pipeline.tree_sizing import assess_lidar_height
        edge = nearest_points(Point(float(x), float(y)), bld)[1]
        wall = np.array([edge.x - x, edge.y - y], dtype=float)
        decision = assess_lidar_height(np.column_stack((sample_xy, hh)), [x, y], wall,
                                       float(np.hypot(edge.x - x, edge.y - y)))
        if decision['status'] == 'rejected-wall-facing-upper-tail': return None
    return height, radius


def draw_tree(put, cyl, x, y, z, height, radius, *, trunk_radius_m=None):
    """One broadleaf tree as the inventory's are drawn (bark trunk and limbs, leaf cards the viewer grows on the 'leaf'
    quads), standing on the ground at (x, y, z) with its crown top `height` above it and `radius` wide. The same position
    draws the same tree."""
    # a tree's leaf cards are gathered into one mesh per leaf material: put() per card held several small arrays each, and
    # an area of measured trees (hundreds of cards apiece) outgrew the machine's memory
    cards = {}
    def blade(a, b, width, key):
        a, b = np.array(a), np.array(b); v = b - a; n = np.cross(v, [0, 0, 1]); n = n / max(.001, np.linalg.norm(n)); mid = a + v * .55
        cards.setdefault(key, []).append(np.array([a, mid + n * width, b, mid - n * width, mid + [0, 0, width * .2]]))
    rng = np.random.default_rng(int(abs(x) * 1000 + abs(y) * 7) % (2 ** 32))
    vr = min(.42 * height, max(1.6, radius * 1.05))          # the crown's vertical half-depth
    centre = np.array([x, y, z + height - vr]); trunk_top = z + max(1.8, height - 1.6 * vr)
    trunk = min(.45, .06 + .018 * height) if trunk_radius_m is None else float(trunk_radius_m)
    cyl([x, y, z + .06], [x, y, trunk_top], trunk, 'vegetation', 'bark', 12)
    for k in range(7):
        ang = k * math.tau / 7; dirn = np.array([math.cos(ang), math.sin(ang), 0])
        tip = np.array([x, y, trunk_top]) + dirn * rng.uniform(.45, .8) * radius + [0, 0, rng.uniform(.4, .9) * vr]
        cyl([x, y, trunk_top], tip, min(.12, .03 + .006 * height), 'vegetation', 'bark', 6)
    n = int(min(520, max(180, 270 * (radius / 2.4) ** 2 * (vr / 2.8))))
    for k in range(n):
        dvec = rng.normal(size=3); dvec /= np.linalg.norm(dvec); r = rng.random() ** .3
        pos = centre + dvec * r * np.array([radius, radius, vr]); ang = rng.uniform(0, math.tau)
        end = pos + np.array([math.cos(ang), math.sin(ang), rng.uniform(-.4, .4)]) * .42
        blade(pos, end, .15, 'leaf' + str(1 + k % 3))
    quad = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    for key, vs in cards.items():
        put(trimesh.Trimesh(np.concatenate(vs), np.concatenate([quad + 5 * i for i in range(len(vs))]), process=False), 'vegetation', key)


def compile(ctx):
    """Draw the measured crowns of the open ground and the yards; returns their records. A yard keeps its crowns of 5 m
    or more, at most 1,500, the tallest first (a block's gardens hold many small shrubs the LiDAR also peaks on)."""
    park = open_ground(ctx['ways'], ctx['boundary']); yard = yards(ctx)
    # a peak at a tree the facade readings already placed in front of a wall (facade_detail) is that tree
    placed = [item['position'] for item in ctx.get('tree_placements', [])] + [t[:2] for t in ctx.get('front_trees') or []]
    found = crowns(ctx['pts'], ctx['elev'], park, ctx.get('all_buildings'), placed)
    in_park = [(x, y) for x, y, *_ in found]
    extra = [c for c in crowns(ctx['pts'], ctx['elev'], yard, ctx.get('all_buildings'), placed + in_park) if c[3] >= 5.]
    extra.sort(key=lambda c: -c[3]); found = [(*c, 'park') for c in found] + [(*c, 'yard') for c in extra[:1500]]
    # trees over the sidewalks the street-tree inventory does not list (privately planted, or planted since): the street
    # photographs show them hiding whole facades, and only the inventory's trees stood there
    walk = (ctx.get('street_plan') or {}).get('sidewalk')
    if walk is not None and not walk.is_empty:
        taken = placed + in_park + [(c[0], c[1]) for c in extra[:1500]]
        found += [(*c, 'sidewalk') for c in crowns(ctx['pts'], ctx['elev'], walk, ctx.get('all_buildings'), taken) if c[3] >= 5.]
    put, cyl = ctx['put'], ctx['cylinder']; records = []
    for x, y, z, height, radius, where in found:
        draw_tree(put, cyl, x, y, z, height, radius)
        records.append({'position': [round(x, 2), round(y, 2)], 'heightM': round(height, 1), 'crownRadiusM': round(radius, 1),
                        'source': 'LiDAR canopy peak in ' + {'park': 'mapped open ground', 'sidewalk': 'a sidewalk (no inventory tree there)'}.get(where, 'unbuilt land off the streets'), 'status': 'measured-crown, species unknown'})
    ctx['counts']['lidarCanopyTrees'] = len(records); ctx['counts']['lidarYardTrees'] = sum(r['source'].endswith('streets') for r in records); ctx['counts']['lidarSidewalkTrees'] = sum('sidewalk' in r['source'] for r in records)
    return records
