"""Dormers and chimneys on a roof, from a given massing: what stands on a roof that the roof form does not hold.

A massing entry's `roofFeatures` (areas/ring/massing-overrides.json, read from the photographs and checked against
the LiDAR's bumps above the roof) lists them, in the world's local frame, heights above the building's foot:
  {"kind": "dormer", "xy": [x, y] (the dormer's centre), "facing": [dx, dy] (the way its window looks), "widthM",
   "depthM", "eaveM", "ridgeM" (its own eave and ridge), "roof": "gable" | "hip" | "shed", "window": true}
  {"kind": "chimney", "xy": [x, y], "widthM", "depthM", "topM", "angleDeg" (optional: its long side's bearing from east;
   default the part's long axis), "material": "brick" | "wall"}
A dormer is a small house on the slope: walls from inside the main roof up to its eave, its own roof form
(pipeline/roof_forms.py) with its ridge running back into the main roof, a window in its front. A chimney is a stack
from inside the roof to its top, with a projecting cap. Location-free: the entries carry the facts.
`build` returns (mesh, role) pairs; role is 'roof', 'wall', 'trim', 'brick' or 'glass'."""
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point
from general_world.roof_planes import roof_solid, evaluate as roof_evaluate
from pipeline import roof_forms


def _rect(c, u, w, d):
    """Rectangle centred at c, width w along unit u, depth d along u's left normal."""
    n = np.array([-u[1], u[0]]); c = np.asarray(c, float)
    return Polygon([c - u * w / 2 - n * d / 2, c + u * w / 2 - n * d / 2, c + u * w / 2 + n * d / 2, c - u * w / 2 + n * d / 2])


def roof_z(xy, patches, base_roof):
    """The roof's height at points (n, 2): the given roof form where a patch covers them, else the flat roof."""
    xy = np.atleast_2d(np.asarray(xy, float)); z = np.full(len(xy), np.nan)
    for p in patches:
        g = p['geometry'].buffer(.05); inside = np.array([g.contains(Point(q)) for q in xy])
        if inside.any(): z[inside] = np.fmax(z[inside], roof_evaluate(p['model'], xy[inside]))
    return np.where(np.isfinite(z), z, float(base_roof))


def _box(c, u, w, h, t, z0):
    """A thin slab on a wall plane: centre c (x, y), along unit u, width w, height h, thickness t (outward), bottom z0."""
    n = np.array([u[1], -u[0]])                    # u's right normal
    m = trimesh.creation.box(extents=[w, t, h])
    R = np.eye(4); R[:2, 0] = u; R[:2, 1] = n; R[2, 2] = 1; R[:3, 3] = [c[0], c[1], z0 + h / 2]
    m.apply_transform(R); return m


def dormer(f, z0, patches, base_roof, part=None):
    out = []; c = np.asarray(f['xy'], float); face = np.asarray(f['facing'], float); face /= max(np.linalg.norm(face), 1e-6)
    u = np.array([face[1], -face[0]])               # along the dormer's front, so that its depth runs back from `face`
    w = float(np.clip(f.get('widthM', 2.), .8, 8.)); d = float(np.clip(f.get('depthM', 2.5), .8, 10.))
    eave = z0 + float(f['eaveM']); ridge = z0 + float(f.get('ridgeM', f['eaveM'] + w * .35))
    # a dormer's front wall stands tall enough for its window (1.4 m): the LiDAR's bump sits toward its ridge, so a front
    # placed from it can come out a sliver; the front moves down the slope (the back stays), never past the eave line
    for s in np.arange(0., 4.01, .1):
        front = c + face * (d / 2 + s)
        if part is not None and part.exterior.distance(Point(front)) < .4 and s > 0: s -= .1; break
        if eave - float(roof_z(front[None], patches, base_roof)[0]) >= 1.4: break
    s = max(float(s), 0.); c = c + face * s / 2; d = d + s
    r = _rect(c, u, w, d); corners = np.asarray(r.exterior.coords)[:4]
    bottom = float(roof_z(corners, patches, base_roof).min()) - .15
    kind = f.get('roof', 'gable') if f.get('roof', 'gable') in ('gable', 'hip', 'shed') else 'gable'
    spec = {'kind': kind, 'ridgeM': ridge - z0, 'overhangM': .15,
            'along': 'long' if d >= w else 'short',          # the ridge runs back from the front, into the main roof
            'lowSide': [float(face[0]), float(face[1])]}     # a shed falls toward the front
    patches_d, _ = roof_forms.build(r, spec, eave, eave - z0)
    reach = d / 2                                   # the front wall's distance from the centre (an overhang moves it out)
    if patches_d:
        p = patches_d[0]; s = roof_solid(p['geometry'], p['model'], bottom)
        reach = float(max((np.asarray(q) - c) @ face for q in p['geometry'].exterior.coords))
        up = s.face_normals[:, 2] > .05; side = np.abs(s.face_normals[:, 2]) <= .05
        out += [(s.submesh([np.where(up)[0]], append=True, repair=False), 'roof'), (s.submesh([np.where(side)[0]], append=True, repair=False), 'wall')]
    else:
        m = trimesh.creation.extrude_polygon(r, height=eave - bottom); m.apply_translation([0, 0, bottom]); out.append((m, 'wall'))
    if f.get('window', True):
        front = c + face * reach                     # the front wall
        sill = float(roof_z(front[None], patches, base_roof)[0]) + .2; top = eave - .15
        if top - sill > .4:
            ww = min(w * .6, 1.4); hh = top - sill; b = .07
            out.append((_box(front + face * .01, u, ww, hh, .02, sill), 'glass'))
            for dx, bw, z, bh in ((0., ww + 2 * b, sill - b, b), (0., ww + 2 * b, top, b), (-(ww + b) / 2, b, sill, hh), ((ww + b) / 2, b, sill, hh)):
                out.append((_box(front + u * dx + face * .03, u, bw, bh, .06, z), 'trim'))      # the frame round it
    return out


def chimney(f, z0, patches, base_roof, axis):
    c = np.asarray(f['xy'], float); a = np.radians(f['angleDeg']) if f.get('angleDeg') is not None else axis
    # a stack: a LiDAR bump round one takes in the wall it rises from, so its size is held to a chimney's (1.6 m at most)
    u = np.array([np.cos(a), np.sin(a)]); w = float(np.clip(f.get('widthM', .8), .3, 1.6)); d = float(np.clip(f.get('depthM', .6), .3, 1.6))
    r = _rect(c, u, w, d); corners = np.asarray(r.exterior.coords)[:4]
    bottom = float(roof_z(corners, patches, base_roof).min()) - .2; top = z0 + float(f['topM'])
    if top - bottom < .3: return []
    role = 'brick' if f.get('material', 'brick') == 'brick' else 'wall'
    m = trimesh.creation.extrude_polygon(r, height=top - bottom); m.apply_translation([0, 0, bottom])
    cap = trimesh.creation.extrude_polygon(r.buffer(.06, join_style=2), height=.1); cap.apply_translation([0, 0, top])
    return [(m, role), (cap, role)]


def build(features, part, z0, base_roof, patches):
    """Meshes for the roof features standing on this part (their centre inside its outline)."""
    out = []; mrr = np.asarray(part.minimum_rotated_rectangle.exterior.coords)
    e = [mrr[1] - mrr[0], mrr[2] - mrr[1]]; axis = float(np.arctan2(*max(e, key=np.linalg.norm)[::-1]))
    for f in features:
        if not part.contains(Point(f['xy'])): continue
        if f.get('kind') == 'dormer' and f.get('eaveM') is not None and f.get('facing'): out += dormer(f, z0, patches, base_roof, part)
        elif f.get('kind') == 'chimney' and f.get('topM') is not None: out += chimney(f, z0, patches, base_roof, axis)
    return out
