"""Bay windows as projecting volumes. A photographed bay (a reading's `bays`: its extent along the wall and up it, and
how far it stands out) was drawn as its windows pushed out one by one, each with its own cheeks: a stack of window
boxes. Here a bay is one volume standing out from the wall, clad like the wall it projects from, its windows set in
its faces:
  angled  a front parallel to the wall and two sides at 45 degrees (the commonest bay on a city street);
  square  a front and two sides square to the wall, when the bay is too narrow for 45-degree sides.
A rectified photograph shows an angled bay's side windows foreshortened inside the side's extent along the wall, so a
window is set on the face its extent falls on (split across two faces if it spans both). A square bay's sides face
along the wall and show no window in a square-on photograph. The bay is capped with a cornice in its windows' trim and
closed underneath when it starts above the ground. Location-free: the shape follows the measured extent alone."""
import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

FACE_M, GLASS_IN_M, FRAME_M = .12, .08, .06


def faces(x0, x1, p):
    """(shape, faces) of a bay over [x0, x1] standing p out: each face (xa, da, xb, db) in (along the wall, out from it),
    running left to right around the bay."""
    if x1 - x0 >= 2 * p + .8:
        return 'angled', [(x0, 0., x0 + p, p), (x0 + p, p, x1 - p, p), (x1 - p, p, x1, 0.)]
    return 'square', [(x0, 0., x0, p), (x0, p, x1, p), (x1, p, x1, 0.)]


def plan(x0, x1, p, shape):
    """The bay's outline in plan (along the wall, out from it), closed along the wall."""
    if shape == 'angled': return Polygon([(x0, 0.), (x0 + p, p), (x1 - p, p), (x1, 0.)])
    return Polygon([(x0, 0.), (x0, p), (x1, p), (x1, 0.)])


def draw(point, put, layer, detail, bay, openings, cladding, trim):
    """Draw one bay. `point(x, z, depth)` maps the wall frame to the world (x along the wall, z up, depth out from the
    wall's face); `put(mesh, key, material)` adds a mesh. `bay` = {'x0','x1','z0','z1','p'} in metres of the wall frame;
    `openings` are the grammar's openings inside it (rect in metres, style). Returns the number of windows set."""
    import trimesh
    x0, x1, z0, z1, p = bay['x0'], bay['x1'], bay['z0'], bay['z1'], bay['p']
    shape, fs = faces(x0, x1, p)

    def slab(poly, frame_of, o0, o1, key, material):
        """A polygon in a face's own (s, z) frame, from o0 to o1 inward of the face (negative: standing out of it)."""
        for q in ([poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))):
            if q.is_empty or q.area < 1e-4: continue
            mesh = trimesh.creation.extrude_polygon(q, height=o1 - o0, engine='earcut')
            mesh.vertices = np.array([frame_of(s, z, o0 + h) for s, z, h in mesh.vertices]); mesh.fix_normals(); put(mesh, key, material)

    windows = 0
    for xa, da, xb, db in fs:
        L = math.hypot(xb - xa, db - da)
        if L < .05: continue
        fx, fd = (xb - xa) / L, (db - da) / L; nx_, nd_ = -fd, fx           # along the face, and out of it
        def frame_of(s, z, o, xa=xa, da=da, fx=fx, fd=fd, nx_=nx_, nd_=nd_):
            return point(xa + fx * s - nx_ * o, z, da + fd * s - nd_ * o)
        holes = []
        lo, hi = sorted((xa, xb))
        for op in openings:
            ox1, oy1, ox2, oy2 = op['rect']; a, b = max(ox1, lo), min(ox2, hi); za, zb = max(oy1, z0 + .05), min(oy2, z1 - .05)
            if abs(fx) < 1e-6 or b - a <= 0 or zb - za < .3: continue
            s1, s2 = sorted(((a - xa) / fx, (b - xa) / fx))
            if s2 - s1 < .3: continue
            holes.append((box(max(s1, .06), za, min(s2, L - .06), zb), op.get('style', {})))
        wall = box(0., z0, L, z1).difference(unary_union([h for h, _ in holes])) if holes else box(0., z0, L, z1)
        slab(wall, frame_of, 0., FACE_M, layer, cladding)
        for h, st in holes:
            hs1, hz1, hs2, hz2 = h.bounds; frame = st.get('frameMaterial', trim)
            slab(h, frame_of, GLASS_IN_M, GLASS_IN_M + .02, detail, st.get('glassMaterial', 'glass'))
            ring = h.difference(h.buffer(-FRAME_M, join_style=2))
            slab(ring, frame_of, 0., GLASS_IN_M + .02, detail, frame)
            for f in st.get('mullionFractions', []):
                slab(box(hs1 + (hs2 - hs1) * f - .02, hz1, hs1 + (hs2 - hs1) * f + .02, hz2), frame_of, .04, GLASS_IN_M, detail, frame)
            for f in st.get('transomFractions', []):
                slab(box(hs1, hz1 + (hz2 - hz1) * f - .02, hs2, hz1 + (hz2 - hz1) * f + .02), frame_of, .04, GLASS_IN_M, detail, frame)
            slab(box(hs1 - .03, hz1 - .06, hs2 + .03, hz1), frame_of, -.05, .1, detail, frame)     # sill
            windows += 1
    outline = plan(x0, x1, p, shape)

    def lid(poly, za, zb, key, material):
        """A plan polygon (along, out) from height za to zb."""
        poly = poly.intersection(box(x0 - 2., 0., x1 + 2., p + 2.))      # never behind the wall's face
        for q in ([poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))):
            if q.is_empty or q.area < 1e-4: continue
            mesh = trimesh.creation.extrude_polygon(q, height=zb - za, engine='earcut')
            mesh.vertices = np.array([point(x, za + h, d) for x, d, h in mesh.vertices]); mesh.fix_normals(); put(mesh, key, material)

    lid(outline, z1, z1 + .14, layer, cladding)
    lid(outline.buffer(.1, join_style=2), z1 + .14, z1 + .26, detail, trim)                       # cornice
    if z0 > .5: lid(outline, z0 - .14, z0, layer, cladding)                                       # underside
    return windows
