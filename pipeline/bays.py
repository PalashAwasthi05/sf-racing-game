"""Bay windows as projecting volumes. A photographed bay (a reading's `bays`: its extent along the wall and up it, and
how far it stands out) was drawn as its windows pushed out one by one, each with its own cheeks: a stack of window
boxes. Here a bay is one volume standing out from the wall, clad like the wall it projects from, its windows set in
its faces:
  angled  a front parallel to the wall and two sides at 45 degrees (the commonest bay on a city street);
  square  a front and two sides square to the wall, when the bay is too narrow for 45-degree sides;
  curved  a shallow arc of glazed facets (a round bay), when the reading names it.
A reading may name the shape, the angle of an angled bay's sides, a trim course at each storey and corbels under a
bay that starts above the ground.
A rectified photograph shows an angled bay's side windows foreshortened inside the side's extent along the wall, so a
window is set on the face its extent falls on (split across two faces if it spans both). A square bay's sides face
along the wall and show no window in a square-on photograph. The bay is capped with a cornice in its windows' trim and
closed underneath when it starts above the ground. Location-free: the shape follows the measured extent alone."""
import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

FACE_M, GLASS_IN_M, FRAME_M = .12, .08, .06


def corners(x0, x1, p, shape=None, angle=45.):
    """(shape, corner points) of a bay over [x0, x1] standing p out, left to right in (along the wall, out from it)."""
    run = p / math.tan(math.radians(float(np.clip(angle, 20., 80.))))
    if shape == 'curved':
        c = x1 - x0; R = (c * c / 4 + p * p) / (2 * p); xm = (x0 + x1) / 2; a = math.asin(min(1., c / 2 / R))
        return 'curved', [(xm + R * math.sin(t), p - R + R * math.cos(t)) for t in np.linspace(-a, a, 7)]
    if shape == 'square' or (shape is None and x1 - x0 < 2 * run + .8) or x1 - x0 < 2 * run + .4:
        return 'square', [(x0, 0.), (x0, p), (x1, p), (x1, 0.)]
    return 'angled', [(x0, 0.), (x0 + run, p), (x1 - run, p), (x1, 0.)]


def faces(x0, x1, p, shape=None, angle=45.):
    """(shape, faces) of a bay over [x0, x1] standing p out: each face (xa, da, xb, db) in (along the wall, out from it),
    running left to right around the bay."""
    shape, cs = corners(x0, x1, p, shape, angle)
    return shape, [(a[0], a[1], b[0], b[1]) for a, b in zip(cs, cs[1:])]


def plan(x0, x1, p, shape, angle=45.):
    """The bay's outline in plan (along the wall, out from it), closed along the wall."""
    return Polygon(corners(x0, x1, p, shape, angle)[1])


def draw(point, put, layer, detail, bay, openings, cladding, trim):
    """Draw one bay. `point(x, z, depth)` maps the wall frame to the world (x along the wall, z up, depth out from the
    wall's face); `put(mesh, key, material)` adds a mesh. `bay` = {'x0','x1','z0','z1','p'} in metres of the wall frame;
    `openings` are the grammar's openings inside it (rect in metres, style). Returns the number of windows set."""
    import trimesh
    x0, x1, z0, z1, p = bay['x0'], bay['x1'], bay['z0'], bay['z1'], bay['p']
    angle = bay.get('sideAngleDeg') or 45.
    shape, fs = faces(x0, x1, p, bay.get('shape'), angle)

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
            if st.get('sash') == 'double-hung':
                # a sash window: the meeting rail, and the panes of each sash
                mr = hz1 + (hz2 - hz1) * float(st.get('meetingRail', .5)); slab(box(hs1, mr - .03, hs2, mr + .03), frame_of, .02, GLASS_IN_M, detail, frame)
                na, nd = (max(1, int(v)) for v in (st.get('lites') or [1, 1]))
                for za, zb in ((hz1, mr), (mr, hz2)):
                    for k in range(1, na): slab(box(hs1 + (hs2 - hs1) * k / na - .011, za, hs1 + (hs2 - hs1) * k / na + .011, zb), frame_of, .05, GLASS_IN_M, detail, frame)
                    for k in range(1, nd): slab(box(hs1, za + (zb - za) * k / nd - .011, hs2, za + (zb - za) * k / nd + .011), frame_of, .05, GLASS_IN_M, detail, frame)
            elif st.get('sash') == 'casement' and hs2 - hs1 > .5:
                slab(box((hs1 + hs2) / 2 - .03, hz1, (hs1 + hs2) / 2 + .03, hz2), frame_of, .02, GLASS_IN_M, detail, frame)
            slab(box(hs1 - .03, hz1 - .06, hs2 + .03, hz1), frame_of, -.05, .1, detail, frame)     # sill
            windows += 1
    outline = plan(x0, x1, p, shape, angle)

    def lid(poly, za, zb, key, material):
        """A plan polygon (along, out) from height za to zb."""
        poly = poly.intersection(box(x0 - 2., 0., x1 + 2., p + 2.))      # never behind the wall's face
        for q in ([poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))):
            if q.is_empty or q.area < 1e-4: continue
            mesh = trimesh.creation.extrude_polygon(q, height=zb - za, engine='earcut')
            mesh.vertices = np.array([point(x, za + h, d) for x, d, h in mesh.vertices]); mesh.fix_normals(); put(mesh, key, material)

    lid(outline, z1, z1 + .14, layer, cladding)
    if bay.get('cap', 'cornice') == 'cornice':
        lid(outline.buffer(.06, join_style=2), z1 + .14, z1 + .22, detail, trim)                  # cornice: a frieze,
        lid(outline.buffer(.16, join_style=2), z1 + .22, z1 + .34, detail, trim)                  # and its cap
    if z0 > .5:
        lid(outline, z0 - .14, z0, layer, cladding)                                               # underside
        lid(outline.buffer(.05, join_style=2), z0 - .22, z0 - .1, detail, trim)                   # its moulding
    for z in bay.get('storeys', []):
        # a trim course round the bay at each floor line inside it
        if z0 + .4 < z < z1 - .4: lid(outline.buffer(.035, join_style=2), z - .08, z + .08, detail, trim)
    if bay.get('corbel') and z0 > 1.5:
        # brackets under a bay that starts above the ground, at its front corners
        _, cs = corners(x0, x1, p, shape, angle)
        for xa, da in cs[1:-1] if shape != 'curved' else cs[1:-1:2]:
            for k, (h0, h1, dd) in enumerate(((z0 - .7, z0 - .45, .25), (z0 - .45, z0 - .22, .5), (z0 - .22, z0 - .1, .8))):
                lid(box(xa - .08, 0., xa + .08, max(.05, dd * da)), h0, h1, detail, trim)   # deeper toward the top
    return windows
