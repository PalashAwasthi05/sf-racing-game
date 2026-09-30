"""Parts added to the body: windscreen and its frame, roll hoops, mirrors and the cabin.

(+z is the car's right side in this frame; the steering wheel is on the left, -z.)

Each builder returns a list of (part, finish, mesh). Thin parts are explicit sweeps and patches; rounded solids (mirror
caps, seats, dash) are implicit and meshed finely. Parts on the car's left are mirrored to the right. Units: metres,
+x forward, +y up, +z to the car's right.
"""
import numpy as np
from mesher import mesh_solid, Mesh, merge
from shapes import sweep, rounded_rect, patch, mirror_z, both_sides_z, box3, capsule, ellipsoid, catmull
from roadster import smin, smax


# ---- windscreen ------------------------------------------------------------------------------------------------------
def screen_point(u, v, lift=0.):
    """Point on the windscreen: u across (-1 right .. 1 left), v up the glass (0 cowl .. 1 header)."""
    xb, yb, zb = .862 - .07 * u * u, .853 + .02 * u * u, .665 * u
    xt, yt, zt = .232 + .06 * u * u, 1.198 - .03 * u * u, .548 * u
    x = xb + (xt - xb) * v
    y = yb + (yt - yb) * v
    z = zb + (zt - zb) * v
    # the glass bows outward across its height; normal of the mean plane is (sin a, cos a) with a = rake
    bow = .016 * np.sin(np.pi * v) * (1 - .35 * u * u) + lift
    nx, ny = .49, .872
    return x + bow * nx, y + bow * ny, z


def windscreen():
    parts = []
    glass = patch(lambda u, v: screen_point(u, v), 48, 26, (-.955, .955), (.05, .95))
    parts.append(('screen', 'glass', glass))
    # black ceramic border round the edge (four strips)
    for ur, vr in (((-1, 1), (0, .05)), ((-1, 1), (.95, 1)), ((-1, -.955), (.05, .95)), ((.955, 1), (.05, .95))):
        parts.append(('screen', 'black_gloss', patch(lambda u, v: screen_point(u, v, -.0005), 30 if ur[1] - ur[0] > 1 else 4, 4 if vr[1] - vr[0] < .1 else 24, ur, vr)))
    # frame: body-coloured pillars and header swept along the glass edge, standing just outside and above it
    edge = []
    for v in np.linspace(-.04, .97, 30):
        edge.append((-1, v))
    for t in np.linspace(0, 1, 7)[1:-1]:           # rounded top corner
        a = t * np.pi / 2
        edge.append((-1 + .06 * (1 - np.cos(a)), .97 + .03 * np.sin(a)))
    for u in np.linspace(-.94, .94, 36):
        edge.append((u, 1.0))
    for t in np.linspace(0, 1, 7)[1:-1]:
        a = t * np.pi / 2
        edge.append((1 - .06 * (1 - np.sin(a)), 1.0 - .03 * (1 - np.cos(a))))
    for v in np.linspace(.97, -.04, 30):
        edge.append((1, v))
    edge = np.array(edge)
    C = np.array([screen_point(u, v, .006) for u, v in edge])
    # push the centreline outward in the glass plane, away from the glass centre
    centre = np.array(screen_point(0, .5))
    out = C - centre
    nrm = np.array([.49, .872, 0])
    out -= (out @ nrm)[:, None] * nrm
    out /= np.linalg.norm(out, axis=1)[:, None]
    C = C + out * .016
    prof = rounded_rect(.052, .030, .012, 5)
    frame = sweep(C, prof, lambda i, p: nrm, caps=True)
    parts.append(('frame', 'paint', frame))
    # inner trim of the frame (black) and the interior mirror
    inner = sweep(C - out * .012 - nrm * .012, rounded_rect(.04, .016, .006, 4), lambda i, p: nrm, caps=True)
    parts.append(('frame', 'black_satin', inner))
    mirror = lambda x, y, z: smin(box3(x, y, z, (.19, 1.12, 0), (.012, .032, .115), .01),
                                   capsule(x, y, z, (.215, 1.17, 0), (.20, 1.15, 0), .009), .01)
    V, T = mesh_solid(mirror, (.15, 1.07, -.14), (.25, 1.2, .14), .004, tolerance=.0002)
    parts.append(('screen', 'black_satin', Mesh(V, T).with_normals(mirror)))
    return parts


# ---- roll hoops and the cover behind them ---------------------------------------------------------------------------------
def hoops():
    parts = []
    for zc in (.345, -.345):
        # an inverted U in a plane leaning back 8 degrees, rising from the deck
        pts = []
        for a in np.linspace(0, np.pi, 40):
            zz = zc + .135 * np.cos(a)
            yy = .955 + .205 * np.sin(a) ** .55
            pts.append((-.655 - (yy - .955) * .14, yy, zz))
        C = np.array(pts)
        C = np.concatenate([C[:1] + np.array([0, -.06, 0]), C, C[-1:] + np.array([0, -.06, 0])])
        prof = rounded_rect(.060, .105, .022, 6)
        hoop = sweep(C, prof, lambda i, p: np.array([1., 0, 0]), caps=True)
        parts.append(('hoop', 'hoop', hoop))
        # black mesh filling the arch, set back from the hoop's front face
        def fill(u, v, zc=zc):
            a = np.pi * (u + 1) / 2
            zz = zc + .115 * np.cos(a) * v
            yy = .96 + (.185 * np.sin(a) ** .55) * v
            return -.668 - (yy - .955) * .14, yy, zz
        parts.append(('hoop', 'black_matte', patch(fill, 24, 8, (-1, 1), (0, 1), both_sides=True)))
    return parts


# ---- mirrors ---------------------------------------------------------------------------------------------------------------
def mirrors():
    def cap(x, y, z):
        e = ellipsoid(x, y, z, (.262, .905, .928), (.115, .062, .085))
        back = .188 - x                       # flat face carrying the glass
        return smax(e, back, .012)

    def stalk(x, y, z):
        return capsule(x, y, z, (.35, .862, .80), (.285, .885, .905), .014)
    V, T = mesh_solid(cap, (.12, .82, .82), (.4, .99, 1.04), .004, tolerance=.00035)
    capm = Mesh(V, T).with_normals(cap)
    V, T = mesh_solid(stalk, (.24, .82, .76), (.40, .93, .93), .004, tolerance=.0002)
    stm = Mesh(V, T).with_normals(stalk)
    glass = patch(lambda u, v: (np.full_like(u, .1865), .905 + (v - .5) * .096 * np.sqrt(1 - .55 * u * u), .928 + .07 * u),
                  12, 8, (-1, 1), (0, 1))
    glass.N = np.tile([-1., 0, 0], (len(glass.V), 1))
    glass.T = glass.T[:, ::-1]
    return [('mirror', 'paint', both_sides_z(capm)), ('mirror', 'black_satin', both_sides_z(stm)),
            ('mirror', 'mirror', both_sides_z(glass))]


# ---- cabin -------------------------------------------------------------------------------------------------------------------
def seat_solid(zc):
    def f(x, y, z):
        # cushion
        cush = box3(x, y, z, (-.08, .355, zc), (.24, .05, .215), .04)
        # backrest leaning back 22 degrees about its foot at (-.33, .38)
        ang = np.deg2rad(22)
        u = (x + .33) * np.cos(ang) + (y - .38) * np.sin(ang)
        w = -(x + .33) * np.sin(ang) + (y - .38) * np.cos(ang)
        back = box3(u, w, z, (-.045, .33, zc), (.05, .33, .215), .04)
        head = box3(u, w, z, (-.04, .66, zc), (.045, .07, .13), .04)
        # side bolsters on cushion and back
        bol = np.minimum(capsule(x, y, z, (.12, .43, zc + .19), (-.28, .41, zc + .19), .055),
                         capsule(x, y, z, (.12, .43, zc - .19), (-.28, .41, zc - .19), .055))
        bb = np.minimum(capsule(u, w, z, (-.02, .08, zc + .19), (-.02, .55, zc + .17), .06),
                        capsule(u, w, z, (-.02, .08, zc - .19), (-.02, .55, zc - .17), .06))
        s = smin(cush, back, .05)
        s = smin(s, head, .06)
        s = smin(s, bol, .04)
        return smin(s, bb, .05)
    return f


def cabin():
    parts = []
    seat = seat_solid(.36)
    V, T = mesh_solid(seat, (-.72, .27, .08), (.28, 1.14, .64), .006, tolerance=.001)
    seatm = Mesh(V, T).with_normals(seat)
    parts.append(('seat', 'cabin_soft', both_sides_z(seatm)))

    def dash(x, y, z):
        za = np.abs(z)
        top = box3(x, y, z, (.53, .70, 0), (.17, .13, .70), .05)
        # instrument hood over the gauges, in front of the left seat
        hood = box3(x, y, z, (.42, .83, -.36), (.10, .035, .17), .03)
        tunnel = box3(x, y, z, (.18, .42, 0), (.42, .13, .11), .04)
        s = smin(top, hood, .04)
        s = smin(s, tunnel, .06)
        return np.maximum(s, za - .74)
    V, T = mesh_solid(dash, (-.3, .28, -.76), (.75, .9, .76), .007, tolerance=.0004)
    parts.append(('dash', 'cabin_trim', Mesh(V, T).with_normals(dash)))
    # steering wheel: rim, three spokes and the boss, in front of the left seat
    c = np.array([.265, .725, -.36])       # left-hand drive: +z is the car's right side
    n = np.array([np.cos(np.deg2rad(25)), -np.sin(np.deg2rad(25)), 0])      # column axis: forward and down into the dash
    e1 = np.array([0, 0, 1.])
    e2 = np.cross(n, e1)
    ring = np.array([c + .185 * (np.cos(a) * e1 + np.sin(a) * e2) for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)])
    prof = [(.016 * np.cos(a), .013 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    rim = sweep(np.concatenate([ring, ring[:1]]), prof, lambda i, p: n, caps=False)
    parts.append(('wheel', 'cabin', rim))

    def boss(x, y, z):
        d = np.stack([x - c[0], y - c[1], z - c[2]])
        along = np.tensordot(n, d, 1)
        rad = np.sqrt(np.maximum((d * d).sum(0) - along * along, 0))
        hub = np.maximum(rad - .06, np.abs(along + .01) - .03)
        sp = []
        for a in (0, np.pi, -np.pi / 2):
            dirv = np.cos(a) * e1 + np.sin(a) * e2
            end = c + .175 * dirv
            sp.append(capsule(x, y, z, c, end, .014))
        s = hub
        for q in sp:
            s = smin(s, q, .02)
        col = capsule(x, y, z, c + .02 * n, c + .30 * n, .03)
        return smin(s, col, .02)
    V, T = mesh_solid(boss, (c[0] - .25, c[1] - .25, c[2] - .22), (c[0] + .3, c[1] + .3, c[2] + .22), .005, tolerance=.0003)
    parts.append(('wheel', 'cabin_trim', Mesh(V, T).with_normals(boss)))
    return parts


def all_parts():
    return (windscreen() + hoops() + mirrors() + cabin() + exhaust() + diffuser_fins() + grille_slats() + tail_leds()
            + headlamp_internals() + wipers() + door_handles() + spoiler())


# ---- rear details ------------------------------------------------------------------------------------------------------------
def exhaust():
    """Two round tailpipes in the diffuser: a rolled outer lip and a darker inner pipe."""
    import roadster as R
    from wheels import lathe
    parts = []
    r = R.TIP_R - .006
    prof = [(r - .006, -.20), (r - .006, -.004), (r - .003, 0), (r + .002, -.002), (r + .004, -.012), (r + .004, -.20)]
    inner = [(r - .018, -.20), (r - .018, -.012), (r - .010, -.006), (r - .006, -.004)]
    # mouths flush with the diffuser face
    mouth = float(R._tail(R.TIP_Y)) + .004
    for zs in (R.TIP_Z, -R.TIP_Z):
        for pr, fin in ((prof, 'steel'), (inner, 'black_matte')):
            m = lathe(pr[::-1] if fin == 'black_matte' else pr, segments=48)
            # lathe axis is z: turn it to point along -x and move it to the tip position
            V = m.V.copy()
            # lathe (r cos, r sin, along) -> car (-along, r sin, r cos), a rotation: the pipe runs forward into the car
            # from its mouth at the back of the diffuser
            m.V = np.stack([mouth - V[:, 2], V[:, 1] + R.TIP_Y, V[:, 0] + zs], 1)
            N = m.N.copy()
            m.N = np.stack([-N[:, 2], N[:, 1], N[:, 0]], 1)
            parts.append(('exhaust', fin, m))
    return parts


def diffuser_fins():
    def f(x, y, z):
        s = None
        for zc in (.13, -.13, .47, -.47):
            b = box3(x, y, z, (-2.015, .258, zc), (.055, .052, .005), .003)
            s = b if s is None else np.minimum(s, b)
        return s
    V, T = mesh_solid(f, (-2.12, .12, -.5), (-1.94, .31, .5), .003, tolerance=.0002)
    return [('diffuser', 'black_satin', Mesh(V, T).with_normals(f))]


def grille_slats():
    def f(x, y, z):
        s = None
        for xc in np.linspace(-1.158, -1.092, 5):
            b = box3(x, y, z, (xc, .955, 0), (.0035, .012, .24), .002)
            s = b if s is None else np.minimum(s, b)
        return s
    V, T = mesh_solid(f, (-1.18, .93, -.26), (-1.07, .98, .26), .003, tolerance=.0002)
    return [('grille', 'black_satin', Mesh(V, T).with_normals(f))]


def _onto(f, P, depth=0., direction=None):
    """Points carried along `direction` (default: toward the car's centre line at their height) until they meet the
    surface f = 0, then moved `depth` inward along the surface normal."""
    from mesher import ray_hit, normals
    P = np.atleast_2d(np.asarray(P, float))
    if direction is None:
        out = []
        for p in P:
            d = -np.array([p[0], 0, p[2]])
            out.append(ray_hit(f, p[None], d, far=2.0)[0])
        Q = np.array(out)
    else:
        Q = ray_hit(f, P, direction, far=2.0)
    return Q - depth * normals(f, Q)


def tail_leds():
    """Four stacked LED bars in each tail lamp, following the lamp's taper, 12 mm behind the smoked lens."""
    import roadster as R
    parts = []
    for k in range(4):
        y0 = .709 + k * .0155
        zs = np.linspace(.53 + .018 * k, .86, 26)
        # the bars run across the rear face and turn forward round the corner with the lamp
        P = np.stack([np.full_like(zs, -2.4), np.full_like(zs, y0 + .012 * (zs - .53)), zs], 1)
        # across the rear face cast forward; round the corner cast inward, so each point lands on its own part of the lens
        C = np.array([_onto(R.skin, p[None], .012, (1, 0, -.9 * max(0, (p[2] - .70) / .16)))[0] for p in P])
        prof = rounded_rect(.0045, .0045, .0015, 2)
        bar = sweep(C, prof, lambda i, p: np.array([0, 1., 0]), caps=True)
        parts.append(('taillamp', 'tail_led', both_sides_z(bar)))
    return parts


def headlamp_internals():
    """Projector module with a chrome bezel, four daytime lights in a square beside it, and a light bar along the foot."""
    import roadster as R
    from wheels import lathe
    parts = []
    # the lamp's inner floor: 60 mm under the lens; modules stand on it facing forward
    centre = _onto(R.skin, [[2.5, .655, .74]], .045, (-1, 0, 0))[0]
    proj = lathe([(0, .012), (.026, .012), (.030, .004), (.030, -.05)], segments=40, sharp=(1,))
    bezel = lathe([(.030, .010), (.040, .006), (.042, -.02)], segments=40)
    for m, fin in ((proj, 'lamp_black'), (bezel, 'lamp_chrome')):
        V = m.V.copy()
        m.V = np.stack([centre[0] + V[:, 2], centre[1] + V[:, 1], centre[2] + V[:, 0]], 1)
        m.N = np.stack([m.N[:, 2], m.N[:, 1], m.N[:, 0]], 1)
        m.T = m.T[:, ::-1]
        parts.append(('headlamp', fin, both_sides_z(m)))
    # four points of light: small rounded squares on the inner half of the lamp
    dots = []
    for dy in (-.028, .028):
        for dz in (-.028, .028):
            p = _onto(R.skin, [[2.5, .655 + dy, .60 + dz]], .02, (-1, 0, 0))[0]
            dots.append(p)
    def dotf(x, y, z):
        s = None
        for p in dots:
            b = box3(x, y, z, p, (.004, .009, .009), .003)
            s = b if s is None else np.minimum(s, b)
        return s
    lo = np.min(dots, 0) - .03
    hi = np.max(dots, 0) + .03
    V, T = mesh_solid(dotf, lo, hi, .0015, tolerance=.0001)
    parts.append(('headlamp', 'drl', both_sides_z(Mesh(V, T).with_normals(dotf))))
    # light bar along the foot of the lamp
    zs = np.linspace(.54, .83, 20)
    P = np.stack([np.full_like(zs, 2.5), .572 + .03 * (zs - .54), zs], 1)
    C = _onto(R.skin, P, .01, (-1, 0, 0))
    parts.append(('headlamp', 'drl', both_sides_z(sweep(C, rounded_rect(.004, .006, .0015, 2), lambda i, p: np.array([0, 1., 0])))))
    return parts


def wipers():
    parts = []
    for z0, z1 in ((.05, .56), (-.52, -.02)):
        a = np.array(screen_point(z0 / .665, .06, .012))
        b = np.array(screen_point(z1 / .6, .10, .012))
        f = lambda x, y, z, a=a, b=b: capsule(x, y, z, a, b, .006)
        lo, hi = np.minimum(a, b) - .02, np.maximum(a, b) + .02
        V, T = mesh_solid(f, lo, hi, .002, tolerance=.0004)
        parts.append(('wiper', 'black_satin', Mesh(V, T).with_normals(f)))
    return parts


def door_handles():
    import roadster as R
    # a pull handle pressed into the skin: about half its section stands proud of the door
    a = _onto(R.skin, [[-.14, .712, 1.2]], .001, (0, 0, -1))[0]
    b = _onto(R.skin, [[-.39, .716, 1.2]], .001, (0, 0, -1))[0]
    f = lambda x, y, z: capsule(x, y, z, a, b, .0095)
    V, T = mesh_solid(f, np.minimum(a, b) - .03, np.maximum(a, b) + .03, .002, tolerance=.0001)
    return [('handle', 'paint', both_sides_z(Mesh(V, T).with_normals(f)))]


def spoiler():
    """The third brake light: a thin bar on the band's top wall, tucked under the spoiler's trailing edge (the lip
    itself is part of the body)."""
    import roadster as R
    zs = np.linspace(-.24, .24, 25)
    y = .7765
    C = np.array([(float(R._tail(y)) + float(R._TAIL_PLAN(abs(z))) + .012, y, z) for z in zs])
    return [('spoiler', 'tail_led', sweep(C, rounded_rect(.004, .003, .001, 2), lambda i, p: np.array([0, 1., 0])))]
