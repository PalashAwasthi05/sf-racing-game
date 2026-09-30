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
# Door mirror housing: a rounded, slightly wedge-shaped face (taller outboard) carrying the glass, domed forward to a
# blunt nose; it sits clear of the belt on a slim arm that rises from the door's upper front corner.
_MC = np.array([.205, .945, .905])        # centre of the glass face (x is the face plane)
_MA, _MB, _MD = .100, .049, .088          # half width (z), half height at the centre (y), depth forward (x)


def _mirror_face(w, v):
    """Normalised radius in the face: 1 on the housing outline (a superellipse, taller toward the outboard end)."""
    b = _MB * (1 + .16 * np.clip(w / _MA, -1, 1))
    vv = v - .004 * (w / _MA)                          # the outline's top rises a little outboard
    return (np.abs(w / _MA) ** 3.2 + np.abs(vv / b) ** 3.2) ** (1 / 3.2)


def _mirror_cap(x, y, z):
    w, v = z - _MC[2], y - _MC[1]
    u = np.maximum(x - _MC[0], 0) / _MD
    r = np.sqrt(_mirror_face(w, v) ** 2 + u ** 2.4)
    e = (r - 1) * .045
    return smax(e, _MC[0] - x, .005)


def _mirror_arm(x, y, z):
    a = capsule(x, y, z, (.312, .872, .793), (.262, .905, .845), .0085)
    b = capsule(x, y, z, (.322, .866, .789), (.300, .869, .791), .012)      # the arm's foot on the door
    return smin(a, b, .012)


def mirrors():
    V, T = mesh_solid(_mirror_cap, (.18, .87, .78), (.32, 1.02, 1.03), .003, tolerance=.0002)
    capm = Mesh(V, T).with_normals(_mirror_cap)
    from mesher import split
    # the housing's rear face is dark round the glass, up to where it rolls into the painted shell
    def frame(x, y, z):
        return x - _MC[0] - .004
    rim, body = split(capm, frame, _mirror_cap)
    V, T = mesh_solid(_mirror_arm, (.22, .83, .75), (.39, .94, .88), .003, tolerance=.0002)
    arm = Mesh(V, T).with_normals(_mirror_arm)

    def gl(u, v):
        a = np.pi * u
        cw, sv = np.cos(a) * _MA, np.sin(a) * _MB
        k = .92 / np.maximum(_mirror_face(cw, sv), 1e-9)       # scale of this direction that meets the 90% outline
        return np.full_like(u, _MC[0] - .003), _MC[1] + sv * k * v, _MC[2] + cw * k * v
    glass = patch(gl, 48, 6, (-1, 1), (0, 1))
    glass.N = np.tile([-1., 0, 0], (len(glass.V), 1))
    fn = glass.face_normals()
    if (fn[:, 0] > 0).sum() > len(fn) / 2:
        glass.T = glass.T[:, ::-1]
    return [('mirror', 'paint', both_sides_z(body)), ('mirror', 'black_gloss', both_sides_z(rim)),
            ('mirror', 'black_satin', both_sides_z(arm)), ('mirror', 'mirror', both_sides_z(glass))]


# ---- cabin -------------------------------------------------------------------------------------------------------------------
SEAT_FOOT = (-.25, .365)              # where the backrest meets the cushion (x, y)
SEAT_LEAN = np.deg2rad(15)            # backrest rake


def _seat_frame(x, y):
    """Coordinates along (w, up the backrest) and through (u, forward from it) the backrest."""
    c, s = np.cos(SEAT_LEAN), np.sin(SEAT_LEAN)
    dx, dy = x - SEAT_FOOT[0], y - SEAT_FOOT[1]
    return dx * c + dy * s, -dx * s + dy * c


def seat_solid(zc):
    """A sports bucket with an integrated headrest: cushion and backrest with raised side bolsters, the head section
    narrower at the neck and wider at the top."""
    def f(x, y, z):
        dz = z - zc
        cush = box3(x, y, z, (.0, .345, zc), (.235, .045, .19), .035)
        u, w = _seat_frame(x, y)
        back = box3(u, w, z, (.04, .30, zc), (.045, .30, .19), .035)
        # the head section: waisted at the neck
        head = smin(box3(u, w, z, (.04, .695, zc), (.04, .06, .125), .035),
                    box3(u, w, z, (.04, .60, zc), (.036, .07, .095), .03), .035)     # waisted at the neck
        head = smax(head, -box3(u, w, z, (.05, .625, zc), (.06, .016, .05), .012), .006)   # the belt slot
        bol = np.minimum(capsule(x, y, z, (.20, .415, zc + .175), (-.21, .40, zc + .175), .05),
                         capsule(x, y, z, (.20, .415, zc - .175), (-.21, .40, zc - .175), .05))
        bb = np.minimum(capsule(u, w, z, (.075, .06, zc + .17), (.075, .52, zc + .15), .055),
                        capsule(u, w, z, (.075, .06, zc - .17), (.075, .52, zc - .15), .055))
        s = smin(cush, back, .04)
        s = smin(s, head, .03)
        s = smin(s, bol, .035)
        return smin(s, bb, .04)
    return f


def smoothstep_(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _dash(x, y, z):
    za = np.abs(z)
    # the dash: its top runs from the screen base back to a rounded brow over the cockpit
    top = box3(x, y, z, (.62, .64, 0), (.22, .15, .76), .06)
    brow = capsule(x, y, z, (.44, .76, -.70), (.44, .76, .70), .045)
    s = smin(top, brow, .05)
    # instrument hood over the gauges in front of the driver
    hood = box3(x, y, z, (.47, .815, -.36), (.075, .035, .165), .03)
    s = smin(s, hood, .03)
    s = smax(s, -box3(x, y, z, (.40, .79, -.36), (.07, .03, .15), .02), .01)          # the hood's open face
    # centre console: a sloping face under the screen, running back between the seats as the tunnel
    fx, fy = .46, .66
    cu = (x - fx) * -.6 + (y - fy) * .8          # the console's face looks back and up at the occupants
    console = np.maximum(np.maximum(cu, za - .115), np.maximum(np.maximum(y - .70, .40 - x), np.maximum(.32 - y, x - .80)))
    tunnel = box3(x, y, z, (.02, .43, 0), (.45, .13, .11), .04)
    s = smin(s, smin(console, tunnel, .05), .05)
    return np.maximum(s, za - .745)


def cabin():
    from mesher import split
    parts = []
    for zc in (.36, -.36):
        seat = seat_solid(zc)
        V, T = mesh_solid(seat, (-.62, .27, zc - .29), (.30, 1.18, zc + .29), .004, tolerance=.0005)
        m = Mesh(V, T).with_normals(seat)
        # cloth centre panels on cushion and backrest, leather bolsters and head section
        def centre(x, y, z, zc=zc):
            u, w = _seat_frame(x, y)
            front = np.where(w > .1, np.maximum(.04 - u, w - .56), .345 - y)     # backrest front / cushion top
            return np.maximum(np.abs(z - zc) - .115, front)
        soft, leather = split(m, centre, seat)
        parts += [('seat', 'cabin_soft', soft), ('seat', 'cabin', leather)]
    from mesher import mesh_solid_sharp, autosmooth
    V, T = mesh_solid_sharp(_dash, (-.45, .28, -.77), (.86, .92, .77), .005, tolerance=.0004)
    parts.append(('dash', 'cabin_trim', autosmooth(Mesh(V, T), _dash)))
    parts += _instruments() + _steering_wheel()
    return parts


def _disc(c, n, r, depth=0., segments=40):
    """A flat disc of radius r centred at c facing along n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    e1 = np.cross(n, [0, 1., 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, n)
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    V = np.concatenate([[c], c + r * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)])
    T = np.array([[0, 1 + k, 1 + (k + 1) % segments] for k in range(segments)])
    m = Mesh(V, T, np.tile(n, (len(V), 1)))
    if (m.face_normals() @ n).mean() < 0:
        m.T = m.T[:, ::-1]
    return m


def _ring(c, n, r, t, segments=40):
    n = np.asarray(n, float) / np.linalg.norm(n)
    e1 = np.cross(n, [0, 1., 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, n)
    C = np.array([c + r * (np.cos(a) * e1 + np.sin(a) * e2) for a in np.linspace(0, 2 * np.pi, segments + 1)])
    prof = [(t * np.cos(a), t * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    return sweep(C, prof, lambda i, p: n, caps=False)


def _instruments():
    """Three round dials under the hood, the centre screen, round vents and the gear lever."""
    parts = []
    face = np.array([-1., .25, 0])                     # dials look back and a little up at the driver
    for dz, r in ((-.105, .042), (0, .05), (.105, .042)):
        c = np.array([.448, .792 + (.004 if dz == 0 else -.004), -.36 + dz])
        parts.append(('dash', 'black_gloss', _disc(c, face, r)))
        parts.append(('dash', 'lamp_chrome', _ring(c - .002 * face, face, r + .002, .0022)))
        # the needle and tick ring read as a pale arc on each dial
        parts.append(('dash', 'drl', _ring(c - .0015 * face, face, r * .82, .0009)))
    # centre screen on the sloping console face
    nrm = np.array([-.6, .8, 0])
    tang = np.array([.8, .6, 0])
    c0 = np.array([.46, .66, 0]) + .003 * nrm + .005 * tang
    sq = patch(lambda u, v: tuple(c0[k] + u * .095 * np.array([0, 0, 1.])[k] + v * .055 * tang[k] for k in range(3)), 2, 2, (-1, 1), (-1, 1))
    sq.N = np.tile(nrm, (len(sq.V), 1))
    if (sq.face_normals() @ nrm).mean() < 0:
        sq.T = sq.T[:, ::-1]
    parts.append(('dash', 'black_gloss', sq))
    # round vents: two in the centre above the screen, one at each end of the dash
    for c in ((.43, .705, .085), (.43, .705, -.085), (.40, .74, .64), (.40, .74, -.64)):
        c = np.array(c)
        parts.append(('dash', 'lamp_chrome', _ring(c, (-1, 0, 0), .034, .004)))
        parts.append(('dash', 'black_matte', _disc(c + np.array([.004, 0, 0]), (-1, 0, 0), .033)))
    # gear lever
    knob = lambda x, y, z: smin(capsule(x, y, z, (.12, .54, 0), (.10, .62, 0), .008), ellipsoid(x, y, z, (.10, .635, 0), (.026, .022, .022)), .01)
    V, T = mesh_solid(knob, (.05, .50, -.05), (.16, .68, .05), .003, tolerance=.0002)
    parts.append(('dash', 'cabin', Mesh(V, T).with_normals(knob)))
    return parts


def _steering_wheel():
    """A sports wheel: thick rim, three flat spokes (left, right, bottom) and a padded centre, on its column."""
    c = np.array([.265, .725, -.36])       # left-hand drive: +z is the car's right side
    n = np.array([np.cos(np.deg2rad(25)), -np.sin(np.deg2rad(25)), 0])      # column axis: forward and down into the dash
    e1 = np.array([0, 0, 1.])
    e2 = np.cross(n, e1)
    parts = []
    ring = np.array([c + .182 * (np.cos(a) * e1 + np.sin(a) * e2) for a in np.linspace(0, 2 * np.pi, 96, endpoint=False)])
    prof = [(.0165 * np.cos(a), .0135 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    parts.append(('wheel', 'cabin', sweep(np.concatenate([ring, ring[:1]]), prof, lambda i, p: n, caps=False)))

    def hub(x, y, z):
        d = np.stack([x - c[0], y - c[1], z - c[2]])
        along = np.tensordot(n, d, 1)
        a1 = np.tensordot(e1, d, 1)
        a2 = np.tensordot(e2, d, 1)
        # the padded centre: a rounded, slightly wider-than-tall cushion
        pad = (np.abs(a1 / .078) ** 3 + np.abs((a2 + .005) / .062) ** 3) ** (1 / 3) - 1
        pad = np.maximum(pad * .05, np.abs(along + .012) - .026)
        s = pad
        for a, w in ((0, .022), (np.pi, .022), (np.pi / 2, .03)):
            dv = np.cos(a) * e1 + np.sin(a) * e2
            t = np.clip(d[0] * dv[0] + d[1] * dv[1] + d[2] * dv[2], 0, .175)
            q = [d[k] - t * dv[k] for k in range(3)]                    # from the spoke's centre line
            al = q[0] * n[0] + q[1] * n[1] + q[2] * n[2]
            lat = np.sqrt(np.maximum(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 - al * al, 0))
            s = smin(s, np.maximum(lat - w, np.abs(al + .004) - .007), .012)
        col = capsule(x, y, z, c + .03 * n, c + .30 * n, .032)
        return smin(s, col, .015)
    from mesher import mesh_solid_sharp, autosmooth
    V, T = mesh_solid_sharp(hub, (c[0] - .23, c[1] - .23, c[2] - .22), (c[0] + .3, c[1] + .25, c[2] + .22), .003, tolerance=.0002)
    parts.append(('wheel', 'cabin_trim', autosmooth(Mesh(V, T), hub)))
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
