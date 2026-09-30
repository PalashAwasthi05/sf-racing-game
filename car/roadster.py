"""The player's roadster: a two-seat mid-engined sports car built from measured outlines.

The body is one smooth implicit solid. Its sides follow the measured plan outline with a tumblehome above and a tuck
below the widest line; its top is a surface across the car (bonnet between raised wings at the front, the belt at the
doors, the deck between the haunches at the rear); nose and tail are faces curved in plan; all of them meet in fillets.
Wheel wells, the cockpit, the side intakes and the front openings are cut out of it with rounded edges. Everything is
in metres in the car frame: +x forward from the midpoint between the axles, +y up from the ground, +z to the car's
right (the solid is symmetric, so fields take |z|).
"""
import json
import os
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, make_smoothing_spline
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(HERE, 'reference_outline.json')))

AXLE = 2.475 / 2
R_FRONT, R_REAR = .336, .347           # 235/35 R20 and 265/35 R20
W_FRONT, W_REAR = .235, .265
TRACK_FRONT, TRACK_REAR = 1.527, 1.535
DROP = .02                             # lowered chassis: the body sits 20 mm below the reference outline
NOSE, TAIL = 2.245, -2.146


# ---- field helpers ------------------------------------------------------------------------------------------------
def smin(a, b, k):
    """Cubic smooth minimum (C2): a fillet of about radius k where the two fields meet."""
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * h * k / 6


def smax(a, b, k):
    return -smin(-a, -b, k)


def pchip(pts):
    """A C2 curve through design points (named for the monotone C1 interpolant it replaced: reflections on a C1
    surface show a kink wherever the curvature jumps, so every design curve is a natural cubic spline)."""
    p = np.asarray(pts, float)
    return CubicSpline(p[:, 0], p[:, 1], bc_type='natural', extrapolate=True)


def smoothstep(t):
    """Quintic smoothstep: C2, so blends leave no curvature step for reflections to show."""
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)


def box2(u, v, hu, hv, r):
    """Signed distance to a rounded rectangle centred at the origin."""
    qu, qv = np.abs(u) - hu + r, np.abs(v) - hv + r
    return np.hypot(np.maximum(qu, 0), np.maximum(qv, 0)) + np.minimum(np.maximum(qu, qv), 0) - r


# ---- measured outlines ------------------------------------------------------------------------------------------------
_x = np.array(REF['x'])
_nose_ref = _x.max()
_xs = np.where(_x > 1.5, 1.5 + (_x - 1.5) * (NOSE - 1.5) / (_nose_ref - 1.5), _x)     # the longer front apron
_half = np.array(REF['half'])
_mirror = (_xs > .23) & (_xs < .47)                                                        # the mirror bump in plan
_half[_mirror] = np.interp(_xs[_mirror], _xs[~_mirror], _half[~_mirror])
_half_spline = make_smoothing_spline(_xs, _half, lam=2e-5)
_top = gaussian_filter1d(np.array(REF['top']), 1.0) - DROP


def plan_half(x):
    """Widest half-width of the body at station x. Toward the ends it stays at the wing width: there the rounded
    corners in plan belong to the nose and tail faces (their measured plan curves), so no two surfaces describe the same
    corner and the faces roll into the sides without a crease."""
    xc = np.clip(x, _xs[0], _xs[-1])
    w = _half_spline(xc)
    # ease into the constant wing width toward the ends (smoothly, so no kink shows in the side)
    w = w + np.maximum(.884 - w, 0) * smoothstep((x - 1.50) / .15)
    return w + np.maximum(.892 - w, 0) * smoothstep((-1.55 - x) / .15)


# setback of the nose and tail faces from their tips, by distance from the centre line, from the measured plan outline
def _plan_setback(z_pts, c_pts, z_end):
    """Setback c(|z|) of a nose or tail face from its tip, smooth (C2) and monotone: fitted as half-width against
    setback (well behaved where the corner turns into the side), inverted on a fine table, then continued steeply past
    the wing width so the face never reaches beyond the side."""
    c = np.asarray(c_pts, float)
    z = np.asarray(z_pts, float)
    # the square of the half-width is close to linear in the setback at the tip (a rounded nose), so fit that
    wts = np.ones_like(c)
    wts[0] = 1e4                                        # the tip stays at the tip
    wc = make_smoothing_spline(c, z * z, w=wts, lam=1e-5)
    cc = np.linspace(0, c[-1], 1200)
    ww = np.maximum.accumulate(np.maximum(wc(cc), 0))
    zz = np.sqrt(ww)
    keep = np.r_[True, np.diff(zz) > 1e-7]
    inv = CubicSpline(zz[keep], cc[keep], bc_type='natural')
    z_top = zz[keep][-1]

    def f(za):
        za = np.asarray(za, float)
        inside = inv(np.minimum(za, z_top))
        return np.where(za > z_top, c[-1] + 12 * (za - z_top) ** 2 + 2 * (za - z_top), inside)
    return f


_NOSE_PLAN = _plan_setback([0, .20, .353, .483, .54, .614, .688, .743, .783, .812, .832, .847, .859, .868, .875, .884],
                           [0, .010, .031, .076, .115, .167, .213, .258, .303, .349, .394, .44, .485, .53, .576, .70], .884)
_TAIL_PLAN = _plan_setback([0, .26, .40, .493, .593, .653, .71, .757, .791, .816, .837, .855, .87, .883, .892],
                           [0, .006, .022, .047, .092, .138, .183, .229, .274, .319, .365, .41, .456, .501, .546], .892)


def top_line(x):
    return np.interp(x, _xs, _top)


# ---- body sections ----------------------------------------------------------------------------------------------------
# Each station's section: below the widest line the side tucks under; above it a superelliptic shoulder (exponent n)
# runs from the widest point up and in to the crest over the wing or haunch (crest height and lateral position); inboard
# of the crest the top falls to the centre (bonnet between the wings, deck between the haunches): flat over the given
# share of the crest's lateral position, then rising in an S to the crest.
#            x     widest-y  crest  crest-z   n    centre  flat   tuck
_SEC = np.array([
    [-2.22,  .45,  .745,  .52,  2.0,  .730,  .50,  .40],
    [-2.10,  .47,  .800,  .62,  2.3,  .800,  .50,  .30],
    [-2.00,  .49,  .830,  .67,  2.5,  .800,  .50,  .25],
    [-1.85,  .52,  .854,  .70,  2.6,  .806,  .50,  .22],
    [-1.70,  .55,  .889,  .71,  2.6,  .836,  .50,  .22],
    [-1.55,  .58,  .926,  .715, 2.55, .870,  .50,  .25],
    [-1.40,  .60,  .953,  .715, 2.5,  .904,  .50,  .28],
    [-1.25,  .61,  .973,  .71,  2.4,  .940,  .50,  .30],
    [-1.10,  .61,  .965,  .70,  2.35, .975,  .50,  .30],
    [-1.00,  .60,  .940,  .70,  2.4,  .990,  .45,  .30],
    [-.850,  .58,  .905,  .72,  2.6, 1.005,  .40,  .35],
    [-.700,  .56,  .890,  .74,  2.8, 1.010,  .35,  .40],
    [-.550,  .54,  .887,  .755, 3.0, 1.000,  .30,  .40],
    [-.400,  .53,  .884,  .76,  3.0,  .950,  .30,  .40],
    [.000,   .52,  .876,  .76,  3.0,  .880,  .30,  .40],
    [.400,   .52,  .873,  .75,  3.0,  .850,  .30,  .40],
    [.600,   .52,  .876,  .73,  2.8,  .835,  .35,  .40],
    [.800,   .52,  .878,  .71,  2.6,  .825,  .40,  .35],
    [.950,   .52,  .874,  .69,  2.5,  .808,  .45,  .30],
    [1.24,   .52,  .861,  .68,  2.5,  .784,  .50,  .30],
    [1.50,   .51,  .822,  .665, 2.45, .742,  .50,  .30],
    [1.80,   .48,  .722,  .64,  2.35, .658,  .52,  .35],
    [2.00,   .44,  .646,  .60,  2.2,  .612,  .55,  .40],
    [2.12,   .40,  .590,  .52,  2.0,  .578,  .55,  .40],
    [2.26,   .34,  .470,  .40,  2.0,  .470,  .50,  .40],
])
_sec = [CubicSpline(_SEC[:, 0], _SEC[:, k], bc_type='natural', extrapolate=True) for k in range(1, 8)]

# the underside is not seen: a monotone interpolant (no overshoot across the long flat floor)
_bottom = PchipInterpolator(*np.array([[-2.2, .27], [-2.05, .24], [-1.9, .205], [-1.6, .175], [-1.45, .135], [1.5, .135], [1.8, .142], [2.05, .152], [2.26, .17]]).T, extrapolate=True)

# Nose and tail: forward (rearward) extent at the centre by height, set back in plan.
_nose = pchip([[.10, 2.15], [.14, 2.19], [.18, 2.232], [.25, 2.245], [.35, 2.240], [.45, 2.22], [.52, 2.19], [.57, 2.15], [.62, 2.10], [.70, 2.0], [.80, 1.85]])
# (the spoiler lip over the band is its own blade, see parts.spoiler: a sub-grid bump here meshes as a torn edge)
_tail = pchip([[.14, -1.99], [.17, -2.01], [.21, -2.06], [.25, -2.10], [.29, -2.128], [.33, -2.143], [.38, -2.147], [.45, -2.144], [.52, -2.132], [.60, -2.117], [.68, -2.103], [.73, -2.088], [.78, -2.078], [.81, -2.062], [.835, -2.035], [.86, -1.995], [.92, -1.90]])



def spoiler_lip(y, z):
    """The retracted spoiler's trailing edge: it overhangs the band between the tail lamps by about 13 mm, with a
    short flat underside and a longer ramp up into the deck."""
    under = smoothstep((y - .774) / .008)
    over = smoothstep((.818 - y) / .026)
    return .013 * under * over * smoothstep((.50 - np.abs(z)) / .05)


ARCH_FRONT = (AXLE, .311, .389)
ARCH_REAR = (-AXLE, .321, .404)


def deck_dome(x, z):
    """The rear lid's raised centre: a panel about 20 mm proud between two creases that close in slightly toward the lip."""
    half = .37 - .05 * smoothstep((-1.15 - x) / .8)
    across = smoothstep((half - z) / .085)
    along = smoothstep((x + 2.02) / .12) * smoothstep((-1.07 - x) / .06)
    return .019 * across * along


def _section_fields(x, y, z):
    """Raw (unnormalised) section fields at points: shoulder corner, inner top, lower side."""
    hw, pk, zp, n, cz, t0, tuck = (f(x) for f in _sec)
    W = plan_half(x)
    t = np.minimum(z / zp, 1)
    rise = smoothstep((t - t0) / (1 - t0))
    ht = cz - .012 * t * t + (pk - (cz - .012)) * rise + deck_dome(x, z)
    # outboard of the crest the top continues level: a plane that rose there would pass just inside the shoulder and
    # its short distance would take over the field's depth (fillets and the side dent then cut grooves along the crest)
    ht = np.where(z > zp, pk, ht)
    top = y - ht
    a = np.maximum(W - zp, .02)
    # inboard of the crest the shoulder must not cap the top below the deck (behind the seats the deck stands higher
    # than the haunches): there its height is the inner top's
    b = np.maximum(np.where(z < zp, np.maximum(ht, pk), pk) - hw, .02)
    u = np.maximum(z - zp, 0) / a
    v = np.maximum(y - hw, 0) / b
    corner = (u ** n + v ** n) ** (1 / n) - 1
    # above the widest line the side continues as a vertical wall outside the shoulder (a plane leaning out there lay
    # just under the shoulder and cut a groove along the widest line wherever a fillet or the side dent met it)
    ws = W - tuck * np.maximum(hw - y, 0) ** 2
    side = z - ws
    return corner, top, side, np.maximum(a, b)


def _norm(fn, x, y, z, e=1e-3, floor=None):
    v = fn(x, y, z)
    gx = (fn(x + e, y, z) - fn(x - e, y, z)) / (2 * e)
    gy = (fn(x, y + e, z) - fn(x, y - e, z)) / (2 * e)
    gz = (fn(x, y, z + e) - fn(x, y, np.maximum(z - e, 0)) ) / (e + np.minimum(z, e))
    g = np.sqrt(gx * gx + gy * gy + gz * gz)
    if floor is not None:
        g = np.maximum(g, floor)
    return v / np.maximum(g, 1e-6)


def _section_normalised(x, y, z, e=1e-3):
    """Corner, top and side fields divided by their gradient lengths, from one set of seven section evaluations."""
    f0 = _section_fields(x, y, z)
    d = [(_section_fields(x + e, y, z), _section_fields(x - e, y, z), 2 * e),
         (_section_fields(x, y + e, z), _section_fields(x, y - e, z), 2 * e),
         (_section_fields(x, y, z + e), _section_fields(x, y, np.maximum(z - e, 0)), e + np.minimum(z, e))]
    out = []
    for k, floor in ((0, 2.0), (1, None), (2, None)):
        g = np.sqrt(sum(((p[k] - m[k]) / h) ** 2 for p, m, h in d))
        if floor is not None:
            g = np.maximum(g, floor)
        out.append(f0[k] / np.maximum(g, 1e-6))
    return out


def shell(x, y, z):
    z = np.abs(z)
    corner, top, side = _section_normalised(x, y, z)
    bottom = _bottom(x) - y
    front = _norm(lambda a, b, c: a - (_nose(b) - _NOSE_PLAN(np.abs(c))), x, y, z)
    rear = _norm(lambda a, b, c: (_tail(b) + _TAIL_PLAN(np.abs(c)) - spoiler_lip(b, c)) - a, x, y, z)
    # the shoulder, the inner top and the lower side meet tangentially by construction: a plain maximum keeps them
    # smooth (a smooth maximum of two coincident surfaces would cut a groove along the seam); real edges get fillets
    # the nose and tail faces are filleted only into the top (the bonnet's leading edge, the lip); against the sides a
    # fillet would cut a groove wherever a face runs close alongside the side surface
    s = np.maximum(np.maximum(corner, side), smax(smax(top, front, .035), rear, .012))
    s = np.maximum(s, np.maximum(front, rear))
    s = smax(s, bottom, .03)
    return s


def wheel_well(x, y, z, arch, z_in=.60):
    xa, ya, r = arch
    return np.maximum(np.hypot(x - xa, y - ya) - r, z_in - np.abs(z))


def cockpit(x, y, z):
    z = np.abs(z)
    # opening in plan from the bulkhead behind the seats to the dash, inside the door tops
    xc, hx = (.55 + -.50) / 2, (.55 + .50) / 2
    d = box2(x - xc, z, hx, .72, .14)
    return np.maximum(d, .30 - y)


# ---- features cut into the shell ------------------------------------------------------------------------------------
from shapes import Outline, catmull   # noqa: E402

# Side intake behind the door, in side view (x, y): a slot leaning back at about 48 degrees.
SIDE_INTAKE = Outline(catmull([(-.368, .318), (-.392, .298), (-.44, .305), (-.515, .345), (-.60, .42), (-.668, .50), (-.708, .575),
                               (-.716, .628), (-.698, .657), (-.655, .664), (-.595, .662), (-.556, .648), (-.528, .608), (-.49, .532),
                               (-.445, .445), (-.405, .375)], 8), (-.8, .2), (-.3, .75))


def side_scoop(x, y, z):
    """The door's rear lower quarter falls in toward the intake: a smooth dent in the side, deepest at the intake."""
    # along and across the intake axis
    ax, ay = -.74, .67                       # unit-ish direction of the slot axis (back and up)
    n = np.hypot(ax, ay)
    ax, ay = ax / n, ay / n
    dx, dy = x - (-.36), y - .40
    along = dx * ax + dy * ay
    across = -dx * ay + dy * ax
    g = np.exp(-(along / .20) ** 2 - (across / .13) ** 2)
    ramp = smoothstep((x - .05) / (-.42 - .05))   # grows from the middle of the door to the intake
    return .030 * g * ramp


def side_intake(x, y, z):
    """The intake duct: open inside the outline and running inboard toward the engine behind the cockpit; ahead of
    the cockpit's rear wall it stays shallow so it never opens into the cabin."""
    inner = .66 + .10 * smoothstep((x + .52) / .06)
    return np.maximum(SIDE_INTAKE(x, y), inner - np.abs(z))


# Front openings in front view (z, y): the two corner intakes under the lamps and the centre intake below the plate
# position; they are cut back along -x from the bumper face.
FRONT_SIDE_INTAKE = Outline(catmull([(.47, .375), (.62, .378), (.78, .372), (.835, .352), (.84, .29), (.82, .235), (.76, .222),
                                     (.60, .224), (.545, .232), (.505, .30)], 8), (.3, .1), (1., .5))
FRONT_CENTRE_INTAKE = Outline(catmull([(0, .318), (.24, .316), (.37, .31), (.415, .285), (.39, .215), (.30, .196), (0, .192),
                                       (-.30, .196), (-.39, .215), (-.415, .285), (-.37, .31), (-.24, .316)], 8), (-.6, .1), (.6, .45))


def front_intakes(x, y, z):
    za = np.abs(z)
    sidei = np.maximum(FRONT_SIDE_INTAKE(za, y), 1.86 - x)
    centre = np.maximum(FRONT_CENTRE_INTAKE(z, y), 2.02 - x)
    return np.minimum(sidei, centre)


# Headlamps: the lens takes the top of each wing's front. It is the region above the lamp's lower edge in side view and
# inside its outline in front view.
LAMP_SIDE = Outline(np.array([(1.44, .95), (1.515, .80), (1.545, .762), (1.61, .71), (1.71, .648), (1.81, .598), (1.90, .560),
                              (1.98, .540), (2.10, .525), (2.25, .515), (2.25, .95)]), (1.3, .4), (2.35, 1.0))
LAMP_FRONT = Outline(catmull([(.60, .765), (.71, .758), (.84, .745), (.96, .74), (1.0, .66), (.96, .56), (.80, .556), (.70, .552),
                              (.58, .545), (.515, .548), (.508, .58), (.54, .67)], 8), (.4, .45), (1.1, .85))


def lamp_outline(x, y, z):
    return np.maximum(LAMP_SIDE(x, y), LAMP_FRONT(np.abs(z), y))


# Tail lamps: on the rear corners, from the band out round the corner along the haunch.
# the lamps' top edge stays under the spoiler lip, so the lip keeps its thickness
TAIL_SIDE = Outline(catmull([(-1.49, .764), (-1.60, .787), (-1.75, .805), (-1.90, .812), (-2.05, .806), (-2.20, .80), (-2.20, .68),
                             (-2.05, .688), (-1.90, .695), (-1.75, .72), (-1.60, .745)], 6), (-2.3, .6), (-1.4, .9))
TAIL_REAR = Outline(np.array([(.49, .776), (.60, .792), (.75, .806), (.95, .815), (.95, .685), (.75, .69), (.60, .705), (.49, .731)]), (.3, .6), (1.05, .9))


def tail_outline(x, y, z):
    return np.maximum(TAIL_SIDE(x, y), TAIL_REAR(np.abs(z), y))


def body_base(x, y, z):
    s = shell(x, y, z)
    za = np.abs(z)
    s = s + side_scoop(x, y, z) * smoothstep((za - .62) / .16)
    s = smax(s, -wheel_well(x, y, z, ARCH_FRONT), .012)
    s = smax(s, -wheel_well(x, y, z, ARCH_REAR), .012)
    s = smax(s, -cockpit(x, y, z), .03)
    return s


# Rear: the band between the tail lamps sits under the spoiler lip; the diffuser is recessed into the bumper's lower
# edge with an opening for each tailpipe; the lid has an air grille at the front of its raised centre.
DIFFUSER = Outline(catmull([(0, .338), (.30, .336), (.46, .33), (.60, .31), (.665, .27), (.67, .12), (-.67, .12), (-.665, .27),
                            (-.60, .31), (-.46, .33), (-.30, .336)], 8), (-.8, 0.), (.8, .45))
TIP_Z, TIP_Y, TIP_R = .30, .232, .058


def rear_recesses(x, y, z, s):
    """Band, diffuser and tailpipe openings. Recess floors follow the skin s at a fixed depth, so no floor plane grazes
    the curved bumper (a grazing plane leaves a dotted trail of dents)."""
    za = np.abs(z)
    band = np.maximum(np.maximum(np.abs(y - .7545) - .0262, za - .495), np.maximum(-(s + .014), x + 1.97))
    diffuser = np.maximum(DIFFUSER(z, y), np.maximum(-(s + .028), x + 1.90))
    tips = np.maximum(np.hypot(za - TIP_Z, y - TIP_Y) - TIP_R, x + 1.90)
    return np.minimum(np.minimum(band, diffuser), tips)


def deck_grille(x, y, z):
    return np.maximum(box2(x + 1.125, z, .045, .25, .02), .95 - y)


# Front: the thin lamp strips above the corner intakes.
FRONT_STRIP = Outline(catmull([(.50, .437), (.66, .441), (.79, .436), (.805, .424), (.79, .414), (.66, .413), (.51, .418), (.495, .428)], 6), (.4, .35), (.9, .5))


def front_strips(x, y, z, s):
    return np.maximum(np.maximum(FRONT_STRIP(np.abs(z), y), 1.90 - x), -(s + .010))


def skin(x, y, z):
    """The painted skin with its openings, before the lamp housings are sunk (the lenses lie on this surface)."""
    s = body_base(x, y, z)
    s = smax(s, -side_intake(x, y, z), .012)
    s = smax(s, -front_intakes(x, y, z), .01)
    s = smax(s, -rear_recesses(x, y, z, s), .006)
    s = smax(s, -deck_grille(x, y, z), .006)
    return smax(s, -front_strips(x, y, z, s), .003)


def body(x, y, z):
    s = skin(x, y, z)
    # lamp housings: the lens area is sunk 60 mm (the lens itself is a separate surface on the original skin)
    lamp = np.maximum(lamp_outline(x, y, z), -(s + .045))
    s = smax(s, -lamp, .008)
    tail = np.maximum(tail_outline(x, y, z), -(s + .025))
    return smax(s, -tail, .008)


# ---- finishes on the body: regions (negative inside) cut out of the painted mesh, in order ---------------------------
def _wells(x, y, z):
    za = np.abs(z)
    d = np.minimum(np.hypot(x - ARCH_FRONT[0], y - ARCH_FRONT[1]) - ARCH_FRONT[2] - .006,
                   np.hypot(x - ARCH_REAR[0], y - ARCH_REAR[1]) - ARCH_REAR[2] - .006)
    return np.maximum(d, .5 - za)


def _openings(x, y, z):
    """Inner walls of the intakes: inside an opening outline and more than 8 mm under the surrounding skin."""
    za = np.abs(z)
    side = np.maximum(SIDE_INTAKE(x, y) - .003, .64 - za)
    fs = np.maximum(FRONT_SIDE_INTAKE(za, y) - .003, 1.84 - x)
    fc = np.maximum(FRONT_CENTRE_INTAKE(z, y) - .003, 2.0 - x)
    return np.minimum(side, np.minimum(fs, fc))


def _cabin(x, y, z):
    za = np.abs(z)
    xc, hx = (.55 + -.50) / 2, (.55 + .50) / 2
    return np.maximum(box2(x - xc, za, hx + .01, .735, .15), y - .845)


def _lower(x, y, z):
    """Underside, front splitter and rear diffuser."""
    under = y - .15
    lip = np.maximum(y - .205, 1.85 - x)
    diff = np.maximum(DIFFUSER(z, y) - .004, x + 1.93)
    return np.minimum(under, np.minimum(lip, diff))


def _band(x, y, z):
    return np.maximum(np.maximum(np.abs(y - .7545) - .0265, np.abs(z) - .49), x + 1.97)


def _cover(x, y, z):
    """The folded roof's cover behind the seats: black, from the bulkhead back to a rounded rear edge."""
    za = np.abs(z)
    rear_edge = -1.05 + .20 * (za / .72) ** 2.2
    return np.maximum(np.maximum(x + .49, rear_edge - x), np.maximum(za - .74, .93 - y))


def _reflectors(x, y, z):
    return np.maximum(np.maximum(np.abs(y - .413) - .0165, .60 - np.abs(z)), x + 1.72)


def _intake_frame(x, y, z):
    """A dark satin surround about 15 mm wide round the side intake."""
    return np.maximum(SIDE_INTAKE(x, y) - .016, .70 - np.abs(z))


def _strip(x, y, z):
    return np.maximum(FRONT_STRIP(np.abs(z), y) - .002, 1.90 - x)


def _grille(x, y, z):
    return np.maximum(box2(x + 1.125, np.abs(z), .047, .252, .02), .9 - y)


# Panel gaps: 4 mm dark lines where the doors, bonnet and rear lid meet the body, and the fuel flap. Each is a band round
# a centre line; GAP_LINES lists the centre-line distance fields so the mesh can be refined along them before cutting.

GAP = .0021
_DOOR = catmull([(.618, .872), (.628, .70), (.636, .50), (.640, .36), (.622, .292), (.57, .268), (.30, .262), (-.05, .266), (-.18, .282),
                 (-.28, .318), (-.345, .37), (-.395, .43), (-.44, .505), (-.475, .585), (-.495, .665), (-.500, .745), (-.492, .815), (-.475, .88)],
                6, closed=False)
_BONNET = catmull([(.865, .598), (1.00, .604), (1.20, .594), (1.40, .574), (1.55, .545), (1.70, .505), (1.85, .478), (1.98, .438), (2.065, .36),
                   (2.11, .20), (2.125, 0), (2.11, -.20), (2.065, -.36), (1.98, -.438), (1.85, -.478), (1.70, -.505), (1.55, -.545),
                   (1.40, -.574), (1.20, -.594), (1.00, -.604), (.865, -.598)], 6, closed=False)
_RLID = catmull([(-1.97, .505), (-1.80, .522), (-1.55, .536), (-1.30, .545), (-1.13, .545), (-1.09, .45), (-1.08, 0), (-1.09, -.45), (-1.13, -.545),
                 (-1.30, -.545), (-1.55, -.536), (-1.80, -.522), (-1.97, -.505)], 6, closed=False)
FLAP = (1.02, .64, .062)          # fuel filler flap on the right front wing: plan centre and radius


# Each line is drawn only on the skin it belongs to: a gate field (negative where the line applies) keeps its extrusion
# off hidden surfaces below (wheel-well walls, lamp housings, the cockpit's door panels). The gap is cut exactly between
# signed distances -GAP and +GAP from the centre line, each cut field taken as max(edge field, gate) so every field stays
# continuous (a gate that switched a field on and off would itself read as an edge and cut a false line).
from shapes import polyline_signed   # noqa: E402


def _door_line(x, y, z):
    return polyline_signed(_DOOR, x, y), .77 - np.abs(z)


def _bonnet_line(x, y, z):
    floor = .74 - .17 * smoothstep((x - 1.55) / .45)
    return polyline_signed(_BONNET, x, z), floor - y


def _rlid_line(x, y, z):
    return polyline_signed(_RLID, x, z), .83 - y


def _flap_line(x, y, z):
    return np.hypot(x - FLAP[0], z - FLAP[1]) - FLAP[2], np.maximum(-z, .7 - y)


GAP_LINES = [_door_line, _bonnet_line, _rlid_line, _flap_line]


def gap_centre(line):
    """Unsigned distance to the line where it applies (for refining the mesh along it)."""
    def f(x, y, z):
        g, gate = line(x, y, z)
        return np.where(gate < 0, np.abs(g), 1.0)
    return f


def gap_cuts(line):
    """The two cut fields of a gap: inside the first means g < +GAP, inside the second g > -GAP (both gated)."""
    def near(x, y, z):
        g, gate = line(x, y, z)
        return np.maximum(g - GAP, gate)

    def band(x, y, z):
        g, gate = line(x, y, z)
        return np.maximum(-g - GAP, gate)
    return near, band


FINISH_REGIONS = [('lamp_black', lambda x, y, z: lamp_outline(x, y, z) - .003),
                  ('lamp_black', lambda x, y, z: tail_outline(x, y, z) - .003),
                  ('black_matte', _openings), ('black_matte', _wells), ('cabin', _cabin),
                  ('black_satin', _lower), ('black_gloss', _band), ('black_satin', _cover),
                  ('reflector', _reflectors), ('black_satin', _intake_frame), ('drl', _strip), ('black_matte', _grille)]


BOUNDS = ((-2.2, .06, -.95), (2.3, 1.12, .95))
