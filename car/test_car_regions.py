"""Finish regions land where they are meant to (guards against inverted half-space tests)."""
import numpy as np
import roadster as R


def region(name):
    return [f for n, f in R.FINISH_REGIONS if n == name or getattr(f, '__name__', '') == name]


def inside(f, p):
    return f(np.array([p[0]]), np.array([p[1]]), np.array([p[2]]))[0] < 0


def test_band_only_at_the_tail():
    (band,) = region('_band')
    assert inside(band, (-2.05, .755, 0))
    assert not inside(band, (1.9, .755, 0)) and not inside(band, (0, .755, 0))


def test_lower_black_only_below_front_lip_diffuser_and_underside():
    (lower,) = region('_lower')
    assert inside(lower, (2.2, .18, 0)) and inside(lower, (-2.1, .3, .2)) and inside(lower, (0, .14, 0))
    assert not inside(lower, (0, .3, .88)) and not inside(lower, (-1.5, .3, .88))


def test_cabin_covers_the_cockpit_walls():
    (cabin,) = region('_cabin')
    assert inside(cabin, (0, .6, .725)) and not inside(cabin, (0, .6, .80))


def test_roof_cover_behind_the_seats_only():
    (cover,) = region('_cover')
    assert inside(cover, (-.8, .99, 0)) and inside(cover, (-.8, .99, .5))
    assert not inside(cover, (-.3, .99, 0)) and not inside(cover, (-1.3, .95, 0)) and not inside(cover, (-.8, .99, .85))


def test_gap_lines_are_signed_and_bounded():
    # a point on a gap line is at zero, points a centimetre either side have opposite signs, far away is outside
    x, y = R._DOOR[20]
    g, gate = R._door_line(np.array([x, x + .01, x - .01, 0.1]), np.array([y, y, y, .6]), np.array([.85, .85, .85, .85]))
    assert abs(g[0]) < 1e-6 and g[1] * g[2] < 0 and abs(g[3]) > .05 and (gate < 0).all()
    # the cut fields are continuous across the gate (no false zero crossing where the gate switches)
    near, band = R.gap_cuts(R._bonnet_line)
    xs = np.full(200, 1.6); zs = np.zeros(200); ys = np.linspace(.6, .9, 200)
    v = near(xs, ys, zs)
    assert np.abs(np.diff(v)).max() < .01
