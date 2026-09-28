"""v34 evidence rules: building occlusion masks, regular window layouts, and colours that agree under different light."""
import numpy as np
from shapely.geometry import box
import frontage_evidence_v34 as E
from pipeline.colour_agreement import same_material


def flat_surface(blocks, size=100., step=.5):
    """A surface grid over [-size, size]^2 with rectangular blocks (x0, y0, x1, y1, top)."""
    n = int(2 * size / step); top = np.full((n, n), np.nan, np.float32)
    xs = -size + (np.arange(n) + .5) * step; ys = size - (np.arange(n) + .5) * step
    for x0, y0, x1, y1, z in blocks:
        top[np.ix_((ys >= y0) & (ys <= y1), (xs >= x0) & (xs <= x1))] = z
    return top, (-size, size, step)


def test_a_building_in_front_hides_the_lower_wall_behind_it_and_not_the_top():
    wall = {'midpoint': [0., 0.], 'normal': [0., -1.], 'lengthM': 10., 'heightM': 20.}   # faces south, camera 30 m south
    surf = flat_surface([(-20, -12, 20, -8, 8.)])                                        # an 8 m block 8-12 m in front
    m = E.hidden_mask(wall, {'xy': [0., -30.]}, 0., 20., 10., surf)
    col = m[:, m.shape[1] // 2]; zs = 20. - (np.arange(len(col)) + .5) / E.PPM
    hidden_to = zs[col].max()
    assert 10. < hidden_to < 14.5 and not col[zs > 16].any()            # ray over an 8 m block from a 2.4 m eye: hidden up to ~12 m


def test_the_walls_own_building_does_not_hide_it():
    wall = {'midpoint': [0., 0.], 'normal': [0., -1.], 'lengthM': 10., 'heightM': 12.}
    surf = flat_surface([(-5, -3, 5, 10, 12.)])                         # own mass runs 3 m in front of the evidence plane
    own = box(-5, -3, 5, 10)
    assert E.hidden_mask(wall, {'xy': [0., -25.]}, 0., 12., 10., surf, own_poly=own).mean() == 0.
    assert E.hidden_mask(wall, {'xy': [0., -25.]}, 0., 12., 10., surf).mean() > .5    # without knowing it is its own


def test_a_post_does_not_hide_a_wall():
    wall = {'midpoint': [0., 0.], 'normal': [0., -1.], 'lengthM': 10., 'heightM': 12.}
    surf = flat_surface([(-.2, -10.2, .2, -9.8, 9.)])                  # a single 0.5 m cell 9 m tall
    assert E.hidden_mask(wall, {'xy': [0., -30.]}, 0., 12., 10., surf).mean() == 0.


def window(x0, x1, z0, z1, L, H):
    return {'rect': [x0 / L, z0 / H, x1 / L, z1 / H], 'style': {'kind': 'window'}, 'evidence': 'image-detected'}


def test_scattered_detections_become_storeys_and_bays():
    L, H = 30., 16.
    rng = np.random.default_rng(3); ops = []
    for zb in (5., 8.4, 11.8):                                           # three storeys, 3 m bays
        for k in range(9):
            cx = 1.8 + 3. * k + rng.normal(0, .25); w = 1.4 + rng.normal(0, .25)
            ops.append(window(cx - w / 2, cx + w / 2, zb + rng.normal(0, .08), zb + 1.8 + rng.normal(0, .08), L, H))
    ops.append(window(10., 10.25, 5., 12., L, H))                        # a pole read as a window
    p = {'frameU': [0., L], 'heightRefM': H, 'reading': {'groundStoreyM': 4.}, 'openings': ops}
    s = E.regular_layout(p)
    assert s['sliversDropped'] == 1 and s['storeys'] == 3 and abs(s['bayPitchM'] - 3.) < .1
    ws = [o for o in p['openings'] if o['style']['kind'] == 'window']
    widths = {round((o['rect'][2] - o['rect'][0]) * L, 2) for o in ws}; sills = {round(o['rect'][1] * H, 2) for o in ws}
    assert len(widths) == 1 and len(sills) == 3
    centres = sorted({round((o['rect'][0] + o['rect'][2]) / 2 * L, 1) for o in ws})
    assert np.allclose(np.diff(centres), 3., atol=.15)


def test_a_storey_without_a_pitch_keeps_its_positions():
    L, H = 20., 12.
    ops = [window(2., 3.2, 5., 7., L, H), window(11., 12.5, 5., 7.1, L, H)]
    p = {'frameU': [0., L], 'heightRefM': H, 'reading': {'groundStoreyM': 4.}, 'openings': ops}
    E.regular_layout(p)
    xs = sorted(round((o['rect'][0] + o['rect'][2]) / 2 * L, 2) for o in p['openings'])
    assert xs == [2.6, 11.75]


def test_sun_and_shade_of_one_wall_agree_and_different_materials_do_not():
    assert same_material([219, 151, 130], [133, 75, 55])                # brick in sun and in shade
    assert same_material([200, 200, 195], [110, 115, 125])              # grey render in sun and in shade
    assert not same_material([219, 151, 130], [119, 115, 109])          # brick against grey render
    assert not same_material([150, 70, 55], [200, 170, 130])            # brick against tan
    assert not same_material([90, 95, 60], [140, 70, 55])               # foliage against brick


def test_a_wall_higher_than_the_mass_behind_it_is_not_this_buildings_face():
    wall = {'midpoint': [0., 0.], 'normal': [0., -1.], 'lengthM': 10., 'heightM': 15.}
    low = flat_surface([(-5, 0, 5, 12, 9.5)])                           # only a 9.5 m roof behind a 15 m wall
    assert E.unbacked_mask(wall, 0., 15., 10., low).all()
    ok = flat_surface([(-5, 0, 5, 12, 14.)])                            # a roof a metre under the parapet
    assert not E.unbacked_mask(wall, 0., 15., 10., ok).any()


def test_a_set_back_face_of_the_same_building_is_kept_above_the_lower_mass():
    from shapely.geometry import box as bx
    wall = {'midpoint': [0., 0.], 'normal': [0., -1.], 'lengthM': 10., 'heightM': 15.}
    surf = flat_surface([(-5, 0, 5, 6, 9.5), (-5, 6, 5, 20, 15.)])     # a 9.5 m roof in front, the 15 m block set back
    sb = {}; m = E.unbacked_mask(wall, 0., 15., 10., surf, bx(-5, 0, 5, 20), sb)
    zs = 15. - (np.arange(m.shape[0]) + .5) / E.PPM
    assert sb['share'] > .9 and m[zs < 10.].all() and not m[zs > 11.].any()
    assert E.unbacked_mask(wall, 0., 15., 10., surf, None).all()        # without its own mass behind, nothing of it is there


def test_a_set_back_face_is_read_only_where_this_building_is_the_first_mass_on_the_ray():
    from shapely.geometry import box as bx
    wall = {'midpoint': [0., 0.], 'normal': [0., -1.], 'lengthM': 10., 'heightM': 15.}
    # low roof right behind the plane; own tall block 6-20 m behind on the left half; a neighbour 6-20 m behind on the right
    surf = flat_surface([(-5, 0, 5, 6, 9.5), (-5, 6, 0, 20, 15.), (0, 6, 5, 20, 16.)])
    own = bx(-5, 0, 5, 6).union(bx(-5, 6, 0, 20))
    sb = {}; E.unbacked_mask(wall, 0., 15., 10., surf, own, sb)
    other = E.setback_other(wall, {'xy': [0., -30.]}, 0., 15., 10., surf, own, sb['cols'])
    left, right = other[:, :40], other[:, 80:]
    assert not other[:, sb['cols']].all() and right[:, sb['cols'][80:]].all() if sb['cols'][80:].any() else True
    assert left.mean() < .5


def test_a_window_one_photograph_saw_and_two_clear_photographs_did_not_is_dropped():
    H = 10.; clear = np.zeros((int(H * E.PPM), int(20 * E.PPM)), bool); hidden = np.ones_like(clear)
    window, trunk = (2., 4., 3.4, 6., 'window'), (9., 1., 9.4, 7., 'window')
    others = [([(2.3, 4.1, 3.6, 6.1, 'window')], clear), ([(1.8, 3.9, 3.2, 5.9, 'window')], clear)]
    keep, dropped = E.agreed_openings([window, trunk], others, H)
    assert keep == [window] and dropped == 1
    keep, dropped = E.agreed_openings([window, trunk], [(o, hidden) for o, _ in others], H)   # the others did not see it: it stays
    assert dropped == 0
