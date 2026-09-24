"""Envelope frame binding: every box on a frontage stays aligned with that frontage.

Location-free: one synthetic footprint, one synthetic profile. The compile helpers read the frontage heading when
they are called, so a component that reused the heading's name for its own loop (the clock's tick angle) re-aimed
every box drawn after it on the same frontage. On a long frontage the raked deck treads then spun about their centres
and crossed the building, overhanging the street as a floating striped sheet.
"""
import math
import numpy as np
from shapely.geometry import box
from pipeline import envelopes

PROFILE = {'id': 'frame-binding-test', 'match': {'featureId': 'test'}, 'normal': [0, 1], 'replaceDepthM': 20, 'levels': {},
           'nodes': [{'id': 'clock-bay', 'span': [0, .1], 'z': [0, 14], 'material': 'brick', 'depthM': 0,
                      'components': [{'op': 'clock', 'at': .8, 'radiusM': 1.5}]},
                     {'id': 'long-wall', 'span': [.1, 1], 'z': [0, 10], 'material': 'brick', 'depthM': 0,
                      'components': [{'op': 'cornice', 'at': 1}]}],
           'assemblies': [{'op': 'bracedFrame', 'span': [.1, .9], 'z': [10, 20], 'depthM': -2, 'rearDepthM': 12, 'pitchM': 7},
                          {'op': 'lightBank', 'span': [0, 1], 'z': [20, 30], 'depthM': -5, 'centers': [.5], 'widthM': 10}]}
FOOT = box(0, -30, 120, 0)          # frontage along y=0, facing +y; the building lies at y<0


def compile_cubes(meshes=None):
    cubes = []
    ctx = {'put': lambda m, k, mat: meshes.append((m, mat)) if meshes is not None else None,
           'cube': lambda centre, extents, key, mat, angle: cubes.append((np.asarray(centre, float), extents, mat, angle)),
           'cylinder': lambda *a: None}
    c = {'id': 'test', 'g': FOOT}
    resolved = envelopes.prepare({}, c, 0, np.empty((0, 3)), [PROFILE])
    envelopes.compile(ctx, c, 0, resolved)
    return resolved, cubes


def test_boxes_after_a_clock_keep_the_frontage_heading():
    resolved, cubes = compile_cubes()
    u = resolved[0]['frame']['tangent']; heading = math.atan2(u[1], u[0])
    assert cubes, 'no boxes were drawn'
    wrong = [(mat, round(angle, 3)) for _, _, mat, angle in cubes if abs(math.remainder(angle - heading, math.pi)) > 1e-6]
    assert not wrong, (round(heading, 3), wrong[:5])


def test_deck_treads_stay_inside_the_building_behind_the_frontage():
    _, cubes = compile_cubes()
    treads = [(centre, extents, angle) for centre, extents, mat, angle in cubes if mat == 'concrete']
    assert treads, 'the raked deck drew nothing'
    for centre, (w, d, h), angle in treads:
        # the long side of a tread runs along the frontage, so its ends stay within the footprint's length
        ends = [centre[:2] + s * w / 2 * np.array([math.cos(angle), math.sin(angle)]) for s in (-1, 1)]
        for e in ends:
            assert -.5 <= e[0] <= 120.5 and -30 <= e[1] <= .5, (centre, w, angle, e)


def test_raked_deck_steps_down_away_from_the_frontage():
    # Seating faces the interior: the highest tread is the back row on the frontage frame, the lowest sits at the frame foot.
    _, cubes = compile_cubes()
    treads = sorted((centre[2], -centre[1]) for centre, (w, d, h), mat, angle in cubes if mat == 'concrete' and h < .5)
    heights = [z for z, _ in treads]; behind = [b for _, b in treads]
    assert abs(heights[0] - 10) < 1e-6 and abs(heights[-1] - 20) < 1e-6, (heights[0], heights[-1])
    assert all(b1 > b2 for b1, b2 in zip(behind, behind[1:])), behind           # higher rows are nearer the frontage
    assert abs(behind[-1] - 2) < 1e-6 and abs(behind[0] - 14) < 1e-6, (behind[0], behind[-1])


def test_light_masts_stand_on_the_deck():
    # The masts sit 5 m behind the frontage, 3 m into a 12 m deck falling 10 m from the frame top: the deck is at 17.5 m there.
    meshes = []
    compile_cubes(meshes)
    masts = [m for m, mat in meshes if mat == 'verdigris' and abs(m.bounds[0][1] + m.bounds[1][1] + 10) < .5
             and m.bounds[1][2] - m.bounds[0][2] > 9]
    assert len(masts) == 2, len(masts)
    for m in masts:
        assert m.bounds[0][2] < 17.6 and m.bounds[1][2] > 29.8, m.bounds
