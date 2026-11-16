import copy

from shapely.geometry import box
from shapely.ops import unary_union

from pipeline.facade_signs import compile_reading, parts
from regen.specs import clean, to_profile
from regen.remap_readings import remap


PIECE = {'lengthM': 10., 'heightM': 8., 'views': [{'image': 'source-photo'}],
         'building': 'example', 'id': 'front', 'normal': [0, 1], 'frameU': [0, 10], 'planeV': 0.}


def sign():
    return {'x0': 3., 'x1': 7., 'z0': 4., 'z1': 5., 'mount': 'wall',
            'sourceImages': ['source-photo'], 'text': 'A & B 17', 'textLegible': True,
            'textHex': '#efe9db', 'board': {'family': 'metal-panel', 'hex': '#283438', 'thicknessM': .04}}


def test_literal_source_text_and_independent_board_survive_compiler():
    raw = {'signs': [sign()]}
    profile = to_profile(clean(raw, PIECE), PIECE)
    assert len(profile['signs']) == 1
    sg = profile['signs'][0]
    assert sg['text'] == 'A & B 17' and sg['textMaterial'] == '#efe9db'
    assert sg['boardMaterial'] != sg['textMaterial']
    assert sg['letterDepthM'] == .001
    assert sg['sourceIds'] == ['mapillary:source-photo']


def test_unread_text_does_not_create_fake_letters_but_keeps_observed_board():
    raw = sign(); raw['textLegible'] = False
    signs, gaps = compile_reading({'signs': [raw]}, PIECE, lambda m: m['hex'])
    assert len(signs) == 1 and 'text' not in signs[0]
    assert signs[0]['boardMaterial'] == '#283438' and gaps


def test_unrelated_source_and_unsupported_mount_are_unresolved():
    raw = sign(); raw['sourceImages'] = ['unrelated-photo']
    assert compile_reading({'signs': [raw]}, PIECE, lambda m: m['hex'])[0] == []
    raw = sign(); raw['mount'] = 'rooftop'
    assert compile_reading({'signs': [raw]}, PIECE, lambda m: m['hex'])[0] == []


def test_split_host_clips_original_letters_without_repeating_or_resizing():
    sg = compile_reading({'signs': [sign()]}, PIECE, lambda m: m['hex'])[0][0]
    contours = lambda *args: (unary_union([box(0, 0, 1, 2), box(2, 0, 4, 2)]), 4., 2.)
    whole = parts(sg, 10., 8., box(0, 0, 10, 8), contours)
    left = parts(sg, 10., 8., box(0, 0, 5, 8), contours)
    right = parts(sg, 10., 8., box(5, 0, 10, 8), contours)
    for idx in (0, 1):
        assert whole[idx][0].symmetric_difference(unary_union([left[idx][0], right[idx][0]])).area < 1e-9
        assert left[idx][0].intersection(right[idx][0]).area < 1e-9
    assert whole[1][1] > whole[0][1]  # lettering in front of the board


def test_sign_bounds_follow_camera_remap_without_scaling_physical_depth():
    class Map:
        L = 20.
        def __call__(self, x, z):
            import numpy as np
            return np.asarray(x)*2, np.asarray(z)+1
        def stretch(self, x, z): return 2., 1.
    raw = {'signs': [sign()]}
    mapped = remap(copy.deepcopy(raw), Map())['signs'][0]
    assert [mapped[k] for k in ('x0', 'x1', 'z0', 'z1')] == [6., 14., 5., 6.]
    assert mapped['board']['thicknessM'] == .04
    assert mapped['text'] == raw['signs'][0]['text']
