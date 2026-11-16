import numpy as np
import pytest
from pathlib import Path

from pipeline import material_recipes
from pipeline.materials import Materials
from pipeline.surface_evidence import resolve_material
from regen.specs import material, to_profile
from general_world.texture_selection import select_texture


def test_legacy_material_key_is_unchanged_without_optional_observations():
    assert material({'family': 'brick', 'hex': '#a04532'}) == 'brick_modular#a04532'


def test_measured_brick_module_and_finish_make_a_stable_recipe_variant():
    key = material({'family': 'brick', 'hex': '#a04532', 'pattern': 'running_bond',
                    'finish': 'matte', 'moduleM': {'x': .24, 'z': .08}})
    assert key == material({'family': 'brick', 'hex': '#a04532', 'pattern': 'running_bond',
                            'finish': 'matte', 'moduleM': {'z': .08, 'x': .24}})
    assert '#a04532' in key
    recipe = material_recipes.observed_recipe(key)
    assert recipe['cell'] == [.24, .08]
    assert recipe['scale'] == pytest.approx([1.92, .64])
    assert recipe['observed_pattern'] == 'running_bond'


def test_material_bank_reconstructs_encoded_variant_after_name_roundtrip():
    key = material({'family': 'brick', 'hex': '#a04532', 'pattern': 'running_bond',
                    'finish': 'matte', 'moduleM': {'x': .24, 'z': .08}})
    mat, scale = Materials(Path(__file__).resolve().parent / 'material-assets', {}).get(key)
    assert mat.name == key
    assert scale == pytest.approx([1.92, .64])


def test_timber_course_axis_controls_cell_without_inventing_a_second_module():
    horizontal = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'horizontal_courses',
                           'finish': 'matte', 'moduleM': {'z': .2}})
    vertical = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'vertical_boards',
                         'finish': 'matte', 'moduleM': {'x': .15}})
    h = material_recipes.observed_recipe(horizontal)
    v = material_recipes.observed_recipe(vertical)
    assert h['cell'] == pytest.approx([2.4, .2])
    assert h['scale'] == pytest.approx([2.4, 1.6])
    assert v['cell'] == pytest.approx([.15, 1.2])
    assert v['scale'] == pytest.approx([3.6, 1.2])
    assert horizontal != vertical


def test_timber_shingle_patterns_roundtrip_and_use_both_measured_axes():
    straight = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'straight_shingles',
                         'finish': 'matte', 'moduleM': {'x': .2, 'z': .15}})
    scalloped = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'scalloped_shingles',
                          'finish': 'matte', 'moduleM': {'z': .15, 'x': .2}})
    for key, pattern in ((straight, 'straight_shingles'), (scalloped, 'scalloped_shingles')):
        decoded = material_recipes.parse_observed_key(key)
        recipe = material_recipes.observed_recipe(key)
        assert decoded['pattern'] == pattern
        assert decoded['moduleM'] == {'x': .2, 'z': .15}
        assert recipe['observed_pattern'] == pattern
        assert recipe['cell'] == pytest.approx([.2, .15])
        assert recipe['module_source'] == 'observed'
        assert recipe['scale'][0] / recipe['cell'][0] == pytest.approx(round(recipe['scale'][0] / recipe['cell'][0]))
        assert int(round(recipe['scale'][1] / recipe['cell'][1])) % 2 == 0


def test_shingles_allow_unmeasured_default_but_reject_partial_measured_module():
    key = material({'family': 'timber', 'hex': '#eeeeec', 'pattern': 'straight_shingles', 'finish': 'matte'})
    recipe = material_recipes.observed_recipe(key)
    assert recipe['observed_moduleM'] is None
    assert recipe['module_source'] == 'recipe_default_unmeasured'
    assert material_recipes.selector_target(key)['unit_size_m'] is None
    with pytest.raises(ValueError, match='both x and z'):
        material({'family': 'timber', 'hex': '#eeeeec', 'pattern': 'straight_shingles',
                  'moduleM': {'x': .2}})


def test_shingle_patterns_are_distinct_from_boards_and_tile_on_even_course_count():
    keys = [material({'family': 'timber', 'hex': '#bdbbb7', 'pattern': 'vertical_boards',
                      'finish': 'matte', 'moduleM': {'x': .2}})]
    keys.extend(material({'family': 'timber', 'hex': '#bdbbb7', 'pattern': pattern,
                          'finish': 'matte', 'moduleM': {'x': .2, 'z': .15}})
                for pattern in ('straight_shingles', 'scalloped_shingles'))
    mats = [material_recipes.create(key)[0] for key in keys]
    board_normal = np.asarray(mats[0].normalTexture)
    straight_normal = np.asarray(mats[1].normalTexture)
    scalloped_normal = np.asarray(mats[2].normalTexture)
    assert not np.array_equal(board_normal, straight_normal)
    assert not np.array_equal(straight_normal, scalloped_normal)
    for pattern, key in zip(('straight_shingles', 'scalloped_shingles'), keys[1:]):
        recipe = material_recipes.observed_recipe(key)
        target = material_recipes.selector_target(key)
        assert target['pattern'] == f'timber_{pattern}'
        assert target['pattern'] != 'timber_boards'
        assert select_texture(target, {'boards': {
            'substrate_family': 'wood', 'pattern': 'timber_boards', 'finish_class': 'matte',
            'strong_weathering': False, 'tile_size_m': recipe['scale'],
            'unit_size_m': [.2, .15], 'mean_rgb': [189, 187, 183], 'intrinsic_color': False,
        }})['action'] == 'request_new'


def test_shingle_recolor_preserves_height_normals_and_uses_paint_derived_seams():
    keys = [material({'family': 'timber', 'hex': color, 'pattern': 'scalloped_shingles',
                      'finish': 'matte', 'moduleM': {'x': .2, 'z': .15}})
            for color in ('#eeeeec', '#66635e')]
    light = material_recipes.observed_recipe(keys[0])
    dark = material_recipes.observed_recipe(keys[1])
    assert light['mortar'] == [round(c * .72) for c in [238, 238, 236]]
    assert light['mortar'][0] > 150
    assert dark['mortar'][0] < light['mortar'][0]
    light_mat = material_recipes.create(keys[0])[0]
    dark_mat = material_recipes.create(keys[1])[0]
    assert np.array_equal(np.asarray(light_mat.normalTexture), np.asarray(dark_mat.normalTexture))


def test_shingle_stagger_closes_over_even_number_of_courses():
    key = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'scalloped_shingles',
                    'finish': 'matte', 'moduleM': {'x': .2, 'z': .15}})
    recipe = material_recipes.observed_recipe(key)
    columns = int(round(recipe['scale'][0] / recipe['cell'][0]))
    rows = int(round(recipe['scale'][1] / recipe['cell'][1]))
    assert rows % 2 == 0
    assert (columns * recipe['cell'][0]) == pytest.approx(recipe['scale'][0])
    assert (rows * recipe['cell'][1]) == pytest.approx(recipe['scale'][1])


def test_adjacent_shingle_course_side_joints_shift_half_a_module():
    width, course, tile_w, tile_h = .2, .15, 2.4, 1.2
    n = 512
    x = np.tile(np.arange(n) * tile_w / n, (n, 1))
    z = np.tile((np.arange(n) * tile_h / n)[:, None], (1, n))
    _, _, side_joint, _ = material_recipes._shingle_surface(
        x, z, width=width, course=course, period_x=tile_w, period_z=tile_h,
        pattern='straight_shingles',
        joint=.004, relief=.0015)
    dx = tile_w / n
    for course_index, expected_offset in ((2, 0.), (3, width / 2)):
        row_index = int(round((course_index + .5) * course / (tile_h / n)))
        actual = np.flatnonzero(side_joint[row_index]) * dx
        expected = np.arange(expected_offset, tile_w + width, width)
        for edge in expected:
            distance = np.abs(((actual - edge + tile_w / 2) % tile_w) - tile_w / 2)
            assert distance.min() <= dx


def test_shingle_height_edge_curve_and_wrapped_normal_derivatives_are_periodic():
    width, course, tile_w, tile_h = .2, .15, 2.4, 1.2
    n = 512
    dx, dz = tile_w / n, tile_h / n
    x = np.tile(np.arange(n) * dx, (n, 1))
    z = np.tile((np.arange(n) * dz)[:, None], (1, n))
    height, seam, _, _ = material_recipes._shingle_surface(
        x, z, width=width, course=course, period_x=tile_w, period_z=tile_h,
        pattern='scalloped_shingles',
        joint=.004, relief=.0015)

    # The circular lower edge reaches its maximum depth at the shingle center,
    # returns to the course line at the ends, and stays narrow around the arc.
    half_width = width / 2
    depth = min(half_width, course * .9)
    radius = (half_width * half_width + depth * depth) / (2 * depth)
    for x_at in (width / 2, width / 4):
        offset = x_at - half_width
        edge_z = np.sqrt(radius * radius - offset * offset) - np.sqrt(radius * radius - half_width * half_width)
        ix = int(round(x_at / dx))
        iz = int(round(edge_z / dz))
        assert seam[iz, ix]
        assert not seam[min(n - 1, iz + int(round(.025 / dz))), ix]

    shifted, shifted_seam, _, _ = material_recipes._shingle_surface(
        x + tile_w, z + tile_h, width=width, course=course,
        period_x=tile_w, period_z=tile_h,
        pattern='scalloped_shingles', joint=.004, relief=.0015)
    assert np.allclose(height, shifted, atol=1e-12)
    assert np.array_equal(seam, shifted_seam)

    normal = material_recipes._periodic_normals(height, dx, dz)
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * dx)
    gz = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (2 * dz)
    expected = np.stack([-gx, -gz, np.ones_like(gx)], axis=2)
    expected /= np.linalg.norm(expected, axis=2, keepdims=True)
    assert np.allclose(normal[[0, -1], :, :], expected[[0, -1], :, :])
    assert np.allclose(normal[:, [0, -1], :], expected[:, [0, -1], :])


def test_pattern_only_course_direction_uses_recipe_default_without_claiming_measured_module():
    horizontal = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'horizontal_courses'})
    vertical = material({'family': 'timber', 'hex': '#8c7051', 'pattern': 'vertical_boards'})
    hdata = material_recipes.parse_observed_key(horizontal)
    vdata = material_recipes.parse_observed_key(vertical)
    h, v = material_recipes.observed_recipe(horizontal), material_recipes.observed_recipe(vertical)
    assert hdata['moduleM'] is None and vdata['moduleM'] is None
    assert h['observed_moduleM'] is None and v['observed_moduleM'] is None
    assert h['module_source'] == v['module_source'] == 'recipe_default_unmeasured'
    assert h['cell'] == pytest.approx([2.4, .15])
    assert v['cell'] == pytest.approx([.1, 1.2])
    assert h['observed_pattern'] == 'horizontal_courses'
    assert v['observed_pattern'] == 'vertical_boards'
    target = material_recipes.selector_target(material({'family': 'timber', 'hex': '#8c7051',
                                                        'pattern': 'horizontal_courses', 'finish': 'matte'}))
    assert target['unit_size_m'] is None
    assert target['module_source'] == 'recipe_default_unmeasured'


def test_pattern_only_brick_bond_uses_family_recipe_without_fabricated_module():
    key = material({'family': 'brick', 'hex': '#a04532', 'pattern': 'running_bond'})
    data = material_recipes.parse_observed_key(key)
    recipe = material_recipes.observed_recipe(key)
    assert data['moduleM'] is None
    assert recipe['observed_moduleM'] is None
    assert recipe['observed_pattern'] == 'running_bond'
    assert recipe['cell'] == pytest.approx([.225, .075])
    assert recipe['scale'] == pytest.approx([1.8, .6])
    assert recipe['module_source'] == 'recipe_default_unmeasured'


def test_recipe_finish_changes_roughness_but_color_alone_does_not_change_structure():
    matte = material({'family': 'metal-panel', 'hex': '#a0a0a0', 'pattern': 'panel_grid',
                      'finish': 'matte', 'moduleM': {'x': 1.2, 'z': .6}})
    glossy = material({'family': 'metal-panel', 'hex': '#a0a0a0', 'pattern': 'panel_grid',
                       'finish': 'glossy', 'moduleM': {'x': 1.2, 'z': .6}})
    recolored = material({'family': 'metal-panel', 'hex': '#445566', 'pattern': 'panel_grid',
                          'finish': 'matte', 'moduleM': {'x': 1.2, 'z': .6}})
    mr_matte = material_recipes.create(matte)[0].metallicRoughnessTexture
    mr_glossy = material_recipes.create(glossy)[0].metallicRoughnessTexture
    mr_recolored = material_recipes.create(recolored)[0].metallicRoughnessTexture
    assert np.asarray(mr_matte)[:, :, 1].mean() > np.asarray(mr_glossy)[:, :, 1].mean()
    assert np.array_equal(np.asarray(mr_matte), np.asarray(mr_recolored))


def test_profile_material_evidence_preserves_only_supplied_optional_fields():
    reading = {'use': 0, 'confidence': 'high', 'materials': [
        {'family': 'timber', 'hex': '#8c7051', 'x0': 0, 'x1': 4, 'z0': 0, 'z1': 6,
         'pattern': 'horizontal_courses', 'finish': 'matte', 'moduleM': {'z': .2}}
    ]}
    piece = {'lengthM': 4, 'heightM': 6, 'building': 'part-1', 'id': 'side-1',
             'views': [{'image': 'photo-1'}], 'normal': [1, 0], 'frameU': [0, 4], 'planeV': 0}
    profile = to_profile(reading, piece)
    assert profile['material'].startswith('timber_dark~obs1')
    evidence = profile['materialEvidence']
    assert evidence['pattern'] == 'horizontal_courses'
    assert evidence['finish'] == 'matte'
    assert evidence['moduleM'] == {'z': .2}


def test_texture_selector_consumes_encoded_module_and_observed_pigment():
    key = material({'family': 'brick', 'hex': '#914631', 'pattern': 'running_bond',
                    'finish': 'matte', 'moduleM': {'x': .24, 'z': .08}})
    target = material_recipes.selector_target(key)
    assert target['unit_size_m'] == [.24, .08]
    assert target['color_rgb'] == [145, 70, 49]
    asset = {'brick-match': {'substrate_family': 'brick', 'pattern': 'brick_running',
                             'finish_class': 'matte', 'strong_weathering': False,
                             'tile_size_m': [1.92, .64], 'unit_size_m': [.24, .08],
                             'mean_rgb': [145, 70, 49], 'intrinsic_color': False}}
    assert select_texture(target, asset)['action'] == 'reuse'
    assert select_texture({**target, 'color_rgb': [110, 65, 48]}, asset)['action'] == 'recolor'


def test_unmatched_typed_structure_requests_new_without_losing_recipe_evidence():
    key = material({'family': 'brick', 'hex': '#914631', 'pattern': 'stack_bond',
                    'finish': 'matte', 'moduleM': {'x': .24, 'z': .08}})
    target = material_recipes.selector_target(key)
    decision = select_texture(target, {})
    assert decision['action'] == 'request_new'
    assert decision['requirements']['pattern'] == 'stack_bond'
    mat, scale = material_recipes.create(key)
    assert mat.name == key
    assert scale == pytest.approx([1.92, .64])


def test_legacy_color_key_does_not_claim_unobserved_selector_structure():
    assert material_recipes.selector_target('brick_modular#914631') is None


def test_facade_material_transfer_preserves_typed_observed_pattern_and_module():
    profile = {'id': 'side-1', 'material': 'brick_modular#914631',
               'materialEvidence': {'family': 'brick', 'finish': 'matte',
                                    'colorRgb': [145, 70, 49], 'sourceIds': ['photo-1'],
                                    'pattern': 'running_bond', 'moduleM': {'x': .24, 'z': .08}}}
    key = resolve_material(profile)
    decoded = material_recipes.parse_observed_key(key)
    assert decoded['pattern'] == 'running_bond'
    assert decoded['moduleM'] == {'x': .24, 'z': .08}
    assert material_recipes.observed_recipe(key)['cell'] == [.24, .08]


@pytest.mark.parametrize('material_record', [
    {'family': 'brick', 'pattern': 'running_bond', 'moduleM': {'x': .24}},
    {'family': 'render', 'pattern': 'running_bond', 'moduleM': {'x': .24, 'z': .08}},
    {'family': 'timber', 'pattern': 'horizontal_courses', 'moduleM': {'x': .1}},
    {'family': 'brick', 'pattern': 'running_bond', 'moduleM': {'x': 0, 'z': .08}},
])
def test_invalid_or_unresolved_typed_structure_is_rejected(material_record):
    with pytest.raises(ValueError):
        material({**material_record, 'hex': '#a04532'})
