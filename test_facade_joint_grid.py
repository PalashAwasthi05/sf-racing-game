"""Independent geometry/photometry cases for street-photo module measurement."""
import numpy as np
import pytest
from general_world.facade_joint_grid import measure_joint_grid


def panels(*, pitch_x=43, pitch_y=58, sx=.08, sy=-.06, seed=7):
    y, x = np.indices((384, 512))
    # The fixtures use analytic distances, not the detector's fitting code.
    u = x - sx * y - 13.7
    v = y - sy * x - 22.3
    distance_x = np.abs(u - pitch_x * np.rint(u / pitch_x))
    distance_y = np.abs(v - pitch_y * np.rint(v / pitch_y))
    shade = 205. + 15 * x / 512 + 5 * y / 384
    shade -= 65 * np.exp(-distance_x**2 / 1.8) + 45 * np.exp(-distance_y**2 / 1.8)
    shade += np.random.default_rng(seed).normal(0, 2, shade.shape)
    mask = np.ones_like(x, dtype=bool)
    # Glass/foliage remove substantial evidence, but remain in the source.
    mask[100:190, 55:420] = False
    mask[275:335, 240:470] = False
    shade[~mask] = 45
    return np.repeat(np.clip(shade, 0, 255)[..., None], 3, axis=2), mask


@pytest.mark.parametrize('px,py,sx,sy', [(43, 58, .08, -.06), (31, 71, -.12, .10)])
def test_recovers_pitch_and_shear_despite_occlusion(px, py, sx, sy):
    image, mask = panels(pitch_x=px, pitch_y=py, sx=sx, sy=sy)
    result = measure_joint_grid(image, mask, min_pitch_px=18, max_pitch_px=100)
    for key, pitch, shear in [('vertical', px, sx), ('horizontal', py, sy)]:
        assert result[key]['accepted'], result[key]
        assert abs(result[key]['pitch_px'] - pitch) < 1.1
        assert abs(result[key]['shear'] - shear) < .015
    assert not result['world_registration_verified']
    assert not result['metric_scale_verified']


def test_rejects_flat_surface_and_windows_as_material_repeat():
    image = np.full((256, 384, 3), 210.)
    mask = np.ones((256, 384), bool)
    for x in range(20, 360, 48):
        image[50:200, x:x+25] = 20
        mask[50:200, x:x+25] = False
    result = measure_joint_grid(image, mask, max_pitch_px=100)
    assert not result['vertical']['accepted']
    assert not result['horizontal']['accepted']


def test_rejects_repeat_only_in_training_rows():
    image, _ = panels()
    mask = np.ones(image.shape[:2], bool)
    strips = np.minimum(7, np.arange(image.shape[0]) * 8 // image.shape[0])
    image[strips % 2 == 1] = 210
    result = measure_joint_grid(image, mask, min_pitch_px=18, max_pitch_px=100)
    assert not result['vertical']['accepted']


def test_does_not_invent_second_axis():
    y, x = np.indices((256, 384))
    d = np.abs(x - .08*y - 40*np.rint((x-.08*y)/40))
    gray = 210 - 70*np.exp(-d*d/2)
    result = measure_joint_grid(np.repeat(gray[...,None], 3, 2), np.ones_like(x, bool),
                                min_pitch_px=18, max_pitch_px=100)
    assert result['vertical']['accepted']
    assert not result['horizontal']['accepted']


def test_invalid_input_is_rejected():
    image, mask = panels()
    with pytest.raises(ValueError):
        measure_joint_grid(image, mask[:-1])
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        measure_joint_grid(image, mask)


def test_rejects_grain_without_repeated_joints():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(315)
    gray = 190 + gaussian_filter(rng.normal(0, 20, (256, 384)), 2)
    image = np.repeat(gray[..., None], 3, 2)
    result = measure_joint_grid(image, np.ones(gray.shape, bool), max_pitch_px=100)
    assert not result['vertical']['accepted']
    assert not result['horizontal']['accepted']


def test_narrow_material_band_can_supply_only_vertical_joints():
    y, x = np.indices((96, 640))
    q = (x + .03*y) / (1 + .14*y/96)
    d = np.abs(q - 49*np.rint(q/49))
    image = np.repeat((210 - 65*np.exp(-d*d/2))[..., None], 3, 2)
    result = measure_joint_grid(image, np.ones_like(x, bool),
                                min_pitch_px=25, max_pitch_px=100)
    assert result['vertical']['accepted'], result
    assert abs(result['vertical']['pitch_px'] - 49) < 1
    assert not result['horizontal']['accepted']
