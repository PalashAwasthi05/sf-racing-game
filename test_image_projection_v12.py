import numpy as np
import pytest

from general_world.image_projection import (
    CameraTransform,
    Orientation,
    bilinear_sample_equirectangular,
    camera_vectors_to_uv,
    project_facade_crop,
)


def camera(*, registered=True, orientation=Orientation.EQUIRECTANGULAR):
    return CameraTransform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], orientation, registered)


def test_spherical_projection_known_directions_and_vectorization():
    u, v = camera_vectors_to_uv(
        np.array(
            [
                [0.0, 0.0, 1.0],  # centre
                [1.0, 0.0, 0.0],  # right quarter turn
                [0.0, 1.0, 0.0],  # north pole
                [0.0, -1.0, 0.0],  # south pole
                [0.0, 0.0, -1.0],  # seam endpoint
            ]
        )
    )
    np.testing.assert_allclose(u, [0.5, 0.75, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(v, [0.5, 0.5, 1.0, 0.0, 0.5])


def test_bilinear_sampling_wraps_the_horizontal_seam():
    image = np.arange(8, dtype=np.float64)[None, :, None]
    image = np.repeat(image, 3, axis=2)
    a, valid_a = bilinear_sample_equirectangular(image, np.array([0.0]), np.array([0.5]))
    b, valid_b = bilinear_sample_equirectangular(image, np.array([1.0]), np.array([0.5]))
    assert valid_a[0] and valid_b[0]
    np.testing.assert_allclose(a, b, atol=1e-12)
    # The seam blends the final and first pixels instead of clamping to one side.
    np.testing.assert_allclose(a[0], [3.5, 3.5, 3.5], atol=1e-12)


def test_invalid_uvs_are_holes_without_unsafe_indexing():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    result, valid = bilinear_sample_equirectangular(image, [np.nan, np.inf, 0.5], [0.5, 0.5, np.inf])
    assert result.shape == (3, 3)
    assert not valid.any()


def test_invalid_rotation_and_unsupported_orientation_fail():
    with pytest.raises(ValueError):
        CameraTransform([0.0, np.nan, 0.0], [0.0, 0.0, 0.0], 1, True)
    with pytest.raises(ValueError):
        CameraTransform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2, True)


def test_unregistered_pose_requires_explicit_diagnostic_mode():
    with pytest.raises(ValueError, match="registration"):
        project_facade_crop(
            np.zeros((8, 8, 3), dtype=np.uint8),
            camera(registered=False),
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
            1,
            1,
            4,
        )
    result = project_facade_crop(
        np.zeros((8, 8, 3), dtype=np.uint8),
        camera(registered=False),
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0],
        1,
        1,
        4,
        diagnostic=True,
    )
    assert result.diagnostic and not result.registered


def test_occlusion_provenance_is_required_for_admission_and_masks_known_blocks():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    geometry = dict(origin=[0, 0, -1], horizontal=[1, 0, 0], vertical=[0, 1, 0], width_m=1, height_m=1, pixels_per_m=4)
    unchecked = project_facade_crop(image, camera(registered=True), **geometry)
    assert unchecked.valid_mask.all()
    assert unchecked.occlusion_unchecked and not unchecked.admitted
    assert unchecked.provenance_flags["occlusionUnchecked"]

    blocked = np.zeros((8, 8), dtype=bool)
    blocked[:, :] = True
    checked = project_facade_crop(image, camera(registered=True), source_occlusion_mask=blocked, **geometry)
    assert checked.occlusion_checked and not checked.occlusion_unchecked
    assert checked.occlusion_mask.all()
    assert not checked.valid_mask.any()
    assert checked.admitted


def test_camera_behind_facade_is_not_valid():
    image = np.full((16, 16, 3), 80, dtype=np.uint8)
    # h x v = +z, while the camera is at z=0 and the facade is at z=+1.
    result = project_facade_crop(image, camera(registered=True), [0, 0, 1], [1, 0, 0], [0, 1, 0], 1, 1, 4)
    assert not result.front_facing_mask.any()
    assert not result.valid_mask.any()


def test_facade_crop_dimension_limits_are_enforced():
    with pytest.raises(ValueError, match="limits"):
        project_facade_crop(
            np.zeros((4, 4, 3), dtype=np.uint8),
            camera(registered=True),
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
            2,
            2,
            10,
            max_dimension=100,
            max_pixels=100,
        )


def test_facade_frame_rejects_skewed_basis():
    with pytest.raises(ValueError, match="orthogonal"):
        project_facade_crop(
            np.zeros((4, 4, 3), dtype=np.uint8),
            camera(registered=True),
            [0, 0, -1],
            [1, 0, 0],
            [1, 1, 0],
            1,
            1,
            2,
        )
