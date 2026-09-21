import numpy as np
from shapely.geometry import box
from pyproj import Transformer

from infer_ground_paint_v21 import (
    _patch_pair,
    camera_to_ground_xy,
    extract_components,
    green_mask,
    Component,
)
from general_world.image_projection import angle_axis_matrix, camera_vectors_to_uv


def test_green_mask_accepts_muted_green_and_excludes_gray_and_lower_overlay():
    image = np.full((100, 100, 3), [132, 158, 114], dtype=np.uint8)
    image[:] = [145, 148, 146]
    image[52:90] = [132, 158, 114]
    mask = green_mask(image)
    assert not mask[:51].any()
    assert mask[51:68].any()
    assert not mask[69:].any()


def test_camera_rays_intersect_below_camera_plane_and_enforce_range():
    # The rotation maps a camera ray with positive camera-y to world down.
    rotation = [np.pi / 2, 0.0, 0.0]
    east, north, valid = camera_to_ground_xy(
        np.array([0.5, 0.5]), np.array([0.60, 0.54]), rotation, (10.0, 20.0)
    )
    assert valid[0]
    assert np.isfinite(east[0]) and np.isfinite(north[0])
    assert not valid[1] or np.hypot(east[1] - 10.0, north[1] - 20.0) >= 5.0


def test_camera_inverse_roundtrip_uses_nontrivial_angle_axis_rotation():
    rotation = [0.23, -0.31, 0.17]
    target_world = np.array([[3.0, 5.0, -2.5]], dtype=float)
    camera_vector = target_world @ angle_axis_matrix(rotation).T
    u, v = camera_vectors_to_uv(camera_vector)
    east, north, valid = camera_to_ground_xy(u, v, rotation, (10.0, 20.0))
    assert valid[0]
    np.testing.assert_allclose([east[0] - 10.0, north[0] - 20.0], [3.0, 5.0], atol=1e-6)


def test_pair_requires_separated_views_and_local_component_agreement():
    a = Component("a", 1, (0.0, 0.0), frozenset({(0, 0), (1, 0), (0, 1)}), box(0, 0, 1, 1))
    b = Component("b", 2, (3.0, 0.0), frozenset({(0, 0), (1, 0), (0, 1)}), box(0.4, 0.2, 1.4, 1.2))
    c = Component("c", 3, (0.1, 0.0), frozenset({(0, 0), (1, 0), (0, 1)}), box(4, 4, 5, 5))
    assert _patch_pair(a, b) is not None
    assert _patch_pair(a, c) is None


def test_components_are_clipped_and_require_connected_area():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[51:65, 45:65] = [100, 150, 90]
    meta = {
        "id": "pano",
        "captured_at": 1,
        "computed_geometry": {"coordinates": [0.0, 0.0]},
        "computed_rotation": [0.0, 0.0, 0.0],
    }
    # Empty geometry proves the hard street-space clip does not emit image-only paint.
    transformer = Transformer.from_crs(4326, 32610, always_xy=True)
    assert extract_components(image, meta, box(100, 100, 101, 101), 0.0, 0.0, transformer) == []
