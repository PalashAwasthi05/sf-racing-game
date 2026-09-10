import numpy as np
import pytest

from general_world.stereo_rays import closest_positive_rays, metric_plane_point


def test_known_off_plane_point_is_recovered():
    point = np.array([3.0, 2.0, 4.0])
    a, b = np.array([0.0, 0.0, 10.0]), np.array([10.0, 0.0, 10.0])
    result = closest_positive_rays(a, point - a, b, point - b)
    assert result["valid"]
    np.testing.assert_allclose(result["center"], point, atol=1e-8)
    assert result["rayResidualM"] < 1e-8


def test_parallel_rays_are_rejected():
    result = closest_positive_rays([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0])
    assert not result["valid"]
    assert result["reason"] == "parallel_or_low_parallax"


def test_behind_camera_intersection_is_rejected():
    result = closest_positive_rays([0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 1, 0])
    assert not result["valid"]
    assert result["reason"] == "negative_or_zero_depth"


def test_noisy_intersection_over_residual_bound_is_rejected():
    result = closest_positive_rays([0, 0, 10], [0, 0, -1], [10, 0, 10], [-10, 0.2, -1], max_residual_m=0.15)
    assert not result["valid"]
    assert result["reason"] == "ray_residual_too_large"


def test_nonorthogonal_metric_frame_is_rejected():
    record = {"metricFrame": {"origin": [0, 0, 0], "horizontal": [1, 0, 0], "vertical": [0.1, 0, 1]}, "widthM": 10, "diagnosticHeightM": 10}
    with pytest.raises(ValueError, match="orthonormal"):
        metric_plane_point(record, [10, 10], (100, 100))


def test_nonfinite_ray_bound_is_rejected():
    with pytest.raises(ValueError, match="ray bounds"):
        closest_positive_rays([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], max_residual_m=float("nan"))
