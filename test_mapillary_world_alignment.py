import numpy as np
from scipy.spatial.transform import Rotation

from mapillary_world_alignment import bounded_registration, enu_to_local_projected


def test_bounded_registration_recovers_small_rigid_displacement_with_outliers():
    rng = np.random.default_rng(4)
    source = rng.uniform([-25.0, -20.0, 0.0], [25.0, 20.0, 20.0], size=(240, 3))
    rotation = Rotation.from_euler("xyz", [2.0, -1.0, 3.0], degrees=True).as_matrix()
    target = source @ rotation.T + np.array([4.0, -3.0, 2.0])
    # These are provider outliers; the interleaved holdout still has a large
    # majority of valid points, so the report must remain conservative.
    source[-15:] += np.array([90.0, -70.0, 50.0])
    result = bounded_registration(source, target)
    assert result["accepted"]
    assert result["final"]["holdout"]["p75M"] < 1e-5
    assert result["transform"]["translationNormM"] < 30.0


def test_bounded_registration_rejects_wrong_overlap():
    rng = np.random.default_rng(8)
    source = rng.uniform([-20.0, -20.0, 0.0], [20.0, 20.0, 20.0], size=(100, 3)) + 100.0
    target = rng.uniform([-20.0, -20.0, 0.0], [20.0, 20.0, 20.0], size=(200, 3))
    result = bounded_registration(source, target)
    assert not result["accepted"]
    assert "overlap" in result["reason"]


def test_enu_conversion_applies_projected_origin_and_vertical_origin():
    frame = {"projectedCrs": "EPSG:32610", "originProjectedM": [553556.9197708304, 4181427.3639752637], "verticalOriginM": 4.23}
    point = enu_to_local_projected(np.array([[0.0, 0.0, 10.0]]), {"latitude": 37.7786, "longitude": -122.3918, "altitude": 0.0}, frame)
    assert np.isfinite(point).all()
    assert abs(point[0, 2] - 5.77) < 0.02
