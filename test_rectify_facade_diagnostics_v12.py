import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from rectify_facade_diagnostics_v12 import FrameConverter, _facing_cosines, _line_of_sight_clear


def test_line_of_sight_allows_owner_only_at_target_boundary():
    owner = Polygon([(8, -2), (12, -2), (12, 2), (8, 2)])
    assert _line_of_sight_clear(np.array([0.0, 0.0]), np.array([8.0, 0.0]), "owner", [("owner", owner)])


def test_line_of_sight_rejects_another_building_before_owner():
    owner = Polygon([(8, -2), (12, -2), (12, 2), (8, 2)])
    blocker = Polygon([(3, -2), (5, -2), (5, 2), (3, 2)])
    assert not _line_of_sight_clear(np.array([0.0, 0.0]), np.array([10.0, 0.0]), "owner", [("owner", owner), ("blocker", blocker)])


def test_facing_rejects_a_frontage_behind_the_camera():
    frame = {"a": [0, 0], "b": [10, 0], "normal": [0, 1], "baseZM": 0.0, "heightM": 8.0}
    midpoint, corners = _facing_cosines(frame, np.array([5.0, -10.0, 4.0]))
    assert midpoint < 0.5
    assert min(corners) < 0.25


def test_normalized_projected_to_cluster_enu_round_trip():
    normalized = {"coordinateFrame": {"projectedCrs": "EPSG:32631", "originProjectedM": [500000, 5000000], "verticalOriginM": 12}}
    source = {"camera": {"referenceLla": {"latitude": 45.15, "longitude": 3.0, "altitude": 20}}}
    converter = FrameConverter(normalized["coordinateFrame"], source["camera"]["referenceLla"])
    points = np.array([[0.0, 0.0, 0.0], [12.0, -7.0, 4.0], [-80.0, 42.0, 15.0]])
    np.testing.assert_allclose(converter.cluster_enu_to_world(converter.world_to_cluster_enu(points)), points, atol=1e-5)


def test_alignment_report_inverse_is_applied_before_geodetic_conversion():
    normalized = {"coordinateFrame": {"projectedCrs": "EPSG:32631", "originProjectedM": [500000, 5000000], "verticalOriginM": 12}}
    source = {"camera": {"referenceLla": {"latitude": 45.15, "longitude": 3.0, "altitude": 20}}}
    baseline = FrameConverter(normalized["coordinateFrame"], source["camera"]["referenceLla"])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([2.0, -3.0, 0.5])
    aligned = FrameConverter(normalized["coordinateFrame"], source["camera"]["referenceLla"], {"rotationMatrix": rotation.tolist(), "translationM": translation.tolist()})
    source_point = np.array([[-20.0, 15.0, 4.0]])
    world_point = source_point @ rotation.T + translation
    np.testing.assert_allclose(aligned.world_to_ecef(world_point), baseline.world_to_ecef(source_point), atol=1e-5)


def test_owner_mass_cannot_be_ignored_as_pose_uncertainty():
    owner = Polygon([(8, -2), (12, -2), (12, 2), (8, 2)])
    assert not _line_of_sight_clear(np.array([0., 0.]), np.array([10., 0.]), "owner", [("owner", owner)])


def test_independent_pairs_require_known_groups_and_baseline():
    from rectify_facade_diagnostics_v12 import _independent_cameras
    a = {"record": {"viewGroup": "a"}, "centerWorld": np.array([0., 0., 0.])}
    same = {"record": {"viewGroup": "a"}, "centerWorld": np.array([3., 0., 0.])}
    close = {"record": {"viewGroup": "b"}, "centerWorld": np.array([1., 0., 0.])}
    good = {"record": {"viewGroup": "b"}, "centerWorld": np.array([3., 0., 0.])}
    unknown = {"record": {}, "centerWorld": np.array([3., 0., 0.])}
    assert _independent_cameras(a, [same, close, good, unknown]) == [good]


def _geo_camera(source_id, center, *, cluster="cluster-a", image_hash=None):
    return {
        "sourceId": source_id,
        "centerWorld": np.asarray(center, dtype=float),
        "camera": {"clusterId": cluster},
        "record": {"imageSha256": image_hash or source_id + "-hash"},
    }


def test_geometric_pair_requires_shared_cluster_baseline_and_parallax():
    from rectify_facade_diagnostics_v12 import _geometric_pair_metrics
    first = _geo_camera("one", [-10.0, 0.0, 3.0])
    accepted = _geo_camera("two", [-10.0, 2.0, 3.0])
    assert _geometric_pair_metrics(first, accepted, np.array([0.0, 0.0, 3.0]))[0] >= 2.0
    assert _geometric_pair_metrics(first, accepted, np.array([0.0, 0.0, 3.0]))[1] >= 5.0


def test_geometric_pair_rejects_other_cluster_small_baseline_narrow_angle_and_duplicate():
    from rectify_facade_diagnostics_v12 import _geometric_pair_metrics
    first = _geo_camera("one", [-10.0, 0.0, 3.0])
    target = np.array([0.0, 0.0, 3.0])
    assert _geometric_pair_metrics(first, _geo_camera("other-cluster", [-10.0, 2.0, 3.0], cluster="cluster-b"), target) is None
    assert _geometric_pair_metrics(first, _geo_camera("small", [-9.0, 0.0, 3.0]), target) is None
    assert _geometric_pair_metrics(first, _geo_camera("narrow", [-10.0, 0.5, 3.0]), target) is None
    assert _geometric_pair_metrics(first, _geo_camera("duplicate", [-10.0, 2.0, 3.0], image_hash=first["record"]["imageSha256"]), target) is None
