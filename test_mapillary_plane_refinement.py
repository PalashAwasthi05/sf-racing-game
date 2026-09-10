import numpy as np

from mapillary_plane_refinement import _plane_fit


def _three_planes(seed=2):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 12.0, size=(500, 2))
    xy = np.column_stack((np.zeros(len(a)), a))
    yz = np.column_stack((a[:, :1], np.zeros(len(a)), a[:, 1:]))
    xz = np.column_stack((a, np.zeros(len(a))))
    return np.vstack((xy, yz, xz))


def test_plane_refinement_uses_diverse_normals_and_improves_holdout_plane_residual():
    target = _three_planes()
    source = target - np.array([0.35, -0.25, 0.20])
    result = _plane_fit(source, target, np.eye(3), np.zeros(3))
    assert result["status"] == "candidate-evidence"
    assert result["normalDiversity"]["axisGroupsPresent"] >= 2
    assert result["holdoutPlaneImprovement"] > 0.0
    assert result["transform"]["bounded"]


def test_plane_refinement_rejects_one_plane_ambiguity():
    rng = np.random.default_rng(7)
    target = np.column_stack((rng.uniform(0.0, 15.0, size=(700, 2)), np.zeros(700)))
    source = target + np.array([0.2, -0.1, 0.2])
    result = _plane_fit(source, target, np.eye(3), np.zeros(3))
    assert result["status"] == "rejected"
