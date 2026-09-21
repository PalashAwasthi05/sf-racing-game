import json
import math

import numpy as np

from infer_cycle_paint_v21 import (
    GroundSample,
    _pair_supported_pattern,
    classify_green_pattern,
    green_patch_score,
    project_ground_samples,
)


def test_green_patch_accepts_faded_green_and_rejects_gray_and_foliage_like_mixture():
    faded = np.full((5, 5, 3), [132, 158, 114], dtype=np.uint8)
    gray = np.full((5, 5, 3), [145, 148, 146], dtype=np.uint8)
    foliage = np.full((5, 5, 3), [48, 116, 38], dtype=np.uint8)
    foliage[::2, ::2] = [8, 20, 5]
    assert green_patch_score(faded) > 0.9
    assert green_patch_score(gray) < 0.1
    # Mixed high-contrast foliage is deliberately below the paint majority guard.
    assert green_patch_score(foliage) < 0.52


def test_pattern_requires_a_spatial_run_and_distinguishes_dashed_and_solid():
    assert classify_green_pattern([False, True, False, True, False, True, False]) == "dashed"
    assert classify_green_pattern([False, True, True, True, True, False]) == "solid"
    assert classify_green_pattern([False, True, False]) is None
    assert classify_green_pattern([True, True]) is None


def test_projection_uses_angle_axis_and_rejects_lower_vehicle_overlay_band():
    # Zero angle-axis is an ENU-to-camera identity.  North projects below the
    # horizon, while a far sample projects into the excluded v>=.70 band.
    samples = [
        GroundSample("way", 0.0, 0.000045, 0.000027, 0.0),
        GroundSample("way", 20.0, 0.000054, 0.000180, 0.0),
    ]
    meta = {
        "computed_geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
        "computed_rotation": [0.0, 0.0, 0.0],
    }
    projected = project_ground_samples(samples, meta)
    assert len(projected) == 1
    assert 0.5 < projected[0][2] < 0.7


def test_projection_handles_camera_separation_without_claiming_registration():
    sample = GroundSample("way", 1.0, 0.000090, 0.000027, 0.0)
    first = {
        "computed_geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
        "computed_rotation": [0.0, 0.0, 0.0],
    }
    second = {
        "computed_geometry": {"type": "Point", "coordinates": [0.000045, 0.0]},
        "computed_rotation": [0.0, 0.0, 0.0],
    }
    assert project_ground_samples([sample], first)
    assert project_ground_samples([sample], second)


def test_disjoint_green_runs_from_two_views_do_not_promote_a_style():
    first = {
        (0.0, 0.0): False,
        (1.5, 0.0): True,
        (3.0, 0.0): True,
        (4.5, 0.0): False,
    }
    second = {
        (0.0, 0.0): False,
        (1.5, 0.0): False,
        (3.0, 0.0): False,
        (4.5, 0.0): True,
        (6.0, 0.0): True,
    }
    pattern, matched, overlap = _pair_supported_pattern(first, second)
    assert pattern is None
    assert matched == []
    assert overlap == (0.0, 4.5)


def test_lane_like_evidence_keeps_side_offsets_separate():
    left = {
        (0.0, -1.6): False,
        (1.5, -1.6): True,
        (3.0, -1.6): True,
        (4.5, -1.6): True,
        (6.0, -1.6): False,
        (1.5, 1.6): False,
        (3.0, 1.6): False,
        (4.5, 1.6): False,
    }
    right = {
        (0.0, -1.6): False,
        (1.5, -1.6): True,
        (3.0, -1.6): True,
        (4.5, -1.6): True,
        (6.0, -1.6): False,
        (1.5, 1.6): True,
        (3.0, 1.6): True,
        (4.5, 1.6): True,
    }
    left_pattern, _, _ = _pair_supported_pattern(left, right, lane_like=True, side=-1)
    right_pattern, _, _ = _pair_supported_pattern(left, right, lane_like=True, side=1)
    assert left_pattern == "solid"
    assert right_pattern is None
