import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "regen"))
from compile_source_semantics import (apply_open_leaf, project_opening_to_perspective,
                                     unique_roi_match, _apply_pattern)


def test_broad_garage_roi_matching_multiple_openings_abstains():
    projections = [[100, 100, 220, 220], [205, 100, 330, 220], [315, 100, 440, 220]]
    result = unique_roi_match(projections, [100, 100, 440, 220])
    assert result["status"] == "abstain"
    assert result["reason"] in {"no substantial opening/ROI overlap", "ROI matches multiple openings"}


def test_only_explicit_high_confidence_unambiguous_open_state_changes_leaf():
    profile = {"openings": [{"style": {"kind": "rollup", "doorKind": "panel"}}]}
    match = {"status": "matched", "openingIndex": 0}
    assert not apply_open_leaf(profile, 0, {"state": "open", "confidence": "medium"}, match)
    assert not apply_open_leaf(profile, 0, {"state": "open", "confidence": "high", "ambiguity": "rear door"}, match)
    assert profile["openings"][0]["style"]["doorKind"] == "panel"
    assert apply_open_leaf(profile, 0, {"state": "open", "confidence": "high"}, match)
    assert profile["openings"][0]["style"]["openLeaf"] is True


def test_pattern_requires_same_family_primary_surface_exact_source_and_high_confidence():
    profile = {"material": "timber_dark#2e3170", "materialEvidence": {
        "family": "timber", "colorRgb": [46, 49, 112], "sourceIds": ["mapillary:img1"]}}
    packet = {"image": "img1", "sourceSha256": "hash1"}
    valid = {"field": "materialEvidence.pattern", "value": "horizontal_courses", "confidence": "high",
             "surfaceScope": "primary_material_surface", "imageId": "img1", "imageSha256": "hash1"}
    changed, reason = _apply_pattern(profile, valid, packet)
    assert changed and reason == ""
    assert profile["materialEvidence"]["pattern"] == "horizontal_courses"
    assert "pat=horizontal_courses" in profile["material"]
    assert profile["materialEvidence"]["colorRgb"] == [46, 49, 112]


def test_pattern_rejects_partial_or_source_mismatched_evidence():
    profile = {"material": "timber_dark#2e3170", "materialEvidence": {
        "family": "timber", "colorRgb": [46, 49, 112], "sourceIds": ["mapillary:img1"]}}
    packet = {"image": "img1", "sourceSha256": "hash1"}
    partial = {"field": "materialEvidence.pattern", "value": "horizontal_courses", "confidence": "medium",
               "surfaceScope": "local_partial", "imageId": "img1", "imageSha256": "hash1"}
    assert not _apply_pattern(profile, partial, packet)[0]
    mismatch = {**partial, "confidence": "high", "surfaceScope": "primary_material_surface", "imageSha256": "wrong"}
    assert not _apply_pattern(profile, mismatch, packet)[0]
    assert "pattern" not in profile["materialEvidence"]


def test_camera_projection_rejects_points_behind_camera_and_projects_world_facade():
    profile = {"normal": [0, -1], "frameU": [-2, 2], "planeV": 0, "heightRefM": 10}
    opening = {"rect": [.25, 0, .75, .5], "style": {"kind": "rollup"}}
    # Facing north from ten metres away; 90-degree horizontal FOV.
    packet = {"cameraXY": [0, -10], "cameraGroundZ": 0}
    box = project_opening_to_perspective(profile, opening, ground_z=0, packet=packet,
                                         image_size=(1000, 1000), eye_height_m=1.65)
    assert box is not None
    assert 300 < box[0] < 400 and 600 < box[2] < 700
    assert box[1] < 500 < box[3]
    behind = {**packet, "cameraXY": [0, 10]}
    assert project_opening_to_perspective(profile, opening, ground_z=0, packet=behind,
                                          image_size=(1000, 1000), eye_height_m=1.65) is None


def test_default_camera_projection_uses_existing_crop_eye_height_convention():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frontage_crops_v15 import EYE_HEIGHT_M
    profile = {"normal": [0, -1], "frameU": [-2, 2], "planeV": 0, "heightRefM": 10}
    opening = {"rect": [.25, 0, .75, .5], "style": {"kind": "rollup"}}
    packet = {"cameraXY": [0, -10], "cameraGroundZ": 0}
    implicit = project_opening_to_perspective(profile, opening, ground_z=0, packet=packet,
                                              image_size=(1000, 1000))
    explicit = project_opening_to_perspective(profile, opening, ground_z=0, packet=packet,
                                              image_size=(1000, 1000), eye_height_m=EYE_HEIGHT_M)
    assert np.allclose(implicit, explicit)
