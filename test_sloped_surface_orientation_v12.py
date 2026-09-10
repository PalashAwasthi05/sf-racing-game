import numpy as np
import trimesh
from shapely.geometry import Polygon

from general_world.sloped_surface_orientation import audit_sloped_caps


FP = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])


def sloped_patch(z=10.0, slope=0.4):
    vertices = [[0, 0, z], [4, 0, z + 4 * slope], [4, 4, z + 4 * slope], [0, 4, z]]
    # Reverse winding gives a downward-facing sloped cap.
    return trimesh.Trimesh(vertices, [[0, 2, 1], [0, 3, 2]], process=False)


def roof_points(z=10.0, slope=0.4):
    return np.asarray([[x + 0.1, y + 0.1, z + slope * (x + 0.1)] for x in range(4) for y in range(4)])


def wedge_solid():
    bottom = [[0, 0, 10], [4, 0, 11.6], [4, 4, 11.6], [0, 4, 10]]
    top = [[0, 0, 11], [4, 0, 12.6], [4, 4, 12.6], [0, 4, 11]]
    vertices = bottom + top
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return trimesh.Trimesh(vertices, faces, process=False)


def test_supported_reversed_sloped_roof_is_repairable():
    result = audit_sloped_caps(sloped_patch(), [], FP, roof_points(), 5.0)
    assert result["repairable"]
    assert result["reverse_face_indices"] == [0, 1]


def test_canopy_underside_with_points_below_plane_is_rejected():
    result = audit_sloped_caps(sloped_patch(), [], FP, roof_points(z=9.7), 5.0)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "insufficient-distributed-lidar"


def test_closed_sloped_solid_is_rejected():
    result = audit_sloped_caps(wedge_solid(), [], FP, roof_points(), 5.0)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "closed-solid"


def test_overlying_roof_is_rejected():
    overlying = sloped_patch(z=10.6)
    overlying.invert()
    result = audit_sloped_caps(sloped_patch(), [overlying], FP, roof_points(), 5.0)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "overlying-upward-surface"


def test_thin_positive_clearance_roof_is_rejected():
    overlying = sloped_patch(z=10.01)
    overlying.invert()
    result = audit_sloped_caps(sloped_patch(), [overlying], FP, roof_points(), 5.0)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "overlying-upward-surface"


def test_high_ceiling_above_candidate_is_rejected():
    overlying = sloped_patch(z=13.0)
    overlying.invert()
    result = audit_sloped_caps(sloped_patch(), [overlying], FP, roof_points(), 5.0)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "overlying-upward-surface"


def test_insufficient_support_is_rejected():
    result = audit_sloped_caps(sloped_patch(), [], FP, roof_points()[:3], 5.0)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "insufficient-distributed-lidar"
