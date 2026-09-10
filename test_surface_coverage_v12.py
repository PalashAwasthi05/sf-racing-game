import pytest
import trimesh
from shapely.geometry import box

from general_world.surface_coverage import SurfaceCoverageError, audit_mesh_projection, audit_surface_coverage, replacement_transaction


def frame():
    return {"origin": [0, 0, 0], "horizontal": [1, 0, 0], "vertical": [0, 0, 1], "normal": [0, 1, 0]}


def test_generic_56m2_host_gap_is_reported():
    result = audit_surface_coverage(box(0, 0, 10, 8), [box(2, 2, 8, 6)])
    assert result["missing_area"] == pytest.approx(56)


def test_owned_contained_aperture_is_excluded_without_fill():
    result = audit_surface_coverage(box(0, 0, 6, 4), [box(0, 0, 1, 4), box(5, 0, 6, 4), box(1, 0, 5, 1), box(1, 3, 5, 4)],
                                    [{"polygon": box(1, 1, 5, 3), "owned": True}])
    assert result["missing_area"] == pytest.approx(0)


def test_narrow_012m_return_is_not_buffered_away():
    result = audit_surface_coverage(box(0, 0, 1, 1), [box(0, 0, .44, 1), box(.56, 0, 1, 1)])
    assert result["missing_area"] == pytest.approx(.12)


def test_transaction_keeps_missing_host_and_only_replaces_exact_union():
    result = replacement_transaction(box(0, 0, 10, 8), [box(2, 2, 8, 6)])
    assert result["replaceable_union"].area == pytest.approx(24)
    assert result["retained_host_remainder"].area == pytest.approx(56)


def test_mesh_rejects_backwards_and_incorrect_depth_triangles():
    good = trimesh.Trimesh(vertices=[[0, 0, 0], [2, 0, 0], [0, 0, 2]], faces=[[0, 1, 2]], process=False)
    bad_depth = trimesh.Trimesh(vertices=[[2, .2, 0], [4, .2, 0], [2, .2, 2]], faces=[[0, 1, 2]], process=False)
    bad_back = trimesh.Trimesh(vertices=[[2, 0, 0], [2, 0, 2], [4, 0, 0]], faces=[[0, 1, 2]], process=False)
    mesh = trimesh.util.concatenate([good, bad_depth, bad_back])
    result = audit_mesh_projection(mesh, box(0, 0, 4, 2), frame())
    assert result["accepted_triangles"] == 1
    assert result["rejected_depth_triangles"] == 1
    assert result["rejected_normal_triangles"] == 1
    assert result["missing_area"] == pytest.approx(6)


def test_overlapping_emitted_surfaces_are_detected():
    result = audit_surface_coverage(box(0, 0, 4, 2), [box(0, 0, 3, 2), box(1, 0, 4, 2)])
    assert result["overlap_area"] == pytest.approx(4)


@pytest.mark.parametrize("kwargs", [
    {"intentional_openings": [box(5, 5, 6, 6)]},
    {"intentional_openings": [{"polygon": box(0, 0, 1, 1), "owned": False}]},
])
def test_invalid_or_unowned_openings_are_rejected(kwargs):
    with pytest.raises(SurfaceCoverageError):
        audit_surface_coverage(box(0, 0, 4, 4), [box(0, 0, 4, 4)], **kwargs)
