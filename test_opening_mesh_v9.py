import pytest
from shapely.geometry import Polygon

from general_world.opening_mesh import ApertureError, compile_wall


def wall():
    return Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])


def test_arch_hole_is_preserved():
    arch = [(2, 0), (4, 0), (4, 3), (3, 4), (2, 3)]
    out = compile_wall(wall(), [{"kind": "window", "boundary_polygon": arch,
                                 "frame_width_m": .12}], .3)
    assert out["checks"]["closed_manifold"]
    assert out["apertures"][0]["hole_area_m2"] == pytest.approx(7.0)


def test_mullion_is_clipped_to_nonrectangular_glazing():
    poly = [(1, 1), (5, 1), (5, 2), (3, 4), (1, 2)]
    out = compile_wall(wall(), [{"kind": "window", "boundary_polygon": poly,
                                 "frame_width_m": .2,
                                 "mullion_segments": [[(3, 0), (3, 6)]]}], .25)
    assert any("mullion" in p["label"] for p in out["parts"])
    assert out["checks"]["closed_manifold"]


def test_door_has_no_unmeasured_threshold():
    out = compile_wall(wall(), [{"kind": "door", "boundary_polygon": [(1, 0), (3, 0), (3, 4), (1, 4)],
                                 "frame_width_m": .1}], .3)
    assert not any("threshold" in p["label"] for p in out["parts"])


def test_open_passage_has_no_backing_panel():
    out = compile_wall(wall(), [{"kind": "open_passage", "boundary_polygon": [(4, 0), (6, 0), (6, 3), (4, 3)],
                                 "frame_width_m": 0}], .3)
    assert not any("glazing" in p["label"] for p in out["parts"])
    assert out["checks"]["opening_area_m2"] == pytest.approx(6)


def test_bad_overlap_is_rejected():
    aps = [{"kind": "window", "boundary_polygon": [(1, 1), (4, 1), (4, 3), (1, 3)]},
           {"kind": "door", "boundary_polygon": [(3, 2), (5, 2), (5, 4), (3, 4)]}]
    with pytest.raises(ApertureError, match="overlaps"):
        compile_wall(wall(), aps, .3)


def test_nonfinite_thickness_is_rejected():
    with pytest.raises(ApertureError, match="finite"):
        compile_wall(wall(), [], float("nan"))


def test_passage_cannot_gain_measured_leaf():
    with pytest.raises(ApertureError, match="only legal for door"):
        compile_wall(wall(), [{"kind": "open_passage",
                              "boundary_polygon": [(4, 0), (6, 0), (6, 3), (4, 3)],
                              "leaf_polygon": [(4, 0), (6, 0), (6, 3), (4, 3)]}], .3)


def test_detached_threshold_is_rejected():
    with pytest.raises(ApertureError, match="detached"):
        compile_wall(wall(), [{"kind": "door",
                              "boundary_polygon": [(1, 0), (3, 0), (3, 4), (1, 4)],
                              "threshold_polygon": [(1.2, 1), (2.8, 1), (2.8, 1.2), (1.2, 1.2)]}], .3)
