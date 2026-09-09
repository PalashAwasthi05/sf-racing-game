import pytest
from general_world.detail_compiler import DetailCompilerError, compile_detail_records


def record(**changes):
    value = {
        "evidence_approved": True,
        "wall_polygon": [(0, 0), (6, 0), (6, 4), (0, 4)],
        "wall_thickness_m": .3,
        "coordinate_frame": {"origin": [10, 20, 2], "horizontal": [1, 0, 0],
                              "inward": [0, 1, 0], "vertical": [0, 0, 1]},
        "apertures": [{"kind": "window", "boundary_polygon": [(1, 1), (3, 1), (3, 3), (1, 3)],
                       "frame_width_m": .1}],
    }
    value.update(changes)
    return value


def test_compiles_measured_wall_in_world_frame():
    out = compile_detail_records([record()])[0]
    assert out["checks"]["closed_manifold"]
    assert out["parts"][0]["mesh"].vertices[:, 0].min() >= 10
    assert out["frame"].startswith("world:")


@pytest.mark.parametrize("change, message", [
    ({"evidence_approved": False}, "evidence-approved"),
    ({"coordinate_frame": {"origin": [0, 0, 0]}}, "requires origin"),
    ({"wall_thickness_m": None}, "finite"),
])
def test_rejects_unapproved_or_incomplete_records(change, message):
    with pytest.raises(DetailCompilerError, match=message):
        compile_detail_records([record(**change)])


def test_rejects_aperture_without_observed_boundary():
    with pytest.raises(DetailCompilerError, match="boundary_polygon"):
        compile_detail_records([record(apertures=[{"kind": "window"}])])

