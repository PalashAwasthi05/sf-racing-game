import json

from regen.reading_provenance import load_registered_sides, missing_crop_files, relative_path, select_source_view


def _piece(image, views=None, *, length=8.0, height=6.0):
    return {
        "id": "side-a",
        "lengthM": length,
        "heightM": height,
        "frameU": [0.0, length],
        "normal": [1.0, 0.0],
        "planeV": 0.0,
        "views": views if views is not None else [{"image": image, "rect": "nominal-rect", "persp": "nominal-persp"}],
    }


def test_registered_reading_uses_registered_camera_identity(tmp_path):
    nominal = _piece("nominal-camera")
    registered = _piece("registered-camera", [
        {"image": "registered-camera", "rect": "registered-rect", "persp": "registered-persp"},
    ])
    (tmp_path / "sides-reg.json").write_text(json.dumps({"pieces": [registered]}))

    sides = load_registered_sides(tmp_path)
    selected = select_source_view(nominal, sides, 0, registered_required=True)

    assert selected["status"] == "ok"
    assert selected["view"]["image"] == "registered-camera"
    assert selected["sourceIds"] == ["mapillary:registered-camera"]
    assert selected["sourceIds"] != ["mapillary:nominal-camera"]


def test_missing_registered_crop_is_unresolved_without_nominal_fallback(tmp_path):
    nominal = _piece("nominal-camera")
    registered = _piece("registered-camera", [])
    (tmp_path / "sides-reg.json").write_text(json.dumps({"pieces": [registered]}))

    selected = select_source_view(nominal, load_registered_sides(tmp_path), 0, registered_required=True)

    assert selected["status"] == "unresolved"
    assert selected["reason"] == "registered crop view missing"
    assert "sourceIds" not in selected
    assert "view" not in selected


def test_registered_frame_mismatch_is_unresolved(tmp_path):
    nominal = _piece("nominal-camera")
    registered = _piece("registered-camera", length=8.1)
    (tmp_path / "sides-reg.json").write_text(json.dumps({"pieces": [registered]}))

    selected = select_source_view(nominal, load_registered_sides(tmp_path), 0, registered_required=True)

    assert selected["status"] == "unresolved"
    assert selected["reason"] == "registered frame lengthM mismatch"


def test_registered_world_frame_mismatch_is_unresolved(tmp_path):
    nominal = {**_piece("nominal-camera"), "frameU": [10.0, 18.0], "normal": [1.0, 0.0], "planeV": 4.0}
    registered = {**_piece("registered-camera"), "frameU": [10.0, 18.0], "normal": [0.0, 1.0], "planeV": 4.0}
    (tmp_path / "sides-reg.json").write_text(json.dumps({"pieces": [registered]}))

    selected = select_source_view(nominal, load_registered_sides(tmp_path), 0, registered_required=True)

    assert selected["status"] == "unresolved"
    assert selected["reason"] == "registered frame normal mismatch"


def test_missing_crop_names_and_files_are_explicit(tmp_path):
    view = {"image": "camera", "rect": "r.jpg", "persp": "p.jpg"}
    selected = {"view": view}
    assert missing_crop_files(selected, tmp_path) == ["rect file r.jpg", "persp file p.jpg"]
    (tmp_path / "r.jpg").touch()
    assert missing_crop_files(selected, tmp_path) == ["persp file p.jpg"]


def test_relative_path_retains_evidence_outside_area_root(tmp_path):
    project = tmp_path / "project"
    area = project / "areas" / "demo"
    evidence = project / "areas" / "review" / "source.json"
    evidence.parent.mkdir(parents=True)
    evidence.touch()

    assert relative_path(evidence, area) == "../review/source.json"


def test_duplicate_records_prefer_more_views_and_report_tied_ambiguity(tmp_path):
    one_view = _piece("camera-a")
    two_views = _piece("camera-b", [
        {"image": "camera-b", "rect": "b-r.jpg", "persp": "b-p.jpg"},
        {"image": "camera-c", "rect": "c-r.jpg", "persp": "c-p.jpg"},
    ])
    (tmp_path / "sides-0.json").write_text(json.dumps({"pieces": [one_view]}))
    (tmp_path / "sides-1.json").write_text(json.dumps({"pieces": [two_views]}))
    selected = select_source_view(_piece("nominal"), load_registered_sides(tmp_path), 1, registered_required=True)
    assert selected["sourceIds"] == ["mapillary:camera-c"]

    two_views_conflict = _piece("camera-d", [
        {"image": "camera-d"}, {"image": "camera-e"},
    ])
    (tmp_path / "sides-2.json").write_text(json.dumps({"pieces": [two_views_conflict]}))
    ambiguous = select_source_view(_piece("nominal"), load_registered_sides(tmp_path), 0, registered_required=True)
    assert ambiguous["status"] == "unresolved"
    assert ambiguous["reason"] == "registered side record ambiguous"
