from __future__ import annotations

import copy
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from general_world.facade_capture_selection import build_facade_frames, select_facade_captures


AS_OF = "2026-01-01"


def _world(buildings):
    return {
        "snapshotDate": AS_OF,
        "coordinateFrame": {"projectedCrs": "EPSG:32631", "originProjectedM": [500000.0, 5000000.0]},
        "buildings": buildings,
        "images": {},
    }


def _coverage(records):
    return {"images": records}


def _image(image_id, lonlat, *, sequence="seq", heading=None, pano=True, captured="2025-12-01"):
    result = {
        "id": image_id,
        "tile_location": list(lonlat),
        "captured_at": int(date.fromisoformat(captured).strftime("%s")) * 1000,
        "is_pano": pano,
        "quality_score": 1.0,
        "sequence_id": sequence,
    }
    if heading is not None:
        result["compass_angle"] = heading
    return result


def _local_to_lonlat(x, y):
    # EPSG:32631 point conversion used only to make compact synthetic fixtures.
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    return t.transform(500000.0 + x, 5000000.0 + y)


def test_front_back_and_budget_are_geometry_driven():
    building = {"key": "opaque-a", "heightM": 12, "geometry": {"type": "Polygon", "coordinates": [[[10, -5], [20, -5], [20, 5], [10, 5], [10, -5]]]}}
    west = _local_to_lonlat(0, 0)
    east = _local_to_lonlat(30, 0)
    plan = select_facade_captures(
        _coverage([_image("west", west, sequence="west"), _image("east", east, sequence="east")]),
        _world([building]),
        limit=1,
    )
    # Equal-distance opposite frontages are equally useful; projection
    # roundoff must not make one particular opaque camera identity the oracle.
    assert len(plan["selectedSourceIDs"]) == 1
    assert plan["selectedSourceIDs"][0] in {"west", "east"}
    assert plan["planReport"]["selectedCount"] <= 1


def test_foreign_footprint_occludes_line_of_sight():
    target = {"key": "target", "heightM": 10, "geometry": {"type": "Polygon", "coordinates": [[[12, -4], [18, -4], [18, 4], [12, 4], [12, -4]]]}}
    blocker = {"key": "blocker", "heightM": 10, "geometry": {"type": "Polygon", "coordinates": [[[5, -2], [8, -2], [8, 2], [5, 2], [5, -2]]]}}
    camera = _local_to_lonlat(0, 0)
    plan = select_facade_captures(_coverage([_image("blocked", camera)]), _world([target, blocker]), limit=4)
    assert plan["selectedSourceIDs"] == ["blocked"]
    assert not any(frame_id.startswith("target:") for frame_id in plan["selection"][0]["coveredFrameIDs"])


def test_second_independent_sequence_is_rewarded_and_name_changes_do_not_matter():
    building = {"key": "building-key", "heightM": 9, "geometry": {"type": "Polygon", "coordinates": [[[10, -5], [20, -5], [20, 5], [10, 5], [10, -5]]]}}
    camera = _local_to_lonlat(0, 0)
    coverage = _coverage([
        _image("a1", camera, sequence="a"),
        _image("a2", _local_to_lonlat(0, 1), sequence="a"),
        _image("b1", _local_to_lonlat(0, -1), sequence="b"),
    ])
    first = select_facade_captures(coverage, _world([building]), limit=2)
    renamed = copy.deepcopy(building)
    renamed["key"] = "different-opaque-key"
    second = select_facade_captures(coverage, _world([renamed]), limit=2)
    assert first["selectedSourceIDs"] == second["selectedSourceIDs"]
    assert {item["sequenceID"] for item in first["selection"]} == {"a", "b"}
    assert first["planReport"]["selectedCount"] == 2


def test_concave_owner_mass_occludes_its_inner_facade():
    # The ray from the camera to the inner bottom of this U-shaped owner
    # passes through the owner's left arm before reaching the facade edge.
    owner = {
        "key": "concave-owner",
        "heightM": 10,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 0], [10, 10], [7, 10], [7, 3], [3, 3], [3, 10], [0, 10], [0, 0]]],
        },
    }
    camera = _local_to_lonlat(1, 20)
    plan = select_facade_captures(_coverage([_image("concave-view", camera)]), _world([owner]), limit=4)
    frames, _, _ = build_facade_frames([owner])
    inner = {frame["frameID"] for frame in frames if abs(frame["midpoint"][1] - 3.0) < 1e-6 and 3 < frame["midpoint"][0] < 7}
    assert inner
    covered = {frame_id for item in plan["selection"] for frame_id in item["coveredFrameIDs"]}
    assert not inner.intersection(covered)


def test_missing_sequence_ids_do_not_count_as_independent_sequences():
    building = {"key": "opaque", "heightM": 9, "geometry": {"type": "Polygon", "coordinates": [[[10, -5], [20, -5], [20, 5], [10, 5], [10, -5]]]}}
    camera = _local_to_lonlat(0, 0)
    coverage = _coverage([
        _image("unknown-a", _local_to_lonlat(0, 0), sequence=None),
        _image("unknown-b", _local_to_lonlat(0, 20), sequence=None),
    ])
    for record in coverage["images"]:
        record.pop("sequence_id")
    plan = select_facade_captures(coverage, _world([building]), limit=2)
    assert plan["planReport"]["selectedCount"] == 2
    assert all(item["sequenceID"] is None for item in plan["selection"])
    assert max(plan["planReport"]["independentSequenceCounts"].values()) <= 1


def test_nonfinite_catalog_location_is_rejected():
    building = {"key": "opaque", "heightM": 9, "geometry": {"type": "Polygon", "coordinates": [[[10, -5], [20, -5], [20, 5], [10, 5], [10, -5]]]}}
    malformed = _image("nan", _local_to_lonlat(0, 0))
    malformed["tile_location"] = [float("nan"), 0.0]
    plan = select_facade_captures(_coverage([malformed]), _world([building]), limit=2)
    assert plan["selectedSourceIDs"] == []
    assert plan["planReport"]["candidateCounts"]["rejected"]["missingLocation"] == 1


def test_street_level_height_weighting_caps_highrise_and_assumes_unknown():
    footprint = {"type": "Polygon", "coordinates": [[[10, -5], [20, -5], [20, 5], [10, 5], [10, -5]]]}
    unknown = {"key": "unknown-height", "heightM": None, "geometry": footprint}
    highrise = {"key": "known-highrise", "heightM": 80, "geometry": footprint}
    unknown_frames, _, _ = build_facade_frames([unknown])
    highrise_frames, _, _ = build_facade_frames([highrise])
    assert unknown_frames[0]["planningHeightAssumed"] is True
    assert unknown_frames[0]["heightM"] is None
    assert highrise_frames[0]["planningHeightAssumed"] is False
    assert highrise_frames[0]["heightM"] == 80
    assert unknown_frames[0]["planningHeightM"] == highrise_frames[0]["planningHeightM"] == 12
    assert unknown_frames[0]["visibleAreaM2"] == highrise_frames[0]["visibleAreaM2"]
