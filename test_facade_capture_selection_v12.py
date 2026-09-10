from __future__ import annotations

import copy
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from general_world.facade_capture_selection import select_facade_captures


AS_OF = "2026-01-01"


def _world(buildings):
    return {
        "snapshotDate": AS_OF,
        "coordinateFrame": {"projectedCrs": "EPSG:32610", "originProjectedM": [553556.0, 4181427.0]},
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
    # EPSG:32610 point conversion used only to make compact synthetic fixtures.
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    return t.transform(553556.0 + x, 4181427.0 + y)


def test_front_back_and_budget_are_geometry_driven():
    building = {"key": "opaque-a", "heightM": 12, "geometry": {"type": "Polygon", "coordinates": [[[10, -5], [20, -5], [20, 5], [10, 5], [10, -5]]]}}
    west = _local_to_lonlat(0, 0)
    east = _local_to_lonlat(30, 0)
    plan = select_facade_captures(
        _coverage([_image("west", west, sequence="west"), _image("east", east, sequence="east")]),
        _world([building]),
        limit=1,
    )
    assert plan["selectedSourceIDs"] == ["west"]
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
