"""Geometry-first selection of recent street imagery for facade frames.

The selector consumes two deliberately small contracts: a coverage manifest
whose image locations are WGS84 lon/lat, and a normalized world input whose
building geometry is in an explicit projected frame.  Building names and
locality labels are never read.  Opaque building keys are carried only so a
selected image can be joined to the frame it covers.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, shape
from shapely.strtree import STRtree

from .evidence import contiguous_frames


RECENT_DAYS = 730
MAX_DISTANCE_M = 80.0
MIN_FRONT_FACING = 0.5
OWNER_EDGE_TOLERANCE_M = 0.5
DEFAULT_FRAME_LENGTH_M = 12.0


def _as_images(coverage: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(coverage, dict):
        images = coverage.get("images", coverage.get("records", []))
    else:
        images = coverage
    if isinstance(images, dict):
        return [dict(value, id=value.get("id", key)) for key, value in images.items()]
    return [dict(value) for value in images]


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).date()
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return date.fromisoformat(text[:10])


def _capture_date(record: dict[str, Any]) -> date | None:
    value = record.get("captured_at", record.get("capturedAt", record.get("captureDate")))
    if value is None:
        return None
    try:
        return _as_date(value)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _source_id(record: dict[str, Any]) -> str:
    value = record.get("id", record.get("sourceID", record.get("sourceId")))
    if value is None:
        raise ValueError("coverage image is missing opaque id")
    return str(value)


def _projector(normalized_input: dict[str, Any]) -> tuple[Transformer, np.ndarray, str]:
    frame = normalized_input.get("coordinateFrame") or {}
    crs = frame.get("projectedCrs")
    origin = frame.get("originProjectedM")
    if not crs or origin is None or len(origin) != 2:
        raise ValueError("normalized input requires coordinateFrame.projectedCrs and originProjectedM")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True), np.asarray(origin, dtype=float), str(crs)


def _local_point(transformer: Transformer, origin: np.ndarray, lonlat: Iterable[float]) -> np.ndarray:
    values = list(lonlat)
    if len(values) != 2:
        raise ValueError("tile_location must be [longitude, latitude]")
    try:
        lon, lat = float(values[0]), float(values[1])
    except (TypeError, ValueError):
        raise ValueError("tile_location must contain finite numbers") from None
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise ValueError("tile_location must contain finite numbers")
    x, y = transformer.transform(lon, lat)
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise ValueError("projected image location is not finite")
    return np.array([float(x) - origin[0], float(y) - origin[1]], dtype=float)


def _split_frame(frame: dict[str, Any], max_length: float, index: int) -> list[dict[str, Any]]:
    a = np.asarray(frame["a"], dtype=float)
    b = np.asarray(frame["b"], dtype=float)
    length = float(np.linalg.norm(b - a))
    count = max(1, int(math.ceil(length / max_length)))
    result = []
    for part in range(count):
        lo, hi = part / count, (part + 1) / count
        result.append(
            {
                "index": index,
                "part": part,
                "a": (a * (1 - lo) + b * lo).tolist(),
                "b": (a * (1 - hi) + b * hi).tolist(),
                "normal": list(frame["normal"]),
                "lengthM": length / count,
                "courtyard": bool(frame.get("courtyard", False)),
            }
        )
    return result


def build_facade_frames(
    buildings: list[dict[str, Any]],
    *,
    frame_length_m: float = DEFAULT_FRAME_LENGTH_M,
    min_length_m: float = 1.0,
) -> tuple[list[dict[str, Any]], list[Any], list[str]]:
    """Create contiguous, height-weighted frames from footprint boundaries."""

    if frame_length_m <= 0 or min_length_m <= 0:
        raise ValueError("frame lengths must be positive")
    frames: list[dict[str, Any]] = []
    footprints = []
    owner_keys = []
    for building_index, building in enumerate(buildings):
        key = str(building.get("key", f"building-{building_index}"))
        footprint = shape(building["geometry"])
        if footprint.is_empty:
            continue
        footprints.append(footprint)
        owner_keys.append(key)
        raw_height = building.get("heightM", 0.0)
        height = max(0.1, float(raw_height or 0.0))
        source_frames = contiguous_frames(footprint, min_length=min_length_m)
        edge_index = 0
        for source in source_frames:
            for part in _split_frame(source, frame_length_m, edge_index):
                part["frameID"] = f"{key}:frame:{edge_index}:{part['part']}"
                part["buildingKey"] = key
                part["ownerIndex"] = len(footprints) - 1
                part["heightM"] = height
                part["visibleAreaM2"] = float(part["lengthM"] * height)
                line = LineString([part["a"], part["b"]])
                part["midpoint"] = [float(line.centroid.x), float(line.centroid.y)]
                frames.append(part)
            edge_index += 1
    return frames, footprints, owner_keys


def _heading_vector(compass_angle: Any) -> np.ndarray | None:
    if compass_angle is None:
        return None
    try:
        theta = math.radians(float(compass_angle))
    except (TypeError, ValueError):
        return None
    return np.array([math.sin(theta), math.cos(theta)], dtype=float)


def _line_clear(
    camera: np.ndarray,
    sample: np.ndarray,
    owner_index: int,
    footprints: list[Any],
    footprint_tree: STRtree,
    owner_edge_tolerance: float,
) -> bool:
    line = LineString([camera.tolist(), sample.tolist()])
    if line.length <= 1e-6:
        return False
    # The owner may legitimately occupy the final facade edge, but it must
    # not occlude the ray before that edge (important for concave footprints).
    owner = footprints[owner_index]
    before_owner_edge = line.difference(Point(sample.tolist()).buffer(owner_edge_tolerance))
    if owner.buffer(-1e-7).intersects(before_owner_edge):
        return False
    for raw_index in footprint_tree.query(line):
        index = int(raw_index)
        if index == owner_index:
            continue
        foreign = footprints[index]
        intersection = line.intersection(foreign)
        if intersection.is_empty:
            continue
        # A tangent/edge contact is harmless; a material crossing means the
        # view is occluded by another building footprint.
        if float(intersection.length) > owner_edge_tolerance:
            return False
        if foreign.buffer(-1e-7).intersects(line):
            return False
    return True


def _visible_pair(
    record: dict[str, Any],
    frame: dict[str, Any],
    footprints: list[Any],
    footprint_tree: STRtree,
    *,
    max_distance_m: float,
    min_front_facing: float,
    owner_edge_tolerance: float,
) -> dict[str, Any] | None:
    camera = np.asarray(record["point"], dtype=float)
    midpoint = np.asarray(frame["midpoint"], dtype=float)
    owner = footprints[int(frame["ownerIndex"])]
    if owner.contains(Point(float(camera[0]), float(camera[1]))):
        return None
    delta = midpoint - camera
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6 or distance > max_distance_m:
        return None
    normal = np.asarray(frame["normal"], dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-9)
    to_camera = -delta / distance
    geometric_front = max(0.0, float(np.dot(normal, to_camera)))
    if geometric_front < min_front_facing:
        return None
    # Panoramas are omnidirectional.  Perspective records are additionally
    # checked against their compass heading when one is present.
    heading = _heading_vector(record.get("compass_angle", record.get("compassAngle")))
    heading_front = None if bool(record.get("is_pano", False)) else (float(np.dot(heading, delta / distance)) if heading is not None else None)
    front_facing = geometric_front if heading_front is None else max(0.0, min(geometric_front, heading_front))
    if front_facing < min_front_facing:
        return None
    a, b = np.asarray(frame["a"], dtype=float), np.asarray(frame["b"], dtype=float)
    samples = [a * (1 - t) + b * t for t in (0.2, 0.5, 0.8)]
    if not all(_line_clear(camera, sample, int(frame["ownerIndex"]), footprints, footprint_tree, owner_edge_tolerance) for sample in samples):
        return None
    quality = max(0.0, min(1.0, float(record.get("quality_score", record.get("qualityScore", 0.0)) or 0.0)))
    return {
        "frameID": frame["frameID"],
        "visibleAreaM2": float(frame["visibleAreaM2"]),
        "distanceM": distance,
        "frontFacing": front_facing,
        "quality": quality,
    }


def _eligible_records(
    coverage: dict[str, Any] | list[dict[str, Any]],
    normalized_input: dict[str, Any],
    *,
    recent_days: int,
    require_panoramas: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformer, origin, crs = _projector(normalized_input)
    as_of = _as_date(normalized_input.get("snapshotDate"))
    cutoff = as_of - timedelta(days=recent_days)
    all_records = _as_images(coverage)
    records = []
    rejected = {"missingLocation": 0, "notPanorama": 0, "missingDate": 0, "stale": 0}
    for source in all_records:
        try:
            source_id = _source_id(source)
        except ValueError:
            rejected["missingLocation"] += 1
            continue
        if require_panoramas and not bool(source.get("is_pano", source.get("isPano", False))):
            rejected["notPanorama"] += 1
            continue
        captured = _capture_date(source)
        if captured is None:
            rejected["missingDate"] += 1
            continue
        if not cutoff <= captured <= as_of:
            rejected["stale"] += 1
            continue
        location = source.get("tile_location", source.get("tileLocation"))
        try:
            point = _local_point(transformer, origin, location)
        except (TypeError, ValueError, KeyError):
            rejected["missingLocation"] += 1
            continue
        record = dict(source)
        record.update({"sourceID": source_id, "point": point, "captureDate": captured.isoformat(), "projectedCRS": crs})
        records.append(record)
    return records, {"total": len(all_records), "eligible": len(records), "rejected": rejected, "cutoffDate": cutoff.isoformat()}


def _candidate_pairs(
    records: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    footprints: list[Any],
    *,
    max_distance_m: float,
    min_front_facing: float,
    owner_edge_tolerance: float,
) -> dict[str, dict[str, Any]]:
    if not records or not frames:
        return {}
    points = np.asarray([record["point"] for record in records], dtype=float)
    tree = cKDTree(points)
    footprint_tree = STRtree(footprints)
    candidates: dict[str, dict[str, Any]] = {}
    for frame in frames:
        nearby = tree.query_ball_point(np.asarray(frame["midpoint"], dtype=float), max_distance_m)
        for index in nearby:
            record = records[index]
            pair = _visible_pair(record, frame, footprints, footprint_tree, max_distance_m=max_distance_m, min_front_facing=min_front_facing, owner_edge_tolerance=owner_edge_tolerance)
            if pair is None:
                continue
            candidate = candidates.setdefault(record["sourceID"], {"record": record, "pairs": []})
            candidate["pairs"].append(pair)
    for candidate in candidates.values():
        candidate["pairs"].sort(key=lambda value: value["frameID"])
    return candidates


def _sequence_key(record: dict[str, Any], source_id: str) -> str:
    value = record.get("sequence_id", record.get("sequenceID"))
    if value is None or not str(value).strip():
        return "__unknown_sequence__"
    return str(value)


def _greedy_select(candidates: dict[str, dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    selected: list[dict[str, Any]] = []
    covered_sequences: dict[str, set[str]] = {}
    remaining = dict(candidates)
    while remaining and len(selected) < limit:
        best_id = None
        best_value = 0.0
        best_gain: dict[str, float] = {}
        for source_id, candidate in remaining.items():
            sequence = _sequence_key(candidate["record"], source_id)
            gain = 0.0
            gains = {}
            for pair in candidate["pairs"]:
                seen = covered_sequences.get(pair["frameID"], set())
                if sequence in seen:
                    continue
                reward = 1.0 if not seen else (0.65 if len(seen) < 2 else 0.08)
                pair_value = pair["visibleAreaM2"] * reward
                gains[pair["frameID"]] = gains.get(pair["frameID"], 0.0) + pair_value
                gain += pair_value
            if gain <= 0:
                continue
            quality = 0.5 + 0.5 * float(candidate["record"].get("quality_score", candidate["record"].get("qualityScore", 0.0)) or 0.0)
            distance = min((p["distanceM"] for p in candidate["pairs"]), default=MAX_DISTANCE_M)
            front = max((p["frontFacing"] for p in candidate["pairs"]), default=0.0)
            value = gain * quality * (0.65 + 0.35 * front) / (1.0 + distance / 80.0)
            tie = (0.000001 * sum(ord(char) for char in source_id))
            if value + tie > best_value:
                best_id, best_value, best_gain = source_id, value + tie, gains
        if best_id is None:
            break
        candidate = remaining.pop(best_id)
        sequence = _sequence_key(candidate["record"], best_id)
        for frame_id in best_gain:
            covered_sequences.setdefault(frame_id, set()).add(sequence)
        selected.append({"sourceID": best_id, "record": candidate["record"], "pairs": candidate["pairs"], "score": best_value})
    return selected, covered_sequences


def _distance_coverage(selected: Iterable[dict[str, Any]], frame_ids: set[str]) -> dict[str, int]:
    nearest = {frame_id: float("inf") for frame_id in frame_ids}
    for item in selected:
        for pair in item.get("pairs", []):
            nearest[pair["frameID"]] = min(nearest.get(pair["frameID"], float("inf")), float(pair["distanceM"]))
    return {
        "within20m": sum(value <= 20 for value in nearest.values()),
        "within40m": sum(value <= 40 for value in nearest.values()),
        "within80m": sum(value <= 80 for value in nearest.values()),
        "uncovered": sum(not math.isfinite(value) for value in nearest.values()),
    }


def _current_records(normalized_input: dict[str, Any], coverage_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = normalized_input.get("images") or {}
    if not isinstance(current, dict):
        return []
    by_id = {str(record.get("id")): record for record in coverage_records}
    result = []
    for key, value in current.items():
        source_id = str(key).split(":", 1)[-1]
        source = by_id.get(source_id)
        if source is None:
            continue
        record = dict(source)
        record["sourceID"] = source_id
        result.append(record)
    return result


def select_facade_captures(
    coverage: dict[str, Any] | list[dict[str, Any]],
    normalized_input: dict[str, Any],
    *,
    limit: int = 160,
    recent_days: int = RECENT_DAYS,
    max_distance_m: float = MAX_DISTANCE_M,
    min_front_facing: float = MIN_FRONT_FACING,
    owner_edge_tolerance: float = OWNER_EDGE_TOLERANCE_M,
    frame_length_m: float = DEFAULT_FRAME_LENGTH_M,
) -> dict[str, Any]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    frames, footprints, owner_keys = build_facade_frames(normalized_input.get("buildings", []), frame_length_m=frame_length_m)
    eligible, admission = _eligible_records(coverage, normalized_input, recent_days=recent_days, require_panoramas=True)
    candidates = _candidate_pairs(eligible, frames, footprints, max_distance_m=max_distance_m, min_front_facing=min_front_facing, owner_edge_tolerance=owner_edge_tolerance)
    selected, sequence_sets = _greedy_select(candidates, limit)
    # Compare like with like: the incumbent plan is evaluated only against
    # the same recent-panorama admission used by the proposed plan.
    current = _current_records(normalized_input, eligible)
    current_candidates = _candidate_pairs(current, frames, footprints, max_distance_m=max_distance_m, min_front_facing=min_front_facing, owner_edge_tolerance=owner_edge_tolerance)
    current_selected = [{"sourceID": key, "pairs": value["pairs"]} for key, value in current_candidates.items()]
    frame_ids = {frame["frameID"] for frame in frames}
    selection = []
    for item in selected:
        record = item["record"]
        selection.append(
            {
                "sourceID": item["sourceID"],
                "sequenceID": record.get("sequence_id", record.get("sequenceID")),
                "capturedAt": item["record"].get("captureDate", _capture_date(record).isoformat() if _capture_date(record) else None),
                "isPano": bool(record.get("is_pano", record.get("isPano", False))),
                "qualityScore": float(record.get("quality_score", record.get("qualityScore", 0.0)) or 0.0),
                "projectedLocationM": [float(value) for value in record["point"]],
                "coveredFrameIDs": [pair["frameID"] for pair in item["pairs"]],
                "score": float(item["score"]),
            }
        )
    covered = {frame_id for item in selected for frame_id in (pair["frameID"] for pair in item["pairs"])}
    frame_sequence_counts = {frame_id: len(sequence_sets.get(frame_id, set())) for frame_id in frame_ids}
    owner_frame_counts = {key: 0 for key in owner_keys}
    for frame in frames:
        if frame["frameID"] in covered:
            owner_frame_counts[frame["buildingKey"]] = owner_frame_counts.get(frame["buildingKey"], 0) + 1
    return {
        "schema": "facade-capture-plan/v1",
        "snapshotDate": normalized_input.get("snapshotDate"),
        "selectedSourceIDs": [item["sourceID"] for item in selection],
        "selection": selection,
        "provenance": {
            "selectionMethod": "contiguous facade frames + cKDTree distance + STRtree footprint line-of-sight + greedy marginal area",
            "projectedCRS": normalized_input["coordinateFrame"]["projectedCrs"],
            "originProjectedM": normalized_input["coordinateFrame"]["originProjectedM"],
            "recentDays": recent_days,
            "maxDistanceM": max_distance_m,
            "minFrontFacing": min_front_facing,
            "ownerEdgeToleranceM": owner_edge_tolerance,
            "frameLengthM": frame_length_m,
            "independentSequenceReward": {"first": 1.0, "second": 0.65, "later": 0.08},
            "opaqueJoinField": "sourceID",
        },
        "planReport": {
            "currentVsPlannedDistanceCoverageCounts": {
                "current": _distance_coverage(current_selected, frame_ids),
                "planned": _distance_coverage(selected, frame_ids),
            },
            "candidateCounts": {
                **admission,
                "currentCatalogImages": len(normalized_input.get("images") or {}),
                "currentEligibleImages": len(current),
                "visibleSources": len(candidates),
                "visiblePairs": sum(len(c["pairs"]) for c in candidates.values()),
            },
            "frameCounts": {
                "total": len(frames),
                "covered": len(covered),
                "coveredTwiceIndependentSequence": sum(count >= 2 for count in frame_sequence_counts.values()),
                "coveredByOwner": owner_frame_counts,
            },
            "selectedCount": len(selection),
            "independentSequenceCounts": frame_sequence_counts,
        },
    }


def run_paths(coverage_path: str | Path, normalized_input_path: str | Path, output_path: str | Path, *, limit: int = 160) -> dict[str, Any]:
    import json

    coverage = json.loads(Path(coverage_path).read_text())
    normalized = json.loads(Path(normalized_input_path).read_text())
    plan = select_facade_captures(coverage, normalized, limit=limit)
    plan["provenance"]["coverageInput"] = str(coverage_path)
    plan["provenance"]["normalizedInput"] = str(normalized_input_path)
    Path(output_path).write_text(json.dumps(plan, indent=2) + "\n")
    return plan
