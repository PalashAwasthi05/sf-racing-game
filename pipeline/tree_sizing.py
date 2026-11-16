"""Source identity, explicit DBH, and LiDAR checks for street tree sizing."""
from __future__ import annotations

import math
import re

import numpy as np


def _norm(value):
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def inventory_identity(record):
    """Identity requires an explicit tree ordinal plus matching source fields."""
    label = _norm(record.get("description"))
    match = re.search(r"(?:^|\|)\s*tree\s+(\d+)\s*$", label)
    if not match:
        return None
    address = _norm(label[:match.start()].rstrip(" |"))
    species = _norm(record.get("qspecies") or record.get("species"))
    site = _norm(record.get("qsiteinfo") or record.get("siteinfo"))
    if not address or not species or not site:
        return None
    return address, match.group(1), species, site


def _position(record):
    p = record.get("position") or record.get("sourcePosition")
    if p is None or len(p) < 2:
        return None
    return np.asarray(p[:2], dtype=float)


def deduplicate_inventory_records(records, max_distance_m=1.5):
    """Merge only matching source identities whose repeated rows are nearby."""
    keyed = {}
    for index, record in enumerate(records):
        key, pos = inventory_identity(record), _position(record)
        bucket = (key, index) if key is None or pos is None else key
        keyed.setdefault(bucket, []).append((index, record, pos))
    groups = []
    for bucket, members in keyed.items():
        if len(bucket) == 2 and isinstance(bucket[1], int):
            groups.append(members)
            continue
        pending = list(members)
        while pending:
            cluster = [pending.pop(0)]
            remaining = []
            for member in pending:
                if all(np.linalg.norm(member[2] - chosen[2]) <= max_distance_m for chosen in cluster):
                    cluster.append(member)
                else:
                    remaining.append(member)
            groups.append(cluster)
            pending = remaining
    outputs, audit = [], []
    for group in groups:
        source_records = [r for _, r, _ in group]
        positions = np.asarray([p for _, _, p in group if p is not None], dtype=float)
        row = dict(source_records[0])
        if len(positions): row["position"] = np.median(positions, axis=0).tolist()
        radius_values = [float(r["radiusM"]) for r in source_records if _positive(r.get("radiusM"))]
        if radius_values: row["radiusM"] = float(np.median(radius_values))
        raw_dbh = [v for v in (_number(r.get("dbh")) for r in source_records) if v is not None]
        if raw_dbh: row["dbh"] = float(np.median(raw_dbh))
        row["sourceRecordCount"] = len(source_records)
        row["sourceRecordRefs"] = [str(r.get("sourceId", r.get("treeid", i))) for i, r, _ in group]
        outputs.append(row)
        audit.append({"identity": list(inventory_identity(source_records[0]) or ()),
                      "sourceRecordRefs": row["sourceRecordRefs"],
                      "positionsM": [p.tolist() for p in positions],
                      "maxPairDistanceM": _max_pair_distance(positions),
                      "decision": "merged-same-inventory-identity" if len(group) > 1 else "kept"})
    return outputs, audit


def _max_pair_distance(points):
    if len(points) < 2: return 0.0
    return max(float(np.linalg.norm(a - b)) for i, a in enumerate(points) for b in points[i + 1:])


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _positive(value):
    number = _number(value)
    return number is not None and number > 0


def trunk_radius(record, *, dbh_unit=None):
    """Use measured DBH only when the source supplies an explicit unit."""
    raw_dbh = _number(record.get("dbh"))
    if raw_dbh is None or dbh_unit not in ("in", "cm", "m"):
        return {"radiusM": None, "source": "unresolved", "dbhRaw": raw_dbh,
                "dbhUnitMetadata": "missing-explicit-unit", "activePipelineAssumption": "no DBH override applied"}
    factor = {"in": .0254, "cm": .01, "m": 1.0}[dbh_unit]
    return {"radiusM": raw_dbh * factor / 2., "source": "explicit-unit-conversion",
            "dbhRaw": raw_dbh, "dbhUnitMetadata": f"caller-declared-{dbh_unit}",
            "activePipelineAssumption": "no implicit unit conversion"}


def fallback_height_m(seed_value):
    try: seed = int(float(seed_value))
    except (TypeError, ValueError): return None
    return 5.0 + float(np.random.default_rng(seed).random()) * 3.0


def assess_lidar_height(samples, center_xy, wall_direction_xy=None, wall_distance_m=None,
                        *, core_radius_m=3.0, sector_count=8, min_high_returns=20,
                        high_threshold_m=12.0, min_wall_fraction=.8,
                        max_opposite_fraction=.1, wall_reach_m=4.0,
                        fallback_seed=None):
    """Reject a tall return tail concentrated on a nearby wall-facing side."""
    p = np.asarray(samples, dtype=float).reshape(-1, 3)
    c = np.asarray(center_xy, dtype=float).reshape(2)
    radius = np.linalg.norm(p[:, :2] - c, axis=1)
    core = p[(radius <= core_radius_m) & np.isfinite(p[:, 2])]
    if len(core) < 5:
        return {"status": "unresolved-too-few-core-returns", "heightM": None, "coreReturnCount": int(len(core))}
    raw_height = float(np.percentile(core[:, 2], 95))
    result = {"status": "accepted", "heightM": raw_height, "coreReturnCount": int(len(core)),
              "rawP95HeightM": raw_height, "upperTailCount": 0,
              "wallFacingFraction": None, "oppositeWallFraction": None}
    if wall_direction_xy is None or wall_distance_m is None or float(wall_distance_m) > wall_reach_m:
        return result
    wall = np.asarray(wall_direction_xy, dtype=float).reshape(2)
    norm = float(np.linalg.norm(wall))
    if norm < 1e-8: return result
    wall /= norm
    tail = core[core[:, 2] >= max(float(high_threshold_m), .5 * raw_height)]
    result["upperTailCount"] = int(len(tail))
    if len(tail) < min_high_returns: return result
    radial = tail[:, :2] - c
    lengths = np.linalg.norm(radial, axis=1)
    radial = radial[lengths > 1e-8] / lengths[lengths > 1e-8, None]
    dot = radial @ wall
    facing, opposite = float(np.mean(dot >= 0)), float(np.mean(dot <= -.5))
    result.update(wallFacingFraction=facing, oppositeWallFraction=opposite)
    if facing >= min_wall_fraction and opposite <= max_opposite_fraction:
        result.update(status="rejected-wall-facing-upper-tail", heightM=None,
                      fallbackAction="use-existing-species-fallback")
        if fallback_seed is not None: result["fallbackHeightM"] = fallback_height_m(fallback_seed)
    return result
