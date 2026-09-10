"""Bounded stereo-ray hypotheses for paired facade-part proposals.

The output compares ray geometry and an assumed facade plane.  It is a
diagnostic of candidate correspondences; it does not certify registration,
camera pose, openings, or world geometry.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from general_world.stereo_rays import triangulate_pixel_pair


AMBIGUITY_COST_MARGIN = 0.05
OBSTRUCTION_ALPHA_MIN = 0.75
PAIR_LIMIT = 12
MAX_RAY_RESIDUAL_M = 1.0


def _known(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "none", "unknown", "null"}


def _box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        result = np.asarray(list(value), dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (4,) or not np.isfinite(result).all():
        return None
    x0, y0, x1, y1 = map(float, result)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _path(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if record.get(key):
            return str(Path(record[key]).resolve())
    return None


def _size(record: dict[str, Any]) -> tuple[int, int] | None:
    value = record.get("imageSize")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            w, h = int(value[0]), int(value[1])
            if w > 0 and h > 0:
                return w, h
        except (TypeError, ValueError):
            pass
    for wk, hk in (("imageWidth", "imageHeight"), ("widthPx", "heightPx")):
        try:
            w, h = int(record[wk]), int(record[hk])
            if w > 0 and h > 0:
                return w, h
        except (KeyError, TypeError, ValueError):
            pass
    source = _path(record, "path", "cropPath")
    if source:
        try:
            with Image.open(source) as image:
                return image.size
        except (OSError, ValueError):
            pass
    return None


def _alpha_fraction(path: str, box: tuple[float, float, float, float], expected_size: tuple[int, int] | None) -> float | None:
    try:
        with Image.open(path) as image:
            if expected_size is not None and image.size != expected_size:
                return None
            alpha = np.asarray(image.convert("RGBA"))[..., 3] > 0
    except (OSError, ValueError):
        return None
    h, w = alpha.shape
    x0, y0, x1, y1 = box
    left, top = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    right, bottom = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    if right <= left or bottom <= top:
        return None
    return float(np.mean(alpha[top:bottom, left:right]))


def _obstruction_index(report: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if report is None:
        return {}
    return {(_path(item, "cropPath", "path"), str(item.get("sourceId"))): item for item in report.get("crops", []) if _path(item, "cropPath", "path") and item.get("sourceId") is not None}


def _prepare(provenance: list[dict[str, Any]], proposals: list[dict[str, Any]], obstruction_report: dict[str, Any] | None) -> tuple[dict[Any, list[dict[str, Any]]], dict[str, int]]:
    by_index = {item.get("index"): item for item in provenance}
    obstruction = _obstruction_index(obstruction_report)
    enabled = obstruction_report is not None
    output: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    stats = {"invalidBoxes": 0, "blocked": 0, "missingOrInvalidObstructionMask": 0, "notRejectedByObstructionMask": 0, "sourcePathMismatches": 0}
    for item in proposals:
        index, crop = item.get("index"), by_index.get(item.get("index"))
        if crop is None or item.get("sourceId") is None or str(item.get("sourceId")) != str(crop.get("sourceId")) or _path(item, "cropPath", "path") != _path(crop, "path", "cropPath"):
            stats["sourcePathMismatches"] += 1
            continue
        size = _size(crop)
        source_key = (_path(crop, "path", "cropPath") or "", str(crop.get("sourceId")))
        mask = obstruction.get(source_key) if enabled else None
        for raw in item.get("proposals", []):
            box = _box(raw.get("box")) if isinstance(raw, dict) else None
            if box is None or size is None:
                stats["invalidBoxes"] += 1
                continue
            candidate = {"box": box, "category": str(raw.get("category", "unknown")), "score": raw.get("score"), "index": index, "sourceId": str(crop.get("sourceId"))}
            if enabled:
                mask_path = _path(mask, "outputMasked") if mask else None
                fraction = _alpha_fraction(mask_path, box, size) if mask_path else None
                if fraction is None:
                    stats["missingOrInvalidObstructionMask"] += 1
                    continue
                if fraction < OBSTRUCTION_ALPHA_MIN:
                    stats["blocked"] += 1
                    continue
                stats["notRejectedByObstructionMask"] += 1
            output[index].append(candidate)
    return output, stats


def _pair_candidates(crops: list[dict[str, Any]], limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for crop in crops:
        if _known(crop.get("clusterId")) and _known(crop.get("imageSha256")) and _known(crop.get("sourceId")):
            grouped[str(crop["clusterId"])].append(crop)
    candidates = []
    for cluster, records in sorted(grouped.items()):
        for left, right in itertools.combinations(sorted(records, key=lambda r: r.get("index", 0)), 2):
            if left.get("imageSha256") == right.get("imageSha256"):
                continue
            try:
                a, b = np.asarray(left["cameraCenterWorldM"], dtype=float), np.asarray(right["cameraCenterWorldM"], dtype=float)
                baseline = float(np.linalg.norm(a - b))
                if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append((-baseline, cluster, left.get("index", 0), right.get("index", 0), left, right))
    candidates.sort(key=lambda item: item[:4])
    selected = []
    seen_clusters: set[str] = set()
    for item in candidates:
        if item[1] not in seen_clusters:
            selected.append((item[4], item[5])); seen_clusters.add(item[1])
    for item in candidates:
        pair = (item[4], item[5])
        if pair not in selected:
            selected.append(pair)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def compare(provenance: dict[str, Any], proposal_report: dict[str, Any], obstruction_report: dict[str, Any] | None = None, *, pair_limit: int = PAIR_LIMIT) -> dict[str, Any]:
    crops = list(provenance.get("crops", []))
    proposal_items = list(proposal_report.get("crops", []))
    prepared, filtering = _prepare(crops, proposal_items, obstruction_report)
    pairs = _pair_candidates(crops, pair_limit)
    hypotheses: list[dict[str, Any]] = []
    invalid_reasons: dict[str, int] = defaultdict(int)
    assigned_count = ambiguous_count = unmatched_count = 0
    planar_errors: list[float] = []
    offsets: list[float] = []
    pair_reports = []
    for left, right in pairs:
        left_size, right_size = _size(left), _size(right)
        pair_hypotheses: list[dict[str, Any]] = []
        assigned_for_pair = 0
        if left_size is None or right_size is None:
            pair_reports.append({"leftViewId": left.get("sourceId"), "rightViewId": right.get("sourceId"), "status": "invalid_image_dimensions"})
            continue
        left_props, right_props = prepared.get(left.get("index"), []), prepared.get(right.get("index"), [])
        for category in sorted({p["category"] for p in left_props} | {p["category"] for p in right_props}):
            lhs, rhs = [p for p in left_props if p["category"] == category], [p for p in right_props if p["category"] == category]
            if not lhs or not rhs:
                unmatched_count += len(lhs) + len(rhs)
                continue
            details: dict[tuple[int, int], dict[str, Any]] = {}
            matrix = np.full((len(lhs), len(rhs)), 1e9, dtype=float)
            for i, lp in enumerate(lhs):
                for j, rp in enumerate(rhs):
                    result = triangulate_pixel_pair(left, _center(lp["box"]), left_size, right, _center(rp["box"]), right_size, min_angle_deg=5.0, max_condition=100.0, max_residual_m=MAX_RAY_RESIDUAL_M)
                    if not result.get("valid"):
                        invalid_reasons[str(result.get("reason", "invalid"))] += 1
                        continue
                    details[(i, j)] = {**result, "category": category, "leftIndex": left.get("index"), "rightIndex": right.get("index"), "leftViewId": left.get("sourceId"), "rightViewId": right.get("sourceId"), "cost": float(result["rayResidualM"] + 0.1 * abs(result["signedOffsetFromAssumedPlaneM"]) + 0.01 * result["planarReprojectionErrorM"])}
                    matrix[i, j] = details[(i, j)]["cost"]
            if not details:
                unmatched_count += len(lhs) + len(rhs)
                continue
            rows, cols = linear_sum_assignment(matrix)
            assigned = {(int(i), int(j)) for i, j in zip(rows, cols) if (int(i), int(j)) in details}
            assigned_for_pair += len(assigned)
            unmatched_count += len(lhs) + len(rhs) - 2 * len(assigned)
            for (i, j), detail in details.items():
                row_costs = sorted(matrix[i, k] for k in range(matrix.shape[1]) if (i, k) in details)
                col_costs = sorted(matrix[k, j] for k in range(matrix.shape[0]) if (k, j) in details)
                row_unique = matrix[i, j] == row_costs[0] and (len(row_costs) == 1 or row_costs[1] - row_costs[0] > AMBIGUITY_COST_MARGIN)
                col_unique = matrix[i, j] == col_costs[0] and (len(col_costs) == 1 or col_costs[1] - col_costs[0] > AMBIGUITY_COST_MARGIN)
                ambiguous = not (row_unique and col_unique)
                is_assigned = (i, j) in assigned
                if not is_assigned and not ambiguous:
                    continue
                entry = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in detail.items() if key not in {"valid", "pointA", "pointB", "depthA", "depthB", "condition", "angleDeg", "panelLocalXYM", "center"}}
                entry.update({"triangulatedCenterWorldM": detail["center"].tolist(), "assigned": is_assigned, "ambiguous": ambiguous})
                hypotheses.append(entry)
                if is_assigned:
                    assigned_count += 1
                    if ambiguous: ambiguous_count += 1
                    planar_errors.append(detail["planarReprojectionErrorM"]); offsets.append(detail["signedOffsetFromAssumedPlaneM"])
            pair_hypotheses.extend([h for h in hypotheses if h.get("leftIndex") in {left.get("index")} and h.get("rightIndex") in {right.get("index")}])
        pair_reports.append({"leftViewId": left.get("sourceId"), "rightViewId": right.get("sourceId"), "clusterId": left.get("clusterId"), "assigned": assigned_for_pair, "hypotheses": len(pair_hypotheses)})
    offset_count = sum(abs(value) > 0.5 for value in offsets)
    discrepancy_count = sum(value > 0.5 for value in planar_errors)
    interpretation = "insufficient assigned hypotheses"
    if offsets:
        interpretation = f"{offset_count}/{len(offsets)} assigned hypotheses have >0.5m depth offset and {discrepancy_count}/{len(planar_errors)} have >0.5m planar discrepancy; compatible with depth effects for those hypotheses, without validating correspondence."
    return {"schema": "facade-part-stereo-ray-diagnostic/v1", "evidenceScope": "bounded stereo correspondence hypotheses only; no camera/world registration, opening acceptance, or geometry admission", "pairing": {"requestedLimit": pair_limit, "pairsCompared": len(pairs), "sameKnownClusterDifferentHash": True}, "counts": {"assigned": assigned_count, "ambiguousAssigned": ambiguous_count, "unmatched": unmatched_count, "ambiguousHypotheses": sum(h.get("ambiguous", False) for h in hypotheses)}, "proposalFiltering": filtering, "invalidCandidateReasons": dict(invalid_reasons), "offsetInterpretation": interpretation, "planarReprojectionErrorsM": planar_errors, "triangulationRayResidualsM": [h["rayResidualM"] for h in hypotheses if h.get("assigned")], "pairs": pair_reports, "hypotheses": hypotheses}


def run_paths(provenance_path: Path, proposals_path: Path, output_path: Path, obstruction_path: Path | None = None, *, pair_limit: int = PAIR_LIMIT) -> dict[str, Any]:
    provenance, proposals = json.loads(provenance_path.read_text()), json.loads(proposals_path.read_text())
    obstruction = json.loads(obstruction_path.read_text()) if obstruction_path else None
    report = compare(provenance, proposals, obstruction, pair_limit=pair_limit)
    report["inputs"] = {"provenance": str(provenance_path.resolve()), "proposals": str(proposals_path.resolve()), "obstruction": str(obstruction_path.resolve()) if obstruction_path else None}
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obstruction-report", type=Path)
    parser.add_argument("--pair-limit", type=int, default=PAIR_LIMIT)
    args = parser.parse_args()
    print(json.dumps(run_paths(args.provenance, args.proposals, args.output, args.obstruction_report, pair_limit=args.pair_limit), sort_keys=True))


if __name__ == "__main__":
    main()
