"""Compare visual facade-part proposals across independent rectified views.

This is a bounded consistency diagnostic.  It uses only exact opaque owner/frame
joins from crop provenance and never emits accepted openings, registration
corrections, or derived facade geometry.  Pixel boxes are compared after the
fixed crop-to-metric affine conversion so a disagreement cannot be hidden by
an alignment adjustment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


IOU_MIN = 0.30
CENTER_ERROR_MAX_M = 1.0
SIZE_ERROR_MAX_M = 1.0
AMBIGUITY_COST_MARGIN = 0.05
OBSTRUCTION_ALPHA_MIN = 0.75
EXAMPLE_LIMIT = 24


def _finite_box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        a = np.asarray(list(value), dtype=float)
    except (TypeError, ValueError):
        return None
    if a.shape != (4,) or not np.isfinite(a).all():
        return None
    x0, y0, x1, y1 = map(float, a)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _crop_records(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    records = provenance.get("crops") if isinstance(provenance, dict) else provenance
    if not isinstance(records, list):
        raise ValueError("provenance must contain a crops list")
    return [record for record in records if isinstance(record, dict)]


def _frame_key(record: dict[str, Any]) -> tuple[Any, Any, str] | None:
    owner = record.get("buildingKey")
    frame_index = record.get("frameIndex", record.get("frontageFrameIndex"))
    frame = record.get("metricFrame")
    if owner is None or frame_index is None or not isinstance(frame, dict):
        return None
    # JSON text gives exact frame identity while avoiding tolerance/alignment.
    return owner, frame_index, json.dumps(frame, sort_keys=True, separators=(",", ":"))


def _image_size(record: dict[str, Any]) -> tuple[int, int] | None:
    size = record.get("imageSize")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            w, h = int(size[0]), int(size[1])
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
    for key in ("path", "cropPath"):
        path = record.get(key)
        if path:
            try:
                with Image.open(path) as image:
                    return image.size
            except (OSError, ValueError):
                pass
    return None


def _metric_box(box: tuple[float, float, float, float], record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    dimensions = _image_size(record)
    frame = record.get("metricFrame")
    try:
        width_m = float(record["widthM"])
        height_m = float(record["diagnosticHeightM"])
    except (KeyError, TypeError, ValueError):
        return None
    if dimensions is None or not np.isfinite([width_m, height_m]).all() or width_m <= 0 or height_m <= 0:
        return None
    try:
        frame_valid = isinstance(frame, dict) and all(np.asarray(frame[k], dtype=float).shape == (3,) and np.isfinite(np.asarray(frame[k], dtype=float)).all() for k in ("origin", "horizontal", "vertical"))
    except (KeyError, TypeError, ValueError):
        frame_valid = False
    if not frame_valid:
        return None
    image_w, image_h = dimensions
    x0, y0, x1, y1 = box
    # Image y grows downward; metric facade y grows upward from the crop base.
    return (x0 / image_w * width_m, (image_h - y1) / image_h * height_m,
            x1 / image_w * width_m, (image_h - y0) / image_h * height_m)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def _residual(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float, float]:
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    center = math.hypot(ac[0] - bc[0], ac[1] - bc[1])
    width = abs((a[2] - a[0]) - (b[2] - b[0]))
    height = abs((a[3] - a[1]) - (b[3] - b[1]))
    size = max(width, height)
    # Cost is only an assignment aid; thresholds below are the diagnostic rule.
    return _iou(a, b), center, width, height, size


def _proposal_records(proposals: Any) -> list[dict[str, Any]]:
    if isinstance(proposals, dict):
        proposals = proposals.get("crops", proposals.get("records", proposals.get("items", [])))
    if not isinstance(proposals, list):
        raise ValueError("proposal report must contain crops/records")
    return [x for x in proposals if isinstance(x, dict)]


def _proposal_path(record: dict[str, Any]) -> str | None:
    value = record.get("cropPath", record.get("path"))
    return str(Path(value).resolve()) if value else None


def _obstruction_index(report: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if report is None:
        return {}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _crop_records(report):
        path = _proposal_path(item)
        source = item.get("sourceId")
        if path and source is not None:
            result[(path, str(source))] = item
    return result


def _alpha_fraction(path: str, box: tuple[float, float, float, float], expected_size: tuple[int, int] | None = None) -> float | None:
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


def _prepare_proposals(
    proposal_items: list[dict[str, Any]],
    provenance_by_index: dict[Any, dict[str, Any]],
    obstruction: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[Any, list[dict[str, Any]]], dict[str, int]]:
    by_crop: dict[Any, list[dict[str, Any]]] = {}
    counts = {"invalidBoxes": 0, "blocked": 0, "unknownObstruction": 0, "unobstructed": 0}
    for item in proposal_items:
        index = item.get("index")
        provenance = provenance_by_index.get(index)
        if provenance is None:
            continue
        item_source = item.get("sourceId")
        source_id = str(item_source) if item_source is not None else ""
        provenance_source = provenance.get("sourceId")
        path = _proposal_path(item)
        provenance_path = _proposal_path(provenance)
        # Indexes are only a lookup aid.  The opaque source and resolved crop
        # path must both agree before a proposal can enter a comparison.
        if item_source is None or provenance_source is None or source_id != str(provenance_source):
            continue
        if path is None or provenance_path is None or path != provenance_path:
            continue
        for raw in item.get("proposals", []):
            if not isinstance(raw, dict):
                counts["invalidBoxes"] += 1
                continue
            box = _finite_box(raw.get("box"))
            if box is None:
                counts["invalidBoxes"] += 1
                continue
            category = str(raw.get("category", "unknown"))
            candidate = {"box": box, "category": category, "score": raw.get("score"), "sourceId": source_id}
            if obstruction:
                mask = obstruction.get((path or "", source_id))
                mask_path = mask.get("outputMasked") if mask else None
                fraction = _alpha_fraction(mask_path, box, _image_size(provenance)) if mask_path else None
                candidate["nonObstructedFraction"] = fraction
                if fraction is None:
                    counts["unknownObstruction"] += 1
                    continue
                if fraction < OBSTRUCTION_ALPHA_MIN:
                    counts["blocked"] += 1
                    continue
                counts["unobstructed"] += 1
            by_crop.setdefault(index, []).append(candidate)
    return by_crop, counts


def compare(provenance: dict[str, Any], proposals: Any, obstruction_report: dict[str, Any] | None = None, *, example_limit: int = EXAMPLE_LIMIT) -> dict[str, Any]:
    """Return conservative cross-view proposal agreement diagnostics."""
    crops = _crop_records(provenance)
    by_index = {record.get("index"): record for record in crops}
    proposal_items = _proposal_records(proposals)
    obstruction = _obstruction_index(obstruction_report)
    proposal_by_crop, proposal_counts = _prepare_proposals(proposal_items, by_index, obstruction)
    groups: dict[tuple[Any, Any, str], list[dict[str, Any]]] = {}
    for record in crops:
        key = _frame_key(record)
        if key is not None:
            groups.setdefault(key, []).append(record)

    totals = {"agreement": 0, "disagreement": 0, "unmatched": 0, "ambiguous": 0}
    examples: list[dict[str, Any]] = []
    rejected_same_view_group = 0
    residuals: list[dict[str, float]] = []
    for key, records in groups.items():
        for left, right in itertools.combinations(records, 2):
            if left.get("viewGroup") is None or left.get("viewGroup") == right.get("viewGroup"):
                rejected_same_view_group += 1
                continue
            left_props = proposal_by_crop.get(left.get("index"), [])
            right_props = proposal_by_crop.get(right.get("index"), [])
            categories = sorted({p["category"] for p in left_props} | {p["category"] for p in right_props})
            for category in categories:
                lhs = [p for p in left_props if p["category"] == category]
                rhs = [p for p in right_props if p["category"] == category]
                if not lhs or not rhs:
                    totals["unmatched"] += len(lhs) + len(rhs)
                    continue
                left_metric = [_metric_box(p["box"], by_index[left["index"]]) for p in lhs]
                right_metric = [_metric_box(q["box"], by_index[right["index"]]) for q in rhs]
                valid_left = [i for i, box in enumerate(left_metric) if box is not None]
                valid_right = [i for i, box in enumerate(right_metric) if box is not None]
                totals["unmatched"] += (len(lhs) - len(valid_left)) + (len(rhs) - len(valid_right))
                if not valid_left or not valid_right:
                    continue
                matrix = np.array([[1.0 - _residual(left_metric[i], right_metric[j])[0] for j in valid_right] for i in valid_left])
                rows, cols = linear_sum_assignment(matrix)
                totals["unmatched"] += (len(valid_left) - len(rows)) + (len(valid_right) - len(cols))
                for ri, ci in zip(rows, cols):
                    p, q = lhs[valid_left[int(ri)]], rhs[valid_right[int(ci)]]
                    a, b = left_metric[valid_left[int(ri)]], right_metric[valid_right[int(ci)]]
                    iou, center, width, height, size = _residual(a, b)
                    costs = sorted(1.0 - _residual(a, right_metric[j])[0] for j in valid_right)
                    ambiguous = len(costs) > 1 and costs[1] - costs[0] <= AMBIGUITY_COST_MARGIN
                    agreement = (not ambiguous and iou >= IOU_MIN and center <= CENTER_ERROR_MAX_M and size <= SIZE_ERROR_MAX_M)
                    if ambiguous:
                        totals["ambiguous"] += 1
                    if agreement:
                        totals["agreement"] += 1
                    else:
                        totals["disagreement"] += 1
                    residuals.append({"iou": iou, "centerErrorM": center, "widthErrorM": width, "heightErrorM": height, "sizeErrorM": size})
                    if len(examples) < example_limit and (not agreement or ambiguous):
                        examples.append({"buildingKey": key[0], "frameIndex": key[1], "leftIndex": left.get("index"), "rightIndex": right.get("index"), "category": category, "agreement": agreement, "ambiguous": ambiguous, "iou": iou, "centerErrorM": center, "widthErrorM": width, "heightErrorM": height, "scoreLeft": p.get("score"), "scoreRight": q.get("score")})
    def stats(field: str) -> dict[str, float] | None:
        values = [x[field] for x in residuals]
        return {"count": len(values), "mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))} if values else None
    return {
        "schema": "facade-part-multiview-consistency/v1",
        "evidenceScope": "visual proposal consistency only; no registration certification, opening acceptance, or derived geometry",
        "thresholds": {"iouMinimum": IOU_MIN, "centerErrorMaxM": CENTER_ERROR_MAX_M, "sizeErrorMaxM": SIZE_ERROR_MAX_M, "ambiguityCostMargin": AMBIGUITY_COST_MARGIN, "obstructionAlphaMinimum": OBSTRUCTION_ALPHA_MIN},
        "pairing": {"groups": len(groups), "eligibleViewPairs": sum(1 for records in groups.values() for a, b in itertools.combinations(records, 2) if a.get("viewGroup") is not None and a.get("viewGroup") != b.get("viewGroup")), "sameViewGroupRejected": rejected_same_view_group},
        "counts": totals,
        "residuals": {"centerErrorM": stats("centerErrorM"), "sizeErrorM": stats("sizeErrorM"), "iou": stats("iou")},
        "proposalFiltering": proposal_counts,
        "examples": examples[:example_limit],
    }


def run_paths(provenance_path: Path, proposals_path: Path, output_path: Path, obstruction_path: Path | None = None, *, example_limit: int = EXAMPLE_LIMIT) -> dict[str, Any]:
    provenance = json.loads(provenance_path.read_text())
    proposals = json.loads(proposals_path.read_text())
    obstruction = json.loads(obstruction_path.read_text()) if obstruction_path else None
    report = compare(provenance, proposals, obstruction, example_limit=example_limit)
    report["inputs"] = {"provenance": str(provenance_path.resolve()), "proposals": str(proposals_path.resolve()), "obstruction": str(obstruction_path.resolve()) if obstruction_path else None}
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obstruction-report", type=Path)
    parser.add_argument("--example-limit", type=int, default=EXAMPLE_LIMIT)
    args = parser.parse_args()
    print(json.dumps(run_paths(args.provenance, args.proposals, args.output, args.obstruction_report, example_limit=args.example_limit), sort_keys=True))


if __name__ == "__main__":
    main()
