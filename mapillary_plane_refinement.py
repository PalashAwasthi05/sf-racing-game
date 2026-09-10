"""Optional point-to-plane refinement diagnostic for an external alignment report.

This is evidence only.  It does not alter imagery metadata or claim facade,
camera, or world-registration accuracy.  It starts from a fitted transform in
``mapillary_world_alignment`` and checks whether local LiDAR planes support a
small bounded refinement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mapillary_world_alignment import (
    MAX_ROTATION_DEG,
    MAX_TRANSLATION_M,
    _crop_source_to_target_xy,
    _metrics,
    _sample_points,
    _transform,
    enu_to_local_projected,
    load_world_points,
)


NEIGHBOURS = 12
MAX_NEIGHBOUR_RADIUS_M = 1.5
PLANARITY_RATIO = 0.08
MIN_TANGENT_SPREAD_M2 = 0.02
MAX_PLANE_UPDATE_M = 5.0
MAX_PLANE_UPDATE_DEG = 10.0


def _residual_metrics(residuals: np.ndarray) -> dict[str, Any]:
    values = np.abs(np.asarray(residuals, dtype=float))
    if not len(values):
        return {"count": 0, "medianM": None, "p75M": None, "p90M": None}
    return {"count": int(len(values)), "medianM": float(np.median(values)), "p75M": float(np.percentile(values, 75)), "p90M": float(np.percentile(values, 90))}


def _spatial_holdout(points: np.ndarray) -> np.ndarray:
    cells = np.floor(points / 5.0).astype(np.int64)
    hashed = (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (cells[:, 2] * 83492791)
    return (hashed % 5) == 0


def _plane_correspondences(points: np.ndarray, target: np.ndarray, tree: cKDTree) -> dict[str, Any]:
    distances, indices = tree.query(points, k=NEIGHBOURS, distance_upper_bound=MAX_NEIGHBOUR_RADIUS_M)
    valid = np.isfinite(distances).all(axis=1) & (indices < len(target)).all(axis=1)
    if not np.any(valid):
        return {"points": np.empty((0, 3)), "targets": np.empty((0, 3)), "normals": np.empty((0, 3)), "groups": np.empty(0, dtype=int), "nearestDistances": np.empty(0), "normalDiversity": {}}
    src = points[valid]
    nbrs = target[indices[valid]]
    centers = nbrs.mean(axis=1)
    centered = nbrs - centers[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(NEIGHBOURS)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    planar = (eigvals[:, 0] / np.maximum(eigvals.sum(axis=1), 1e-12) <= PLANARITY_RATIO) & (eigvals[:, 1] >= MIN_TANGENT_SPREAD_M2)
    if not np.any(planar):
        return {"points": np.empty((0, 3)), "targets": np.empty((0, 3)), "normals": np.empty((0, 3)), "groups": np.empty(0, dtype=int), "nearestDistances": distances[valid, 0][planar], "normalDiversity": {"axisGroupCounts": {}}}
    src = src[planar]
    centers = centers[planar]
    normals = eigvecs[planar, :, 0]
    groups = np.argmax(np.abs(normals), axis=1)
    counts = {str(axis): int(np.count_nonzero(groups == axis)) for axis in range(3)}
    return {"points": src, "targets": centers, "normals": normals, "groups": groups, "nearestDistances": distances[valid, 0][planar], "normalDiversity": {"axisGroupCounts": counts, "axisGroupsPresent": int(np.count_nonzero(list(counts.values()))), "planarityMedian": float(np.median(eigvals[planar, 0] / np.maximum(eigvals[planar].sum(axis=1), 1e-12)))} }


def _balanced_rows(correspondences: dict[str, Any], maximum: int = 2400, seed: int = 13) -> dict[str, Any]:
    groups = correspondences["groups"]
    if not len(groups):
        return correspondences
    present = [axis for axis in range(3) if np.any(groups == axis)]
    if len(present) < 2:
        return {**correspondences, "degenerateNormalGroups": True}
    per_group = max(1, min(maximum // len(present), min(np.count_nonzero(groups == axis) for axis in present)))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(np.flatnonzero(groups == axis), size=per_group, replace=False) for axis in present])
    return {key: value[chosen] if isinstance(value, np.ndarray) and len(value) == len(groups) else value for key, value in correspondences.items()} | {"degenerateNormalGroups": False}


def _plane_fit(source: np.ndarray, target: np.ndarray, initial_r: np.ndarray, initial_t: np.ndarray) -> dict[str, Any]:
    holdout = _spatial_holdout(source)
    train, test = source[~holdout], source[holdout]
    if len(train) < 24 or len(test) < 6:
        return {"status": "rejected", "reason": "inadequate spatial train/holdout split"}
    tree = cKDTree(target)
    rotation, translation = initial_r.copy(), initial_t.copy()
    initial_train_nn, _ = _metrics(tree.query(_transform(train, rotation, translation), k=1)[0]), None
    initial_test_nn, _ = _metrics(tree.query(_transform(test, rotation, translation), k=1)[0]), None
    normal_report: dict[str, Any] = {}
    iterations = 0
    information_singular = None
    for iteration in range(8):
        current = _transform(train, rotation, translation)
        corr = _plane_correspondences(current, target, tree)
        corr = _balanced_rows(corr)
        normal_report = corr.get("normalDiversity", {})
        if corr.get("degenerateNormalGroups") or len(corr["points"]) < 24:
            break
        residual = np.sum(corr["normals"] * (corr["points"] - corr["targets"]), axis=1)
        median = float(np.median(np.abs(residual)))
        mad = float(np.median(np.abs(np.abs(residual) - median)))
        limit = min(1.0, max(0.05, median + 2.5 * 1.4826 * mad))
        keep = np.abs(residual) <= limit
        if np.count_nonzero(keep) < 24:
            break
        points = corr["points"][keep]
        normals = corr["normals"][keep]
        residual = residual[keep]
        jacobian = np.column_stack((np.cross(points, normals), normals))
        information = jacobian.T @ jacobian
        singular = np.linalg.svd(information, compute_uv=False)
        information_singular = singular.tolist()
        if singular[-1] < max(singular[0] * 1e-7, 1e-8):
            break
        step, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        delta_r = Rotation.from_rotvec(step[:3]).as_matrix()
        proposed_r = delta_r @ rotation
        proposed_t = delta_r @ translation + step[3:]
        delta_translation = np.linalg.norm(proposed_t - translation)
        delta_rotation = np.degrees(np.linalg.norm(step[:3]))
        if np.linalg.norm(proposed_t) > MAX_TRANSLATION_M or _rotation_angle(proposed_r) > MAX_ROTATION_DEG or delta_translation > MAX_PLANE_UPDATE_M or delta_rotation > MAX_PLANE_UPDATE_DEG:
            break
        rotation, translation = proposed_r, proposed_t
        iterations = iteration + 1
        if np.linalg.norm(step) < 1e-4:
            break

    final_train_points = _transform(train, rotation, translation)
    final_test_points = _transform(test, rotation, translation)
    final_train_nn = _metrics(tree.query(final_train_points, k=1)[0])
    final_test_nn = _metrics(tree.query(final_test_points, k=1)[0])
    initial_train_plane = _plane_correspondences(_transform(train, initial_r, initial_t), target, tree)
    initial_test_plane = _plane_correspondences(_transform(test, initial_r, initial_t), target, tree)
    final_train_plane = _plane_correspondences(final_train_points, target, tree)
    final_test_plane = _plane_correspondences(final_test_points, target, tree)
    report = {
        "status": "candidate-evidence" if information_singular is not None and normal_report.get("axisGroupsPresent", 0) >= 2 else "rejected",
        "evidenceOnly": True,
        "reason": None if information_singular is not None else "plane normal information is degenerate or unavailable",
        "iterations": iterations,
        "normalDiversity": normal_report,
        "informationSingularValues": information_singular,
        "initial": {"trainNearestNeighbour": initial_train_nn, "holdoutNearestNeighbour": initial_test_nn, "trainPlane": _residual_metrics(np.sum(initial_train_plane.get("normals", np.empty((0, 3))) * (initial_train_plane.get("points", np.empty((0, 3))) - initial_train_plane.get("targets", np.empty((0, 3)))), axis=1)), "holdoutPlane": _residual_metrics(np.sum(initial_test_plane.get("normals", np.empty((0, 3))) * (initial_test_plane.get("points", np.empty((0, 3))) - initial_test_plane.get("targets", np.empty((0, 3)))), axis=1))},
        "final": {"trainNearestNeighbour": final_train_nn, "holdoutNearestNeighbour": final_test_nn, "trainPlane": _residual_metrics(np.sum(final_train_plane.get("normals", np.empty((0, 3))) * (final_train_plane.get("points", np.empty((0, 3))) - final_train_plane.get("targets", np.empty((0, 3)))), axis=1)), "holdoutPlane": _residual_metrics(np.sum(final_test_plane.get("normals", np.empty((0, 3))) * (final_test_plane.get("points", np.empty((0, 3))) - final_test_plane.get("targets", np.empty((0, 3)))), axis=1))},
        "transform": {"rotationMatrix": rotation.tolist(), "translationM": translation.tolist(), "translationNormM": float(np.linalg.norm(translation)), "rotationDeg": _rotation_angle(rotation), "bounded": bool(np.linalg.norm(translation) <= MAX_TRANSLATION_M and _rotation_angle(rotation) <= MAX_ROTATION_DEG)},
    }
    for metric_name in ("holdoutNearestNeighbour", "holdoutPlane"):
        before = report["initial"][metric_name].get("p75M")
        after = report["final"][metric_name].get("p75M")
        report[f"{metric_name}Improvement"] = float((before - after) / before) if before is not None and after is not None and before > 1e-9 else None
    return report


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def refine(imagery: Path, normalized_path: Path, world_path: Path, alignment_report: Path, cluster_ids: list[str]) -> dict[str, Any]:
    from mapillary_world_alignment import _world_frame

    normalized = json.loads(normalized_path.read_text())
    world = json.loads(world_path.read_text())
    frame = _world_frame(normalized, world)
    dense = json.loads(alignment_report.read_text())
    dense_by_id = {str(item["clusterId"]): item for item in dense.get("clusters", [])}
    if not cluster_ids:
        cluster_ids = [cid for cid, item in dense_by_id.items() if item.get("registration", {}).get("accepted")][:3]
    target = load_world_points(normalized_path, normalized, world)
    target_tree = cKDTree(target)
    records = []
    for cluster_id in cluster_ids:
        item = dense_by_id.get(str(cluster_id), {})
        fitted = item.get("registration", {}).get("fittedWorldTransform")
        path = imagery / "clusters" / f"{cluster_id}.json"
        record: dict[str, Any] = {"clusterId": str(cluster_id), "evidenceOnly": True}
        if not fitted or not path.exists():
            record.update({"status": "rejected", "reason": "dense fitted transform or cluster file missing"})
            records.append(record)
            continue
        cluster = json.loads(path.read_text()); cluster = cluster[0] if isinstance(cluster, list) else cluster
        points = np.asarray([point["coordinates"] for point in cluster.get("points", {}).values()], dtype=float)
        local = enu_to_local_projected(points, cluster["reference_lla"], frame)
        sampled = _sample_points(local, 20_000, seed=19)
        cropped, crop = _crop_source_to_target_xy(sampled, target)
        rotation = np.asarray(fitted["rotationMatrix"], dtype=float)
        translation = np.asarray(fitted["translationM"], dtype=float)
        report = _plane_fit(cropped, target, rotation, translation)
        record.update({"selectedImageCount": item.get("selectedImageCount", 0), "sfmPointCount": int(len(points)), "sourceCropXY": crop, "planeRefinement": report})
        records.append(record)
    return {"schema": "imagery-world-plane-refinement-diagnostic/v1", "worldRegistrationValidated": False, "facadeMeasurementsGenerated": 0, "evidenceScope": "point-to-plane candidate evidence only; no camera or facade accuracy claim", "coordinateFrame": frame, "targetPointCount": int(len(target)), "targetSampling": {"usedFullInputCloud": True}, "clusters": records, "limitations": ["Plane residuals depend on aerial LiDAR normals and nearest-neighbour association; they do not verify image edge or camera pose accuracy.", "Normal information can be ambiguous for one dominant plane; such cases are rejected.", "No source-index, world-registration, renderer, or facade state was changed."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery", type=Path, required=True)
    parser.add_argument("--normalized-input", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", action="append", default=[])
    args = parser.parse_args()
    report = refine(args.imagery, args.normalized_input, args.world, args.alignment_report, args.cluster)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"clusters": len(report["clusters"]), "candidateEvidence": sum(item.get("planeRefinement", {}).get("status") == "candidate-evidence" for item in report["clusters"])}))


if __name__ == "__main__":
    main()
