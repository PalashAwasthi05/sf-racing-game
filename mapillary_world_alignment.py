"""Bounded diagnostic for Mapillary SfM points versus an independent world cloud.

This module deliberately produces a diagnostic report only.  It does not modify
source-index metadata, admit images, validate world registration, or generate
facades.  Provider pose agreement is therefore kept separate from this external
world check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree


MAX_TRANSLATION_M = 30.0
MAX_ROTATION_DEG = 20.0
MAX_CORRESPONDENCE_M = 8.0
MIN_POINTS = 24
MIN_GEOMETRIC_SPAN_M = 10.0
MIN_HOLDOUT_INLIER_RATIO = 0.50
HOLDOUT_INLIER_M = 5.0
MAX_HOLDOUT_P75_M = 5.0
MIN_IMPROVEMENT = 0.20


def enu_basis(latitude: float, longitude: float) -> np.ndarray:
    """Rows mapping ECEF deltas to east, north, up at an LLA reference."""

    lat, lon = np.radians([latitude, longitude])
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def enu_to_local_projected(
    points_enu: np.ndarray,
    reference_lla: dict[str, float],
    frame: dict[str, Any],
) -> np.ndarray:
    """Convert SfM ENU coordinates to the normalized local projected frame.

    The ECEF round trip avoids treating longitude/latitude degrees as metres and
    preserves the provider's ellipsoidal height.  ``verticalOriginM`` is applied
    explicitly, so the report exposes the vertical datum/origin assumption.
    """

    points_enu = np.asarray(points_enu, dtype=float)
    if points_enu.ndim != 2 or points_enu.shape[1] != 3:
        raise ValueError("points_enu must have shape (N, 3)")
    lat = float(reference_lla["latitude"])
    lon = float(reference_lla["longitude"])
    alt = float(reference_lla.get("altitude", 0.0))
    ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    projected = Transformer.from_crs("EPSG:4978", frame["projectedCrs"], always_xy=True)
    x0, y0, z0 = ecef.transform(lon, lat, alt)
    xyz = np.array([x0, y0, z0])[:, None] + enu_basis(lat, lon).T @ points_enu.T
    x, y, z = projected.transform(xyz[0], xyz[1], xyz[2])
    origin_xy = np.asarray(frame["originProjectedM"], dtype=float)
    vertical_origin = float(frame.get("verticalOriginM", 0.0))
    return np.column_stack((np.asarray(x) - origin_xy[0], np.asarray(y) - origin_xy[1], np.asarray(z) - vertical_origin))


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def _sample_points(points: np.ndarray, maximum: int, seed: int = 1729) -> np.ndarray:
    """Deterministic bounded sample that keeps large clouds cheap."""

    if len(points) <= maximum:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=maximum, replace=False)]


def _voxel_density(points: np.ndarray, voxel_size: float = 1.0) -> dict[str, float | int]:
    """Report density without implying that a sampled cloud is uniform."""

    if not len(points):
        return {"voxelSizeM": voxel_size, "occupiedVoxels": 0, "pointsPerOccupiedVoxel": 0.0, "occupiedVolumeM3": 0.0, "pointsPerM3": 0.0}
    voxels = np.unique(np.floor(points / voxel_size).astype(np.int64), axis=0)
    occupied = int(len(voxels))
    volume = float(occupied * voxel_size**3)
    return {
        "voxelSizeM": voxel_size,
        "occupiedVoxels": occupied,
        "pointsPerOccupiedVoxel": float(len(points) / occupied),
        "occupiedVolumeM3": volume,
        "pointsPerM3": float(len(points) / volume) if volume else 0.0,
    }


def _geometry_report(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {"count": int(len(points)), "spanM": [0.0, 0.0, 0.0], "rank": 0, "diverse": False, "voxelDensity1m": _voxel_density(points)}
    centered = points - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    span = np.ptp(points, axis=0)
    rank = int(np.sum(singular > max(singular[0] * 1e-4, 1e-8)))
    diverse = bool(np.count_nonzero(span[:2] >= MIN_GEOMETRIC_SPAN_M) >= 2 and rank >= 2)
    return {"count": int(len(points)), "spanM": span.tolist(), "rank": rank, "diverse": diverse, "voxelDensity1m": _voxel_density(points)}


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t minimizing ||R source + t - target||, with no scale."""

    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def _metrics(distances: np.ndarray, threshold: float = HOLDOUT_INLIER_M) -> dict[str, Any]:
    if not len(distances):
        return {"count": 0, "medianM": None, "p75M": None, "p90M": None, "coarseOverlapInlierRatio5m": 0.0, "inlierRatioAtM": {"0.25": 0.0, "0.5": 0.0, "1.0": 0.0, "5.0": 0.0}}
    ratios = {str(limit): float(np.mean(distances <= limit)) for limit in (0.25, 0.5, 1.0, 5.0)}
    return {
        "count": int(len(distances)),
        "medianM": float(np.median(distances)),
        "p75M": float(np.percentile(distances, 75)),
        "p90M": float(np.percentile(distances, 90)),
        "coarseOverlapInlierRatio5m": float(np.mean(distances <= threshold)),
        "inlierRatioAtM": ratios,
    }


def _nearest_metrics(points: np.ndarray, tree: cKDTree) -> tuple[dict[str, float], np.ndarray]:
    distances, _ = tree.query(points, k=1)
    return _metrics(np.asarray(distances)), np.asarray(distances)


def bounded_registration(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Robust trimmed ICP with bounded rigid motion and held-out scoring.

    Correspondences are nearest-neighbour proposals only; acceptance is based on
    points withheld from fitting.  This makes a small accidental overlap less
    likely to look like world registration.
    """

    source = _finite_points(source)
    target = _finite_points(target)
    result: dict[str, Any] = {
        "status": "rejected",
        "accepted": False,
        "reason": None,
        "sourceGeometry": _geometry_report(source),
        "targetGeometry": _geometry_report(target),
        # The SfM points have already been converted into the normalized world
        # frame.  Identity is the explicit initial source-world -> target-world
        # transform, and is retained even when registration is rejected.
        "transform": {
            "rotationMatrix": np.eye(3).tolist(),
            "translationM": [0.0, 0.0, 0.0],
            "translationNormM": 0.0,
            "rotationDeg": 0.0,
            "bounded": True,
            "stage": "initial_geodetic_world_frame",
        },
        "evidenceOnly": True,
    }
    result["initialWorldTransform"] = result["transform"].copy()
    result["fittedWorldTransform"] = None
    if len(source) < MIN_POINTS:
        result["reason"] = f"insufficient source points ({len(source)} < {MIN_POINTS})"
        return result
    if not result["sourceGeometry"]["diverse"]:
        result["reason"] = "source point distribution is geometrically degenerate"
        return result
    if len(target) < MIN_POINTS:
        result["reason"] = f"insufficient target points ({len(target)} < {MIN_POINTS})"
        return result

    # Hold out spatial cells rather than point indices.  Neighboring provider
    # points are often duplicates, so index interleaving can give a falsely
    # optimistic score.  Five-metre cells retain geographic coverage while
    # separating fit and evaluation neighborhoods.
    cells = np.floor(source / 5.0).astype(np.int64)
    cell_hash = (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (cells[:, 2] * 83492791)
    holdout_mask = (cell_hash % 5) == 0
    if np.count_nonzero(holdout_mask) < MIN_POINTS // 4 or np.count_nonzero(~holdout_mask) < MIN_POINTS:
        # Tiny synthetic or sparse clouds can have too few occupied cells for a
        # spatial split; report rejection rather than silently reverting to a
        # duplicate-prone point-index split.
        result["reason"] = "inadequate spatial train/holdout split"
        return result
    train = source[~holdout_mask]
    holdout = source[holdout_mask]

    tree = cKDTree(target)
    initial_train, initial_train_d = _nearest_metrics(train, tree)
    initial_holdout, initial_holdout_d = _nearest_metrics(holdout, tree)
    result["initial"] = {"train": initial_train, "holdout": initial_holdout}
    # Do not search arbitrarily for a matching region.  The geodetic seed must
    # already overlap within a bounded neighbourhood.
    if initial_train["medianM"] is None or initial_train["medianM"] > MAX_TRANSLATION_M / 2:
        result["reason"] = "initial geodetic overlap is outside bounded search radius"
        return result

    rotation = np.eye(3)
    translation = np.zeros(3)
    iterations = 0
    for iteration in range(20):
        transformed = _transform(train, rotation, translation)
        distances, indices = tree.query(transformed, k=1)
        candidate = distances <= MAX_CORRESPONDENCE_M
        if np.count_nonzero(candidate) < MIN_POINTS:
            break
        selected = np.asarray(distances)[candidate]
        median = float(np.median(selected))
        mad = float(np.median(np.abs(selected - median)))
        robust_limit = min(MAX_CORRESPONDENCE_M, max(1.5, median + 2.5 * 1.4826 * mad))
        candidate &= np.asarray(distances) <= robust_limit
        if np.count_nonzero(candidate) < MIN_POINTS:
            break
        delta_r, delta_t = _rigid_fit(transformed[candidate], target[indices[candidate]])
        proposed_r = delta_r @ rotation
        proposed_t = delta_r @ translation + delta_t
        if np.linalg.norm(proposed_t) > MAX_TRANSLATION_M or _rotation_angle_deg(proposed_r) > MAX_ROTATION_DEG:
            break
        change = np.linalg.norm(proposed_t - translation) + _rotation_angle_deg(delta_r)
        rotation, translation = proposed_r, proposed_t
        iterations = iteration + 1
        if change < 1e-4:
            break

    final_train, _ = _nearest_metrics(_transform(train, rotation, translation), tree)
    final_holdout, _ = _nearest_metrics(_transform(holdout, rotation, translation), tree)
    result["final"] = {"train": final_train, "holdout": final_holdout}
    result["iterations"] = iterations
    result["transform"] = {
        "rotationMatrix": rotation.tolist(),
        "translationM": translation.tolist(),
        "translationNormM": float(np.linalg.norm(translation)),
        "rotationDeg": _rotation_angle_deg(rotation),
        "bounded": bool(np.linalg.norm(translation) <= MAX_TRANSLATION_M and _rotation_angle_deg(rotation) <= MAX_ROTATION_DEG),
        "stage": "fitted_bounded_rigid",
    }
    result["fittedWorldTransform"] = result["transform"].copy()
    initial_p75 = initial_holdout["p75M"]
    final_p75 = final_holdout["p75M"]
    improvement = None
    if initial_p75 is not None and initial_p75 > 1e-9 and final_p75 is not None:
        improvement = float((initial_p75 - final_p75) / initial_p75)
    result["holdoutImprovement"] = improvement
    initially_plausible = bool(
        initial_p75 is not None
        and initial_p75 <= MAX_HOLDOUT_P75_M
        and initial_holdout["coarseOverlapInlierRatio5m"] >= MIN_HOLDOUT_INLIER_RATIO
    )
    final_plausible = bool(
        final_p75 is not None
        and final_p75 <= MAX_HOLDOUT_P75_M
        and final_holdout["coarseOverlapInlierRatio5m"] >= MIN_HOLDOUT_INLIER_RATIO
    )
    result["accepted"] = bool(
        result["transform"]["bounded"]
        and final_plausible
        and (initially_plausible or (improvement is not None and improvement >= MIN_IMPROVEMENT))
    )
    result["status"] = "plausible" if result["accepted"] else "rejected"
    if not result["accepted"]:
        if not result["transform"]["bounded"]:
            result["reason"] = "bounded rigid transform could not be maintained"
        elif not final_plausible:
            result["reason"] = "held-out alignment remains above conservative thresholds"
        else:
            result["reason"] = "held-out improvement is insufficient"
    return result


def _world_frame(normalized: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
    frame = dict(normalized.get("coordinateFrame") or world.get("coordinateFrame") or {})
    required = ("projectedCrs", "originProjectedM", "verticalOriginM")
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError("normalized coordinateFrame must provide " + ", ".join(missing) + "; no location-specific fallback is allowed")
    if len(frame["originProjectedM"]) != 2:
        raise ValueError("coordinateFrame.originProjectedM must contain exactly [x, y]")
    frame["source"] = "normalized-input coordinateFrame"
    frame["kind"] = "normalized-local-projected-frame"
    return frame


def load_world_points(normalized_path: Path, normalized: dict[str, Any], world: dict[str, Any]) -> np.ndarray:
    """Load independent LiDAR points, preserving normalized local coordinates."""

    arrays: list[np.ndarray] = []
    for key in ("surfacePointsFile", "groundFile"):
        value = normalized.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = normalized_path.parent / path
        if path.exists():
            with np.load(path) as data:
                if "points" in data:
                    arrays.append(np.asarray(data["points"], dtype=float)[:, :3])
    if not arrays:
        cloud = (world.get("paths") or {}).get("cloud") or world.get("cloudFile")
        if cloud:
            path = Path(cloud)
            if not path.is_absolute():
                path = normalized_path.parent / path
            if path.exists() and path.suffix == ".npz":
                with np.load(path) as data:
                    key = "points" if "points" in data else data.files[0]
                    arrays.append(np.asarray(data[key], dtype=float)[:, :3])
    if not arrays:
        raise FileNotFoundError("no independent LiDAR point cloud found in normalized input or world paths")
    return _finite_points(np.vstack(arrays))


def _crop_source_to_target_xy(source: np.ndarray, target: np.ndarray, margin_m: float = MAX_TRANSLATION_M) -> tuple[np.ndarray, dict[str, Any]]:
    """Bound the diagnostic to the independent cloud's XY support plus margin."""

    source = _finite_points(source)
    target = _finite_points(target)
    if not len(source) or not len(target):
        return source[:0], {"inputCount": int(len(source)), "keptCount": 0, "excludedCount": int(len(source)), "marginM": margin_m, "targetBoundsXY": None}
    lo = np.min(target[:, :2], axis=0) - margin_m
    hi = np.max(target[:, :2], axis=0) + margin_m
    keep = np.all((source[:, :2] >= lo) & (source[:, :2] <= hi), axis=1)
    return source[keep], {"inputCount": int(len(source)), "keptCount": int(np.count_nonzero(keep)), "excludedCount": int(np.count_nonzero(~keep)), "marginM": margin_m, "targetBoundsXY": [lo.tolist(), hi.tolist()]}


def _cluster_ids(imagery: Path) -> list[str]:
    source_index = imagery / "source-index.json"
    if source_index.exists():
        data = json.loads(source_index.read_text())
        ids = {str(v.get("camera", {}).get("clusterId")) for v in data.values() if v.get("camera", {}).get("clusterId")}
        return sorted(ids)
    return sorted(p.stem for p in (imagery / "clusters").glob("*.json"))


def _cluster_image_counts(imagery: Path) -> dict[str, int]:
    source_index = imagery / "source-index.json"
    if not source_index.exists():
        return {}
    data = json.loads(source_index.read_text())
    counts: dict[str, int] = {}
    for value in data.values():
        cluster_id = value.get("camera", {}).get("clusterId")
        if cluster_id:
            key = str(cluster_id)
            counts[key] = counts.get(key, 0) + 1
    return counts


def diagnose(imagery: Path, normalized_path: Path, world_path: Path, maximum_clusters: int | None = None, selected_clusters: Iterable[str] | None = None) -> dict[str, Any]:
    normalized = json.loads(normalized_path.read_text())
    world = json.loads(world_path.read_text())
    frame = _world_frame(normalized, world)
    world_points = load_world_points(normalized_path, normalized, world)
    target = _sample_points(world_points, 80_000, seed=11)
    ids = list(selected_clusters) if selected_clusters is not None else _cluster_ids(imagery)
    ids = sorted(set(ids))
    if maximum_clusters is not None:
        ids = ids[:maximum_clusters]
    image_counts = _cluster_image_counts(imagery)
    records = []
    for cluster_id in ids:
        path = imagery / "clusters" / f"{cluster_id}.json"
        record: dict[str, Any] = {"clusterId": cluster_id, "selectedImageCount": int(image_counts.get(cluster_id, 0)), "status": "rejected"}
        if not path.exists():
            record["reason"] = "cluster file missing"
            records.append(record)
            continue
        try:
            payload = json.loads(path.read_text())
            cluster = payload[0] if isinstance(payload, list) else payload
            points_map = cluster.get("points", {})
            points = np.asarray([p["coordinates"] for p in points_map.values()], dtype=float)
            reference = cluster.get("reference_lla")
            if not reference:
                record["reason"] = "cluster has no reference_lla"
                records.append(record)
                continue
            local = enu_to_local_projected(points, reference, frame)
            source_sample = _sample_points(local, 20_000, seed=19)
            cropped, crop_report = _crop_source_to_target_xy(source_sample, world_points)
            registration = bounded_registration(cropped, target)
            record.update({"referenceLla": reference, "sfmPointCount": int(len(points)), "sfmPointCountSampled": int(len(source_sample)), "verticalDatumOffsetM": float(reference.get("altitude", 0.0)) - float(frame.get("verticalOriginM", 0.0)), "sourceCropXY": crop_report, "registration": registration})
        except Exception as exc:  # keep one malformed provider cluster from hiding the report
            record["reason"] = f"cluster could not be assessed: {type(exc).__name__}: {exc}"
        records.append(record)
    accepted = sum(r.get("registration", {}).get("accepted", False) for r in records)
    return {
        "schema": "imagery-world-registration-diagnostic/v1",
        "worldRegistrationValidated": False,
        "facadeMeasurementsGenerated": 0,
        "providerPoseAgreementIsNotWorldRegistration": True,
        "evidenceScope": "candidate alignment evidence only; never an admission or registration validation decision",
        "inputs": {"imagery": str(imagery.resolve()), "normalizedInput": str(normalized_path.resolve()), "world": str(world_path.resolve()), "clustersRequested": len(ids)},
        "coordinateFrame": frame,
        "coordinateFrameSource": "normalized-input coordinateFrame (local projected axes; not a detailed world-registration description)",
        "thresholds": {"maxTranslationM": MAX_TRANSLATION_M, "maxRotationDeg": MAX_ROTATION_DEG, "maxCorrespondenceM": MAX_CORRESPONDENCE_M, "coarseOverlapInlierM": HOLDOUT_INLIER_M, "coarseOverlapInlierRatioMinimum": MIN_HOLDOUT_INLIER_RATIO, "maxHoldoutP75M": MAX_HOLDOUT_P75_M, "facadeConservativeInlierThresholdsM": [0.25, 0.5, 1.0], "minImprovement": MIN_IMPROVEMENT, "sourceCropMarginM": MAX_TRANSLATION_M},
        "worldPointCount": int(len(world_points)),
        "worldPointCountSampled": int(len(target)),
        "summary": {"clustersExamined": len(records), "clustersWithCandidateEvidence": int(accepted), "clustersRejected": len(records) - int(accepted), "plausible": bool(records) and accepted == len(records)},
        "clusters": records,
        "limitations": ["The 5 m inlier ratio is a coarse overlap diagnostic only and is not a facade-accuracy threshold; 0.25/0.5/1 m ratios are reported for scrutiny.", "SfM/LiDAR nearest-neighbour agreement is a diagnostic, not semantic feature correspondence.", "Provider-internal camera/shot pose agreement does not validate world registration.", "Vertical conversion uses WGS84 ellipsoidal height minus normalized verticalOriginM; any unmodeled LiDAR datum bias remains an uncertainty.", "No source-index admission, worldRegistrationValidated flag, facade geometry, or renderer input was changed."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery", type=Path, required=True)
    parser.add_argument("--normalized-input", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--cluster", action="append", dest="clusters", help="cluster id to assess; repeatable")
    args = parser.parse_args()
    report = diagnose(args.imagery, args.normalized_input, args.world, args.max_clusters, args.clusters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
