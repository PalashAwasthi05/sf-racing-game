"""Create bounded, unregistered street-facade photo rectification diagnostics.

This command only consumes local imagery and normalized geometry.  It emits
photo crops plus provenance; it does not edit the imagery manifest, validate a
world registration, infer openings, or publish textures.  The conversion from
the normalized projected frame to a provider cluster's ENU frame is measured at
the four crop corners so a cheap local affine approximation remains inspectable.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import LineString, Point, shape

from general_world.evidence import contiguous_frames
from general_world.image_projection import CameraTransform, Orientation, project_facade_crop


DEFAULT_LIMIT = 24
RECENT_DAYS = 730
MIN_FRONTAGE_M = 4.0
MAX_FACADE_WIDTH_M = 30.0
MAX_DIAGNOSTIC_HEIGHT_M = 30.0
PIXELS_PER_M = 8.0
MAX_CAMERA_DISTANCE_M = 300.0


def _enu_basis(latitude: float, longitude: float) -> np.ndarray:
    lat, lon = np.radians([latitude, longitude])
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ],
        dtype=float,
    )


class FrameConverter:
    """Convert normalized local projected points and cluster ENU points."""

    def __init__(self, frame: dict[str, Any], reference: dict[str, Any], inverse_alignment: dict[str, Any] | None = None):
        required = ("projectedCrs", "originProjectedM", "verticalOriginM")
        missing = [k for k in required if k not in frame]
        if missing:
            raise ValueError("normalized coordinateFrame missing " + ", ".join(missing))
        self.frame = frame
        self.reference = reference
        self.inverse_alignment = inverse_alignment
        self.origin_xy = np.asarray(frame["originProjectedM"], dtype=float)
        self.vertical_origin = float(frame["verticalOriginM"])
        self.to_lla = Transformer.from_crs(frame["projectedCrs"], "EPSG:4979", always_xy=True)
        self.to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        self.from_ecef = Transformer.from_crs("EPSG:4978", frame["projectedCrs"], always_xy=True)
        self.reference_lla = (
            float(reference["latitude"]),
            float(reference["longitude"]),
            float(reference.get("altitude", 0.0)),
        )
        lat, lon, alt = self.reference_lla
        self.ecef0 = np.asarray(self.to_ecef.transform(lon, lat, alt), dtype=float)
        self.basis = _enu_basis(lat, lon)

    def world_to_ecef(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=float).reshape(-1, 3)
        if self.inverse_alignment is not None:
            # The report stores source->normalized world R,t.  Apply its
            # inverse only to this diagnostic path, before geodetic conversion.
            rotation = np.asarray(self.inverse_alignment["rotationMatrix"], dtype=float)
            translation = np.asarray(self.inverse_alignment["translationM"], dtype=float)
            points = (points - translation[None, :]) @ rotation
        x = points[:, 0] + self.origin_xy[0]
        y = points[:, 1] + self.origin_xy[1]
        z = points[:, 2] + self.vertical_origin
        lon, lat, alt = self.to_lla.transform(x, y, z)
        ecef = self.to_ecef.transform(lon, lat, alt)
        return np.column_stack(ecef)

    def world_to_cluster_enu(self, points_world: np.ndarray) -> np.ndarray:
        delta = self.world_to_ecef(points_world) - self.ecef0
        return delta @ self.basis.T

    def cluster_enu_to_world(self, points_enu: np.ndarray) -> np.ndarray:
        points = np.asarray(points_enu, dtype=float).reshape(-1, 3)
        ecef = self.ecef0 + points @ self.basis
        x, y, z = self.from_ecef.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
        points_world = np.column_stack((np.asarray(x) - self.origin_xy[0], np.asarray(y) - self.origin_xy[1], np.asarray(z) - self.vertical_origin))
        if self.inverse_alignment is not None:
            rotation = np.asarray(self.inverse_alignment["rotationMatrix"], dtype=float)
            translation = np.asarray(self.inverse_alignment["translationM"], dtype=float)
            # Inverse-alignment metadata is source->world in this direction.
            points_world = points_world @ rotation.T + translation[None, :]
        return points_world

    def ecef_to_world(self, ecef: np.ndarray) -> np.ndarray:
        points = np.asarray(ecef, dtype=float).reshape(-1, 3)
        x, y, z = self.from_ecef.transform(points[:, 0], points[:, 1], points[:, 2])
        return np.column_stack((np.asarray(x) - self.origin_xy[0], np.asarray(y) - self.origin_xy[1], np.asarray(z) - self.vertical_origin))


def _local_affine(converter: FrameConverter, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``enu = offset + (world-anchor) @ jacobian.T`` at one crop."""

    center = converter.world_to_cluster_enu(np.asarray(anchor, dtype=float)[None, :])[0]
    columns = []
    for axis in range(3):
        delta = np.zeros(3, dtype=float)
        delta[axis] = 1.0
        columns.append(converter.world_to_cluster_enu((np.asarray(anchor) + delta)[None, :])[0] - center)
    return center, np.column_stack(columns)


def _affine_error(converter: FrameConverter, anchor: np.ndarray, corners: np.ndarray) -> dict[str, Any]:
    offset, jacobian = _local_affine(converter, anchor)
    approx = offset[None, :] + (corners - anchor[None, :]) @ jacobian.T
    exact = converter.world_to_cluster_enu(corners)
    residual = np.linalg.norm(exact - approx, axis=1)
    return {
        "cornerErrorsM": [float(x) for x in residual],
        "maxCornerErrorM": float(np.max(residual)) if len(residual) else 0.0,
        "meanCornerErrorM": float(np.mean(residual)) if len(residual) else 0.0,
    }


def _orthogonalize_projection_basis(horizontal: np.ndarray, vertical: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Preserve local metric lengths while removing CRS shear for the crop API."""

    h = np.asarray(horizontal, dtype=float)
    v = np.asarray(vertical, dtype=float)
    h_length = float(np.linalg.norm(h))
    v_length = float(np.linalg.norm(v))
    if h_length == 0.0 or v_length == 0.0:
        return h, v, False
    h_unit = h / h_length
    v_perpendicular = v - np.dot(v, h_unit) * h_unit
    v_perpendicular_length = float(np.linalg.norm(v_perpendicular))
    if v_perpendicular_length == 0.0:
        raise ValueError("affine facade basis collapsed")
    changed = not np.isclose(np.dot(h, v), 0.0, atol=1e-7)
    return h_unit * h_length, v_perpendicular / v_perpendicular_length * v_length, bool(changed)


def _height_for_building(building: dict[str, Any]) -> tuple[float | None, str]:
    """Read supplied height evidence without inventing a building-specific value."""

    for key in ("pointSupportedHeightM", "heightM", "roofHeightM"):
        value = building.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value, key
    coords = np.asarray(building.get("geometry", {}).get("coordinates", []), dtype=object)
    # A supplied 3-D footprint is an external geometry height cue.  A 2-D
    # footprint remains explicitly unknown for diagnostic provenance.
    try:
        z_values = []
        def walk(value: Any) -> None:
            if isinstance(value, (list, tuple)) and len(value) == 3 and all(np.isscalar(v) for v in value):
                if np.isfinite(float(value[2])):
                    z_values.append(float(value[2]))
            elif isinstance(value, (list, tuple)):
                for child in value:
                    walk(child)
        walk(building.get("geometry", {}).get("coordinates", []))
        if len(z_values) >= 2 and max(z_values) > min(z_values):
            return max(z_values) - min(z_values), "externalGeometryZRange"
    except (TypeError, ValueError):
        pass
    return None, "unknown"


def _load_ground_points(normalized_input: Path, normalized: dict[str, Any]) -> np.ndarray:
    value = normalized.get("groundFile")
    if not value:
        return np.empty((0, 3), dtype=float)
    path = Path(value)
    if not path.is_absolute():
        path = normalized_input.parent / path
    if not path.exists():
        return np.empty((0, 3), dtype=float)
    try:
        with np.load(path) as data:
            if "points" not in data:
                return np.empty((0, 3), dtype=float)
            points = np.asarray(data["points"], dtype=float)
        points = points[:, :3]
        return points[np.isfinite(points).all(axis=1)]
    except (OSError, ValueError, KeyError):
        return np.empty((0, 3), dtype=float)


def _base_z_for_frontage(frontage: dict[str, Any], ground_points: np.ndarray) -> tuple[float, str]:
    if len(ground_points) == 0:
        return 0.0, "unknown-coordinate-origin"
    a = np.asarray(frontage["a"], dtype=float)
    b = np.asarray(frontage["b"], dtype=float)
    vector = b - a
    length_sq = float(np.dot(vector, vector))
    if length_sq <= 0.0:
        return 0.0, "unknown-coordinate-origin"
    offsets = ground_points[:, :2] - a[None, :2]
    t = np.clip((offsets @ vector[:2]) / length_sq, 0.0, 1.0)
    nearest = a[None, :2] + t[:, None] * vector[:2]
    distance = np.linalg.norm(ground_points[:, :2] - nearest, axis=1)
    nearby = ground_points[distance <= 1.5]
    if len(nearby) == 0:
        return 0.0, "unknown-near-frontage"
    return float(np.median(nearby[:, 2])), "normalized-ground-near-frontage"


def _load_clusters(imagery: Path) -> dict[str, dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for path in sorted((imagery / "clusters").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            cluster = payload[0] if isinstance(payload, list) else payload
            if isinstance(cluster, dict):
                clusters[path.stem] = cluster
        except (OSError, ValueError, TypeError, IndexError):
            continue
    return clusters


def _capture_age(record: dict[str, Any], as_of: date) -> int | None:
    try:
        return (as_of - date.fromisoformat(record["captureDate"])).days
    except (KeyError, TypeError, ValueError):
        return None


def _load_alignment_report(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    alignments: dict[str, dict[str, Any]] = {}
    for entry in payload.get("clusters", []):
        cluster_id = entry.get("clusterId")
        registration = entry.get("registration") or {}
        transform = registration.get("transform")
        if cluster_id is None or registration.get("accepted") is not True or not transform:
            continue
        rotation = np.asarray(transform.get("rotationMatrix"), dtype=float)
        translation = np.asarray(transform.get("translationM"), dtype=float)
        if rotation.shape == (3, 3) and translation.shape == (3,) and np.isfinite(rotation).all() and np.isfinite(translation).all():
            alignments[str(cluster_id)] = {"rotationMatrix": rotation.tolist(), "translationM": translation.tolist(), "source": str(path.resolve())}
    return alignments


def _camera_records(
    imagery: Path,
    source_index: dict[str, Any],
    frame: dict[str, Any],
    clusters: dict[str, dict[str, Any]],
    as_of: date,
    alignments: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    alignments = alignments or {}
    records = []
    for source_id, record in source_index.items():
        camera = record.get("camera") or {}
        image_path = Path(record.get("imagePath", ""))
        if not image_path.is_file() or camera.get("providerPoseConsistent") is not True:
            continue
        if str(camera.get("model", "")).lower() not in {"spherical", "equirectangular"} or camera.get("orientation") != 1:
            continue
        age = _capture_age(record, as_of)
        if age is None or age < 0 or age > RECENT_DAYS:
            continue
        cluster_id = camera.get("clusterId")
        cluster = clusters.get(str(cluster_id))
        reference = camera.get("referenceLla") or (cluster or {}).get("reference_lla")
        if not reference or camera.get("translation") is None or camera.get("rotationAngleAxis") is None:
            continue
        try:
            alignment = alignments.get(str(cluster_id))
            converter = FrameConverter(frame, reference, alignment)
            transform = CameraTransform(camera["rotationAngleAxis"], camera["translation"], Orientation.EQUIRECTANGULAR, False, diagnostic=True)
            center_enu = transform.camera_center
            center_world = converter.cluster_enu_to_world(center_enu)[0]
            center_ecef = converter.ecef0 + center_enu @ converter.basis
        except (ValueError, TypeError, KeyError, IndexError):
            continue
        records.append({"sourceId": source_id, "record": record, "camera": camera, "converter": converter, "transform": transform, "centerWorld": center_world, "centerEcef": center_ecef, "ageDays": age, "alignment": alignment})
    return records


def _frontage_rectangle(frame_record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a = np.asarray(frame_record["a"], dtype=float)
    b = np.asarray(frame_record["b"], dtype=float)
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length < MIN_FRONTAGE_M:
        raise ValueError("frontage shorter than minimum")
    unit = direction / length
    frame_normal = np.asarray(frame_record["normal"], dtype=float)
    cross_normal = np.array([unit[1], -unit[0]])
    sign = 1.0 if float(np.dot(cross_normal, frame_normal)) >= 0.0 else -1.0
    horizontal = np.array([unit[0], unit[1], 0.0], dtype=float) * sign
    width = min(length, MAX_FACADE_WIDTH_M)
    midpoint = (a + b) / 2.0
    origin = np.array([midpoint[0], midpoint[1], float(frame_record.get("baseZM", 0.0))], dtype=float) - horizontal * (width * 0.5)
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    return origin, horizontal, vertical, width


def _facing_cosines(frame_record: dict[str, Any], camera_center: np.ndarray) -> tuple[float, list[float]]:
    origin, horizontal, vertical, width = _frontage_rectangle(frame_record)
    height = frame_record.get("heightM")
    diagnostic_height = MAX_DIAGNOSTIC_HEIGHT_M if height is None else max(0.5, float(height))
    normal = np.cross(horizontal, vertical)
    normal /= np.linalg.norm(normal)
    points = np.array(
        [
            origin,
            origin + horizontal * width,
            origin + vertical * diagnostic_height,
            origin + horizontal * width + vertical * diagnostic_height,
            origin + horizontal * (width * 0.5) + vertical * (diagnostic_height * 0.5),
        ]
    )
    rays = camera_center[None, :] - points
    lengths = np.linalg.norm(rays, axis=1)
    cosines = np.einsum("j,nj->n", normal, rays) / np.maximum(lengths, np.finfo(float).tiny)
    return float(cosines[-1]), [float(value) for value in cosines[:4]]


def _intersection_distances(line: LineString, geometry: Any) -> list[float]:
    boundary = line.intersection(geometry.boundary)
    if boundary.is_empty:
        return []
    geometries = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
    distances: list[float] = []
    for item in geometries:
        if hasattr(item, "coords"):
            coordinates = list(item.coords)
        else:
            coordinates = []
        distances.extend(float(line.project(Point(x, y))) for x, y, *_ in coordinates)
    return distances


def _line_of_sight_clear(camera_xy: np.ndarray, target_xy: np.ndarray, owner_key: str, footprints: list[tuple[str, Any]]) -> bool:
    """Reject a frontage hidden behind another footprint in plan view."""

    line = LineString([tuple(camera_xy), tuple(target_xy)])
    total = float(line.length)
    if total <= 0.0:
        return False
    # Camera uncertainty is not permission to see metres through the host.
    trimmed = LineString([tuple(camera_xy), line.interpolate(max(0.0, total - 0.5))])
    for key, footprint in footprints:
        if footprint.is_empty:
            continue
        if footprint.buffer(-1e-7).intersects(trimmed):
            return False

    return True


def _candidate_for_frame(frame_record: dict[str, Any], cameras: list[dict[str, Any]], footprints: list[tuple[str, Any]]) -> dict[str, Any] | None:
    a = np.asarray(frame_record["a"], dtype=float)
    b = np.asarray(frame_record["b"], dtype=float)
    length = float(np.linalg.norm(b - a))
    if length < MIN_FRONTAGE_M:
        return None
    center = (a + b) / 2.0
    world_center = np.array([center[0], center[1], float(frame_record.get("baseZM", 0.0))])
    choices = []
    for camera in cameras:
        delta = camera["centerWorld"] - world_center
        # Use the ECEF distance for selection while retaining the local
        # projected vector for the frontage-facing test.
        center_ecef = camera["converter"].world_to_ecef(world_center[None, :])[0]
        distance = float(np.linalg.norm(camera["centerEcef"] - center_ecef))
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        if horizontal_distance < 2.0 or horizontal_distance > MAX_CAMERA_DISTANCE_M:
            continue
        midpoint_facing, corner_facings = _facing_cosines(frame_record, camera["centerWorld"])
        if midpoint_facing < 0.5 or min(corner_facings) < 0.25:
            continue
        if not _line_of_sight_clear(camera["centerWorld"][:2], center, str(frame_record["buildingKey"]), footprints):
            continue
        # Prefer nearby, near-normal views while preserving recency as a
        # deterministic tie-break.  This score has no source-specific values.
        score = distance * (1.0 + max(0.0, 1.0 - midpoint_facing)) + 0.01 * camera["ageDays"]
        choices.append((score, camera, distance, midpoint_facing, corner_facings))
    if not choices:
        return None
    _, camera, distance, facing, corner_facings = min(choices, key=lambda x: (x[0], tuple(x[1]["centerWorld"])))
    result = dict(frame_record)
    result.update({"camera": camera, "distanceM": distance, "facingCosine": facing, "cornerFacingCosines": corner_facings, "screenVisibilityGeometryOnly": True})
    return result


def _select_spatial(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    # First round: at most one frontage per building, preferring supported
    # heights above four metres and longer visible frontage intervals.
    best_by_building: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["buildingKey"]
        current = best_by_building.get(key)
        preference = (candidate["heightM"] is not None and candidate["heightM"] > 4.0, candidate["frontageLengthM"], candidate["facingCosine"], -candidate["distanceM"])
        if current is None or preference > (current["heightM"] is not None and current["heightM"] > 4.0, current["frontageLengthM"], current["facingCosine"], -current["distanceM"]):
            best_by_building[key] = candidate
    first_round = list(best_by_building.values())
    remaining = [candidate for candidate in candidates if candidate not in first_round]

    def choose(pool: list[dict[str, Any]], *, spread_weight: bool) -> dict[str, Any]:
        if not selected:
            return max(pool, key=lambda c: (c["heightM"] is not None and c["heightM"] > 4.0, c["frontageLengthM"], c["facingCosine"], -c["distanceM"], tuple(c["midpoint"])))
        def score(candidate: dict[str, Any]) -> tuple[float, bool, float, float, str]:
            midpoint = np.asarray(candidate["midpoint"], dtype=float)
            spread = min(float(np.linalg.norm(midpoint - np.asarray(other["midpoint"], dtype=float))) for other in selected)
            known = candidate["heightM"] is not None and candidate["heightM"] > 4.0
            return ((spread if spread_weight else 0.0), known, candidate["frontageLengthM"], candidate["facingCosine"], tuple(candidate["midpoint"]))
        return max(pool, key=score)

    while first_round and len(selected) < limit:
        pick = choose(first_round, spread_weight=True)
        selected.append(pick)
        first_round.remove(pick)
    while remaining and len(selected) < limit:
        pick = choose(remaining, spread_weight=True)
        selected.append(pick)
        remaining.remove(pick)
    return selected


def _independent_cameras(first, cameras, minimum_baseline_m=2.0):
    group = first["record"].get("viewGroup")
    if not group or group in ("None", "unknown"):
        return []
    return [camera for camera in cameras
            if camera["record"].get("viewGroup") not in (None, "None", "unknown", group)
            and np.linalg.norm(camera["centerWorld"]-first["centerWorld"]) >= minimum_baseline_m]


def _paired_selection(candidates, cameras, footprints, limit):
    selected = []
    for first in _select_spatial(candidates, len(candidates)):
        if len(selected) + 2 > limit:
            break
        second = _candidate_for_frame(first, _independent_cameras(first["camera"], cameras), footprints)
        if second is not None:
            selected.extend([first, second])
    return selected


def _save_contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((320, 240))
                thumbs.append((path.name, image.copy()))
        except (OSError, ValueError):
            continue
    if not thumbs:
        return
    columns = min(4, len(thumbs))
    cell_w, cell_h = 320, 270
    sheet = Image.new("RGB", (columns * cell_w, math.ceil(len(thumbs) / columns) * cell_h), (24, 28, 31))
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(thumbs):
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 4))
        draw.text((x + 6, y + 247), name, fill=(240, 240, 240))
    sheet.save(output)


def rectify(imagery: Path, normalized_input: Path, output: Path, limit: int = DEFAULT_LIMIT, alignment_report: Path | None = None, pixels_per_m: float = PIXELS_PER_M, paired_views: bool = False) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not np.isfinite(pixels_per_m) or not 1 <= pixels_per_m <= 128:
        raise ValueError('pixels_per_m must be finite and between 1 and 128')
    normalized = json.loads(normalized_input.read_text())
    frame = normalized.get("coordinateFrame") or {}
    source_path = imagery / "source-index.json"
    if not source_path.exists():
        raise FileNotFoundError(f"missing local source index: {source_path}")
    source_index = json.loads(source_path.read_text())
    clusters = _load_clusters(imagery)
    alignments = _load_alignment_report(alignment_report)
    as_of = date.fromisoformat(str(normalized.get("snapshotDate")))
    cameras = _camera_records(imagery, source_index, frame, clusters, as_of, alignments)
    if not cameras:
        raise ValueError("no recent local providerPoseConsistent spherical orientation-1 cameras")

    ground_points = _load_ground_points(normalized_input, normalized)
    footprints: list[tuple[str, Any]] = []
    for building in normalized.get("buildings", []):
        try:
            geometry = shape(building.get("geometry"))
        except (TypeError, ValueError):
            continue
        footprints.append((str(building.get("key", "unknown")), geometry))
    candidates: list[dict[str, Any]] = []
    for building in normalized.get("buildings", []):
        geometry = building.get("geometry")
        if not geometry:
            continue
        try:
            footprint = shape(geometry)
            frames = contiguous_frames(footprint, min_length=MIN_FRONTAGE_M)
        except (TypeError, ValueError):
            continue
        height, height_source = _height_for_building(building)
        for index, frontage in enumerate(frames):
            base_z, base_z_source = _base_z_for_frontage(frontage, ground_points)
            candidate = _candidate_for_frame({**frontage, "buildingKey": str(building.get("key", "unknown")), "frameIndex": index, "baseZM": base_z, "heightM": height}, cameras, footprints)
            if candidate is None:
                continue
            a, b = np.asarray(candidate["a"], dtype=float), np.asarray(candidate["b"], dtype=float)
            length = float(np.linalg.norm(b - a))
            candidate.update(
                {
                    "heightM": None if height is None else min(float(height), MAX_DIAGNOSTIC_HEIGHT_M),
                    "heightSource": height_source,
                    "baseZM": base_z,
                    "baseZSource": base_z_source,
                    "frontageLengthM": length,
                    "midpoint": ((a + b) / 2.0).tolist(),
                }
            )
            candidates.append(candidate)
    selected = _paired_selection(candidates, cameras, footprints, limit) if paired_views else _select_spatial(candidates, limit)
    output.mkdir(parents=True, exist_ok=True)
    crops_dir = output / "crops"
    crops_dir.mkdir(exist_ok=True)
    crop_records: list[dict[str, Any]] = []
    crop_paths: list[Path] = []
    failures: list[dict[str, Any]] = []
    for crop_index, candidate in enumerate(selected):
        camera = candidate["camera"]
        record, transform, converter = camera["record"], camera["transform"], camera["converter"]
        height = candidate["heightM"]
        # Unknown height is retained in provenance.  A bounded inspection
        # extent keeps the diagnostic useful without claiming a measurement.
        diagnostic_height = MAX_DIAGNOSTIC_HEIGHT_M if height is None else max(0.5, height)
        a, b = np.asarray(candidate["a"], dtype=float), np.asarray(candidate["b"], dtype=float)
        direction = b - a
        frontage_length = float(np.linalg.norm(direction))
        width = min(frontage_length, MAX_FACADE_WIDTH_M)
        unit_horizontal = np.array([direction[0], direction[1], 0.0], dtype=float) / frontage_length
        frame_normal = np.asarray(candidate["normal"], dtype=float)
        cross_normal = np.array([unit_horizontal[1], -unit_horizontal[0]])
        sign = 1.0 if float(np.dot(cross_normal, frame_normal)) >= 0.0 else -1.0
        horizontal = unit_horizontal * sign
        midpoint = (a + b) / 2.0
        origin = np.array([midpoint[0], midpoint[1], candidate["baseZM"]], dtype=float) - horizontal * (width * 0.5)
        vertical = np.array([0.0, 0.0, 1.0], dtype=float)
        corners = np.array([origin, origin + horizontal * width, origin + vertical * diagnostic_height, origin + horizontal * width + vertical * diagnostic_height])
        anchor = origin + horizontal * (width * 0.5) + vertical * (diagnostic_height * 0.5)
        affine_error = _affine_error(converter, anchor, corners)
        affine_offset, affine_jacobian = _local_affine(converter, anchor)
        # CameraTransform is expressed in cluster ENU coordinates.  Project
        # the locally affine-converted rectangle in that frame; the exact
        # corner conversion residual above remains part of provenance.
        projection_origin = affine_offset + (origin - anchor) @ affine_jacobian.T
        projection_horizontal = horizontal @ affine_jacobian.T
        projection_vertical = vertical @ affine_jacobian.T
        projection_horizontal, projection_vertical, basis_orthogonalized = _orthogonalize_projection_basis(projection_horizontal, projection_vertical)
        try:
            with Image.open(record["imagePath"]) as source:
                image = np.asarray(source.convert("RGB"))
            # Keep this explicitly diagnostic and unregistered.  No occlusion
            # segmentation is fabricated for local source images.
            crop = project_facade_crop(image, transform, projection_origin, projection_horizontal, projection_vertical, width, diagnostic_height, pixels_per_m, diagnostic=True)
            filename = f"crop-{crop_index:03d}.png"
            path = crops_dir / filename
            alpha = np.where(crop.valid_mask, 255, 0).astype(np.uint8)
            rgba = np.dstack((crop.rgb, alpha))
            Image.fromarray(np.flipud(rgba)).save(path)
            mask_path = crops_dir / f"valid-mask-{crop_index:03d}.png"
            Image.fromarray(np.flipud(alpha), mode="L").save(mask_path)
            crop_paths.append(path)
        except (OSError, ValueError, TypeError) as exc:
            failures.append({"index": crop_index, "buildingKey": candidate["buildingKey"], "sourceId": camera["sourceId"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        crop_records.append(
            {
                "index": crop_index,
                "path": str(path.resolve()),
                "validMaskPath": str(mask_path.resolve()),
                "buildingKey": candidate["buildingKey"],
                "frontageFrameIndex": candidate["frameIndex"],
                "frontageLengthM": candidate["frontageLengthM"],
                "widthM": width,
                "heightM": height,
                "diagnosticHeightM": diagnostic_height,
                "heightSource": candidate["heightSource"],
                "sourceId": camera["sourceId"],
                "captureDate": record.get("captureDate"),
                "viewGroup": record.get("viewGroup"),
                "metricFrame": {"origin": origin.tolist(), "horizontal": horizontal.tolist(), "vertical": vertical.tolist()},
                "sourceUrl": record.get("sourceUrl"),
                "imagePath": str(Path(record["imagePath"]).resolve()),
                "ageDays": camera["ageDays"],
                "providerPoseConsistent": True,
                "worldRegistrationValidated": False,
                "diagnostic": True,
                "occlusionUnchecked": True,
                "admitted": False,
                "facadeMeasurementCertified": False,
                "openingInferenceCertified": False,
                "baseZM": candidate["baseZM"],
                "baseZSource": candidate["baseZSource"],
                "validProjectionFraction": float(np.mean(crop.valid_mask)),
                "frontFacingFraction": float(np.mean(crop.front_facing_mask)),
                "grazingFraction": float(np.mean(crop.grazing_mask)),
                "affineConversionError": affine_error,
                "cameraDistanceM": candidate["distanceM"],
                "frontFacingCosineAtMidpoint": candidate["facingCosine"],
                "frontFacingCosinesAtCorners": candidate["cornerFacingCosines"],
                "screenVisibilityGeometryOnly": candidate["screenVisibilityGeometryOnly"],
                "occlusionChecked": False,
                "alignmentApplied": camera.get("alignment") is not None,
                "alignmentReport": None if camera.get("alignment") is None else camera["alignment"].get("source"),
                "uncorrectedBaselineAccessible": True,
                "projectionBasisOrthogonalized": basis_orthogonalized,
            }
        )
    contact_sheet = output / "contact-sheet.png"
    _save_contact_sheet(crop_paths, contact_sheet)
    result = {
        "schema": "facade-photo-rectification-diagnostic/v1",
        "worldRegistrationValidated": False,
        "facadeMeasurementsGenerated": 0,
        "texturesPublished": False,
        "occlusionUnchecked": True,
        "evidenceScope": "unregistered photo rectification diagnostics only; no facade measurements or inferred openings",
        "inputs": {"imagery": str(imagery.resolve()), "normalizedInput": str(normalized_input.resolve()), "alignmentReport": None if alignment_report is None else str(alignment_report.resolve())},
        "selection": {"candidateFrontages": len(candidates), "selected": len(selected), "limit": limit, "recentDays": RECENT_DAYS, "pixelsPerM": pixels_per_m, "pairedIndependentViews": paired_views, "facingCosineMidpointMinimum": 0.5, "facingCosineCornerMinimum": 0.25, "lineOfSightGeometryOnly": True},
        "alignment": {"appliedClusters": len(alignments), "diagnosticOnly": True, "uncorrectedBaselineAccessible": True},
        "contactSheet": str(contact_sheet.resolve()) if contact_paths_exist(contact_sheet) else None,
        "crops": crop_records,
        "failures": failures,
        "limitations": [
            "Provider pose consistency is not world registration validation.",
            "No source occlusion mask was supplied; occlusionUnchecked remains true for every crop.",
            "Crop height is transparent unknown when no supplied height evidence exists; the bounded diagnostic extent is not a measurement.",
            "No facade openings, materials, textures, or building-specific visual heuristics are inferred.",
        ],
    }
    (output / "provenance.json").write_text(json.dumps(result, indent=2))
    return result


def contact_paths_exist(path: Path) -> bool:
    return path.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagery", type=Path, required=True)
    parser.add_argument("--normalized-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument('--pixels-per-m',type=float,default=PIXELS_PER_M)
    parser.add_argument("--paired-views", action="store_true", help="select two independent captures per identical metric frontage")
    parser.add_argument("--alignment-report", type=Path, help="optional accepted cluster alignment report; applied diagnostically only")
    args = parser.parse_args()
    result = rectify(args.imagery, args.normalized_input, args.output, args.limit, args.alignment_report,args.pixels_per_m,args.paired_views)
    print(json.dumps({"selected": result["selection"]["selected"], "crops": len(result["crops"]), "failures": len(result["failures"]), "contactSheet": result["contactSheet"]}, sort_keys=True))


if __name__ == "__main__":
    main()
