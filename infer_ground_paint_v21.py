"""Bounded pixel-component experiment for green bicycle pavement evidence.

The output is deliberately diagnostic: local ENU patch polygons are retained only
when two separated panoramas produce nearby connected green components.  The
experiment never extends a patch along a way or bridges image gaps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage
from pyproj import Transformer
from shapely.geometry import GeometryCollection, Point, box, mapping, shape
from shapely.ops import unary_union
from shapely import contains_xy

from general_world.image_projection import angle_axis_matrix


def _resolve_transformer(world: Mapping[str, Any], world_crs: str | None) -> tuple[Transformer, str]:
    metadata_crs = world.get("coordinateSystem", {}).get("crs") if isinstance(world.get("coordinateSystem"), Mapping) else None
    crs = world_crs or metadata_crs
    if not crs:
        raise ValueError("world CRS is required via --world-crs or world.coordinateSystem.crs")
    return Transformer.from_crs(4326, crs, always_xy=True), str(crs)


CAMERA_HEIGHT_M = 2.5
MAX_IMAGE_WIDTH = 1024
GROUND_V_MIN, GROUND_V_MAX = 0.51, 0.68
MIN_RANGE_M, MAX_RANGE_M = 5.0, 30.0
CELL_M = 0.5
MIN_PATCH_AREA_M2 = 0.6
MIN_VIEW_SEPARATION_M = 3.0
MAX_AGREEMENT_M = 0.75


@dataclass(frozen=True)
class Component:
    image_id: str
    captured_at: Any
    camera_xy: tuple[float, float]
    cells: frozenset[tuple[int, int]]
    geometry: Any

    @property
    def centroid_xy(self) -> tuple[float, float]:
        c = self.geometry.centroid
        return float(c.x), float(c.y)


def _meters_per_degree(lat: float) -> tuple[float, float]:
    phi = math.radians(lat)
    return 111320.0 * math.cos(phi), 110540.0


def lonlat_to_enu(
    lon: float, lat: float, center_lon: float, center_lat: float, transformer: Transformer
) -> tuple[float, float]:
    center_x, center_y = transformer.transform(center_lon, center_lat)
    x, y = transformer.transform(lon, lat)
    return float(x - center_x), float(y - center_y)


def local_utm_jacobian(lon: float, lat: float, transformer: Transformer) -> np.ndarray:
    """Map tangent ENU metres at a camera into the local projected CRS."""

    x0, y0 = transformer.transform(lon, lat)
    east_scale, north_scale = _meters_per_degree(lat)
    dlon, dlat = 1.0 / east_scale, 1.0 / north_scale
    xe, ye = transformer.transform(lon + dlon, lat)
    xn, yn = transformer.transform(lon, lat + dlat)
    return np.array([[xe - x0, xn - x0], [ye - y0, yn - y0]], dtype=np.float64)


def green_mask(image: np.ndarray, *, v_min: float = GROUND_V_MIN, v_max: float = GROUND_V_MAX) -> np.ndarray:
    """Muted green threshold with provider lower-overlay exclusion."""

    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros(image.shape[:2], dtype=bool)
    rgb = image[..., :3].astype(np.float32)
    r, g, b = np.moveaxis(rgb, -1, 0)
    brightness = (r + g + b) / 3.0
    mask = (g > r + 6.0) & (g > b + 1.0) & (brightness > 45.0)
    h = image.shape[0]
    yy = (np.arange(h) + 0.5) / h
    mask &= (yy[:, None] >= v_min) & (yy[:, None] <= v_max)
    return mask


def _load_image(path: str | os.PathLike[str]) -> np.ndarray | None:
    try:
        with Image.open(path) as source:
            source = source.convert("RGB")
            if source.width > MAX_IMAGE_WIDTH:
                height = max(1, int(round(source.height * MAX_IMAGE_WIDTH / source.width)))
                source = source.resize((MAX_IMAGE_WIDTH, height), Image.Resampling.BILINEAR)
            return np.asarray(source)
    except (OSError, ValueError):
        return None


def camera_to_ground_xy(
    u: np.ndarray,
    v: np.ndarray,
    rotation_angle_axis: Any,
    camera_xy: tuple[float, float],
    local_jacobian: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect spherical camera rays with the plane 2.5 m below camera.

    The ray convention mirrors ``camera_vectors_to_uv``: ``u`` wraps around
    longitude and ``v`` is the spherical vertical coordinate.  ``R.T`` converts
    camera rays back to the provider's world ENU frame used by street geometry.
    """

    theta = 2.0 * math.pi * (np.asarray(u, dtype=np.float64) - 0.5)
    phi = math.pi * (np.asarray(v, dtype=np.float64) - 0.5)
    horizontal = np.cos(phi)
    camera_rays = np.stack(
        [np.sin(theta) * horizontal, np.sin(phi), np.cos(theta) * horizontal], axis=-1
    )
    # image_projection uses camera_column = R @ world_column.  With row-wise
    # arrays the inverse is therefore camera_row @ R.
    world_rays = camera_rays @ angle_axis_matrix(rotation_angle_axis)
    dz = world_rays[..., 2]
    valid = np.isfinite(dz) & (dz < -1e-6)
    distance_scale = np.full(dz.shape, np.nan, dtype=np.float64)
    distance_scale[valid] = -CAMERA_HEIGHT_M / dz[valid]
    east_enu = distance_scale * world_rays[..., 0]
    north_enu = distance_scale * world_rays[..., 1]
    jacobian = np.eye(2, dtype=np.float64) if local_jacobian is None else np.asarray(local_jacobian, dtype=np.float64)
    east = camera_xy[0] + jacobian[0, 0] * east_enu + jacobian[0, 1] * north_enu
    north = camera_xy[1] + jacobian[1, 0] * east_enu + jacobian[1, 1] * north_enu
    horizontal_range = np.hypot(east - camera_xy[0], north - camera_xy[1])
    valid &= np.isfinite(east) & np.isfinite(north)
    valid &= (horizontal_range >= MIN_RANGE_M) & (horizontal_range <= MAX_RANGE_M)
    return east, north, valid


def _cell_geometry(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> Any:
    return unary_union(
        [box(ix * CELL_M, iy * CELL_M, (ix + 1) * CELL_M, (iy + 1) * CELL_M) for ix, iy in cells]
    )


def _cell_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    result: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            x, y = stack.pop()
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    group.add(neighbour)
                    stack.append(neighbour)
        result.append(group)
    return result


def extract_components(
    image: np.ndarray,
    meta: Mapping[str, Any],
    allowed: Any,
    center_lon: float,
    center_lat: float,
    transformer: Transformer,
) -> list[Component]:
    """Extract connected metric-cell patches from one image only."""

    mask = green_mask(image)
    labels, count = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    geom = allowed.buffer(1e-8)
    try:
        camera_geom = meta["computed_geometry"]["coordinates"]
        camera_lon, camera_lat = float(camera_geom[0]), float(camera_geom[1])
        camera_xy = lonlat_to_enu(camera_lon, camera_lat, center_lon, center_lat, transformer)
        jacobian = local_utm_jacobian(camera_lon, camera_lat, transformer)
        rotation = meta["computed_rotation"]
        image_id = str(meta["id"])
    except (KeyError, TypeError, ValueError, IndexError):
        return []
    result: list[Component] = []
    for label in range(1, int(count) + 1):
        ys, xs = np.where(labels == label)
        if len(xs) < 3:
            continue
        u = (xs.astype(np.float64) + 0.5) / image.shape[1]
        v = (ys.astype(np.float64) + 0.5) / image.shape[0]
        east, north, valid = camera_to_ground_xy(u, v, rotation, camera_xy, jacobian)
        if not np.any(valid):
            continue
        xvals, yvals = east[valid], north[valid]
        inside = contains_xy(geom, xvals, yvals)
        cells = {(int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M))) for x, y in zip(xvals[inside], yvals[inside])}
        if not cells:
            continue
        # Quantization can split a thin image component.  Only connected cell
        # groups are eligible; this preserves gaps instead of repairing them.
        for group in _cell_components(cells):
            patch = _cell_geometry(group).intersection(allowed)
            if patch.is_empty or patch.area < MIN_PATCH_AREA_M2:
                continue
            result.append(
                Component(
                    image_id=image_id,
                    captured_at=meta.get("captured_at"),
                    camera_xy=camera_xy,
                    cells=frozenset(group),
                    geometry=patch,
                )
            )
    return result


def _camera_distance(a: Component, b: Component) -> float:
    return math.hypot(a.camera_xy[0] - b.camera_xy[0], a.camera_xy[1] - b.camera_xy[1])


def _patch_pair(a: Component, b: Component) -> tuple[Any, float] | None:
    if a.image_id == b.image_id or _camera_distance(a, b) < MIN_VIEW_SEPARATION_M:
        return None
    distance = math.hypot(a.centroid_xy[0] - b.centroid_xy[0], a.centroid_xy[1] - b.centroid_xy[1])
    if distance > MAX_AGREEMENT_M:
        return None
    # Keep one observed component's connected geometry.  A buffered intersection
    # would bridge gaps and imply sub-cell registration precision.
    return a.geometry, distance


def _street_allowed(street: Mapping[str, Any]) -> Any:
    def get(name: str) -> Any:
        value = street.get(name)
        return shape(value) if isinstance(value, Mapping) else GeometryCollection()

    road, cycle, sidewalk, crossings = (get(k) for k in ("road", "cycle", "sidewalk", "crossings"))
    forbidden = get("forbidden")
    building = get("building")
    if building.is_empty and not forbidden.is_empty:
        building = forbidden.difference(road.union(cycle).union(crossings))
    # Cycle buffers admit small projection error, while road/cycle retains the
    # model's street ownership.  Sidewalks/buildings remain hard exclusions.
    return cycle.buffer(2.0).intersection(road.union(cycle)).difference(sidewalk).difference(building).buffer(0)


def infer_ground_paint(
    selection: Mapping[str, Any], world: Mapping[str, Any], street: Mapping[str, Any], world_crs: str | None = None
) -> dict[str, Any]:
    transformer, resolved_crs = _resolve_transformer(world, world_crs)
    center = world.get("center", {})
    center_lon, center_lat = float(center["lon"]), float(center["lat"])
    allowed = _street_allowed(street)
    images = selection.get("images", []) if isinstance(selection, Mapping) else []
    components: list[Component] = []
    for meta in images:
        if not isinstance(meta, Mapping) or meta.get("is_pano") is not True or not meta.get("local_image"):
            continue
        image = _load_image(str(meta["local_image"]))
        if image is None:
            continue
        components.extend(extract_components(image, meta, allowed, center_lon, center_lat, transformer))

    accepted: list[dict[str, Any]] = []
    pairCandidates = 0
    rejectedSeparation = 0
    rejectedAgreement = 0
    for i, first in enumerate(components):
        for second in components[i + 1 :]:
            if first.image_id == second.image_id:
                continue
            pairCandidates += 1
            if _camera_distance(first, second) < MIN_VIEW_SEPARATION_M:
                rejectedSeparation += 1
                continue
            centroid_distance = math.hypot(
                first.centroid_xy[0] - second.centroid_xy[0],
                first.centroid_xy[1] - second.centroid_xy[1],
            )
            if centroid_distance > MAX_AGREEMENT_M:
                rejectedAgreement += 1
                continue
            pair = _patch_pair(first, second)
            if pair is None:
                continue
            patch, agreement = pair
            accepted.append(
                {
                    "geometry": mapping(patch),
                    "imageIds": sorted({first.image_id, second.image_id}),
                    "captureTimestamps": [first.captured_at, second.captured_at],
                    "agreementDistanceM": round(agreement, 3),
                    "projectionUncertaintyM": 1.25,
                    "areaM2": round(float(patch.area), 3),
                }
            )
    features = []
    for item in accepted:
        properties = {k: v for k, v in item.items() if k != "geometry"}
        properties["status"] = "coarse-pixel-component-supported"
        features.append({"type": "Feature", "geometry": item["geometry"], "properties": properties})
    return {
        "type": "FeatureCollection",
        "schema": "infer_ground_paint_v21",
        "worldCRS": resolved_crs,
        "coordinateSystem": "local ENU metres; origin is world.center",
        "features": features,
        "diagnostics": {
            "panoImagesRead": len({c.image_id for c in components}),
            "greenComponents": len(components),
            "acceptedPatches": len(features),
            "acceptedAreaM2": round(sum(float(x["areaM2"]) for x in accepted), 3),
            "pairCandidates": pairCandidates,
            "rejectedViewSeparation": rejectedSeparation,
            "rejectedLocalAgreement": rejectedAgreement,
            "threshold": "g>r+6, g>b+1, brightness>45; v=.51..68",
            "limits": [
                "2.5 m camera height and spherical pose are coarse assumptions.",
                "Components are clipped to cycle.buffer(2 m) and model road/cycle ownership, excluding sidewalk/building geometry when present.",
                "Two viewpoints must be at least 3 m apart and component centroids within 0.75 m; no way-run filling or gap bridging is performed.",
            ],
        },
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--street-space", required=True)
    parser.add_argument("--world-crs", default=None, help="projected CRS used by street-space geometry, e.g. EPSG:32610")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with open(args.selection, encoding="utf-8") as f:
        selection = json.load(f)
    with open(args.world, encoding="utf-8") as f:
        world = json.load(f)
    with open(args.street_space, encoding="utf-8") as f:
        street = json.load(f)
    output = infer_ground_paint(selection, world, street, args.world_crs)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
