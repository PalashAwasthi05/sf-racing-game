"""Coarse, evidence-only inference of green bicycle pavement from cached panoramas.

This module intentionally does not make a geometric registration claim.  It projects
small ground samples from OSM bicycle ways into the cached equirectangular images and
only emits a style when two separated cameras see a spatially coherent green run.
The camera height is an explicit coarse assumption (2.5 m); it is not read from a
signed provider URL or inferred from image content.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from general_world.image_projection import angle_axis_matrix, camera_vectors_to_uv


CAMERA_HEIGHT_M = 2.5
MIN_RANGE_M = 5.0
MAX_RANGE_M = 30.0
MIN_CAMERA_SEPARATION_M = 3.0
MAX_IMAGE_WIDTH = 2048
GROUND_V_MIN = 0.50
GROUND_V_MAX = 0.70  # excludes the vehicle / mapping overlay below this band
SAMPLE_SPACING_M = 1.5


@dataclass(frozen=True)
class CycleWay:
    way_id: str
    node_ids: tuple[str, ...]
    coords: tuple[tuple[float, float], ...]  # (lon, lat)
    tags: Mapping[str, str]
    lane_like: bool


@dataclass(frozen=True)
class GroundSample:
    way_id: str
    chain_m: float
    lon: float
    lat: float
    offset_m: float


def _is_cycleway_tag(tags: Mapping[str, str]) -> bool:
    highway = tags.get("highway", "")
    if highway == "cycleway":
        return True
    for key, value in tags.items():
        if key == "cycleway" or key.startswith("cycleway:"):
            if value not in {"", "no", "none"}:
                return True
    return False


def parse_osm_cycle_ways(osm_path: str | os.PathLike[str]) -> list[CycleWay]:
    """Parse only bicycle-bearing LineString ways; no provider metadata is read."""

    root = ET.parse(osm_path).getroot()
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        try:
            nodes[node.attrib["id"]] = (float(node.attrib["lon"]), float(node.attrib["lat"]))
        except (KeyError, TypeError, ValueError):
            continue
    ways: list[CycleWay] = []
    for way in root.findall("way"):
        tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in way.findall("tag")}
        if not _is_cycleway_tag(tags):
            continue
        refs = tuple(nd.attrib.get("ref", "") for nd in way.findall("nd"))
        coords = tuple(nodes[ref] for ref in refs if ref in nodes)
        if len(coords) < 2:
            continue
        lane_like = tags.get("highway") != "cycleway" and any(
            key == "cycleway" or key.startswith("cycleway:") for key in tags
        )
        ways.append(CycleWay(str(way.attrib.get("id", "")), refs, coords, tags, lane_like))
    return ways


def _meters_per_degree(lat: float) -> tuple[float, float]:
    phi = math.radians(lat)
    return 111320.0 * math.cos(phi), 110540.0


def _enu(lon: float, lat: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    east_scale, north_scale = _meters_per_degree(origin_lat)
    return (lon - origin_lon) * east_scale, (lat - origin_lat) * north_scale


def _lonlat(east: float, north: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    east_scale, north_scale = _meters_per_degree(origin_lat)
    return origin_lon + east / east_scale, origin_lat + north / north_scale


def _polyline_samples(way: CycleWay) -> list[GroundSample]:
    """Densify each OSM way and add conservative side offsets for lane tags."""

    samples: list[GroundSample] = []
    chain = 0.0
    offsets = (0.0, -1.6, 1.6) if way.lane_like else (0.0,)
    for (lon0, lat0), (lon1, lat1) in zip(way.coords, way.coords[1:]):
        ex, ny = _enu(lon1, lat1, lon0, lat0)
        length = math.hypot(ex, ny)
        if length < 0.01:
            continue
        count = max(1, int(math.ceil(length / SAMPLE_SPACING_M)))
        for i in range(count):
            t = (i + 0.5) / count
            lon = lon0 + t * (lon1 - lon0)
            lat = lat0 + t * (lat1 - lat0)
            # Left/right offsets in local ENU.  A lane width is uncertain here,
            # so offset is only a search band and never a placed geometry.
            for offset in offsets:
                ox, oy = -ny / length * offset, ex / length * offset
                slon, slat = _lonlat(ox, oy, lon, lat) if offset else (lon, lat)
                samples.append(GroundSample(way.way_id, chain + t * length, slon, slat, offset))
        chain += length
    return samples


def project_ground_samples(
    samples: Sequence[GroundSample], image_meta: Mapping[str, Any]
) -> list[tuple[GroundSample, float, float, float]]:
    """Project ENU ground points to (sample, u, v, horizontal_range_m).

    ``computed_rotation`` is the provider's world-ENU-to-camera angle-axis pose.
    The world vector uses a ground plane 2.5 m below the camera.  Samples behind
    the horizon or in the lower overlay band are rejected.
    """

    geom = image_meta.get("computed_geometry") or image_meta.get("geometry")
    if not isinstance(geom, Mapping) or geom.get("type") != "Point":
        return []
    try:
        camera_lon, camera_lat = map(float, geom["coordinates"][:2])
        rotation = angle_axis_matrix(image_meta["computed_rotation"])
    except (KeyError, TypeError, ValueError, IndexError):
        return []
    vectors: list[list[float]] = []
    ranges: list[float] = []
    kept: list[GroundSample] = []
    for sample in samples:
        east, north = _enu(sample.lon, sample.lat, camera_lon, camera_lat)
        distance = math.hypot(east, north)
        if not MIN_RANGE_M <= distance <= MAX_RANGE_M:
            continue
        kept.append(sample)
        ranges.append(distance)
        vectors.append([east, north, -CAMERA_HEIGHT_M])
    if not vectors:
        return []
    try:
        u, v = camera_vectors_to_uv(np.einsum("ij,nj->ni", rotation, np.asarray(vectors)))
    except ValueError:
        return []
    result = []
    for sample, uu, vv, distance in zip(kept, u, v, ranges):
        if math.isfinite(float(uu)) and GROUND_V_MIN < float(vv) < GROUND_V_MAX:
            result.append((sample, float(uu) % 1.0, float(vv), distance))
    return result


def _patch(image: np.ndarray, u: float, v: float, radius: int = 2) -> np.ndarray | None:
    h, w = image.shape[:2]
    x = int(round(u * w)) % w
    y = int(round(v * (h - 1)))
    if y - radius < 0 or y + radius >= h:
        return None
    # Horizontal wrapping is valid for equirectangular images.
    xs = (np.arange(x - radius, x + radius + 1) % w).astype(int)
    return image[y - radius : y + radius + 1, xs, :3].astype(np.float32)


def green_patch_score(patch: np.ndarray) -> float:
    """Return the fraction of patch pixels with a muted/faded green signature."""

    if patch is None or patch.ndim != 3 or patch.shape[-1] < 3 or patch.size == 0:
        return 0.0
    rgb = np.asarray(patch, dtype=np.float32)
    r, g, b = np.moveaxis(rgb[..., :3], -1, 0)
    brightness = (r + g + b) / 3.0
    # Relative channel differences catch faded green while rejecting gray paint.
    green = (g >= 45.0) & (g > r + 6.0) & (g > b + 1.0) & ((g - r) / np.maximum(brightness, 1.0) > 0.045)
    # Paint is locally coherent; foliage tends to have mixed holes/edges.  This
    # is only a weak guard and is deliberately not treated as vegetation proof.
    green_fraction = float(np.mean(green))
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).reshape(-1)
    coherence = max(0.0, 1.0 - float(np.std(luminance)) / 65.0)
    return green_fraction * coherence


def classify_green_pattern(values: Sequence[bool]) -> str | None:
    """Classify only a spatially supported run as dashed or solid."""

    vals = [bool(x) for x in values]
    if len(vals) < 3 or not any(vals):
        return None
    runs: list[tuple[bool, int]] = []
    for value in vals:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    green_runs = [length for value, length in runs if value]
    positive = sum(green_runs)
    longest = max(green_runs)
    if len(green_runs) >= 2 and positive / len(vals) <= 0.78 and any(not value for value, _ in runs[1:-1]):
        return "dashed"
    if longest >= 3 and positive / len(vals) >= 0.52:
        return "solid"
    return None


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


def _camera_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    try:
        ga = a["computed_geometry"]["coordinates"]
        gb = b["computed_geometry"]["coordinates"]
        east, north = _enu(float(gb[0]), float(gb[1]), float(ga[0]), float(ga[1]))
        return math.hypot(east, north)
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0


def infer_cycle_paint(selection: Mapping[str, Any], osm_path: str | os.PathLike[str]) -> dict[str, Any]:
    ways = parse_osm_cycle_ways(osm_path)
    samples_by_way = {way.way_id: _polyline_samples(way) for way in ways}
    images = selection.get("images", []) if isinstance(selection, Mapping) else []
    observations: dict[str, dict[str, dict[float, bool]]] = {}
    image_meta_by_id: dict[str, Mapping[str, Any]] = {}
    for meta in images:
        if not isinstance(meta, Mapping):
            continue
        image_id = str(meta.get("id", ""))
        image_path = meta.get("local_image")
        if not image_id or not image_path:
            continue
        image = _load_image(str(image_path))
        if image is None:
            continue
        image_meta_by_id[image_id] = meta
        # Keep only this image's resized array in memory.
        for way in ways:
            projected = project_ground_samples(samples_by_way[way.way_id], meta)
            if not projected:
                continue
            by_chain = observations.setdefault(way.way_id, {}).setdefault(image_id, {})
            for sample, u, v, _distance in projected:
                patch = _patch(image, u, v)
                score = green_patch_score(patch)
                # A modest majority is needed to reject isolated green pixels.
                by_chain[round(sample.chain_m, 2)] = score >= 0.52
    styles: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for way in ways:
        per_image = observations.get(way.way_id, {})
        image_runs: dict[str, str] = {}
        green_images: list[str] = []
        for image_id, chain_values in per_image.items():
            ordered = [value for _, value in sorted(chain_values.items())]
            pattern = classify_green_pattern(ordered)
            if pattern:
                image_runs[image_id] = pattern
            if sum(ordered) >= 2:
                green_images.append(image_id)
        # Agreement requires two distinct camera locations at least 3 m apart,
        # and the same style in both views.  This is a consistency check only.
        agreeing: list[str] = []
        pattern: str | None = None
        for first_index, first in enumerate(green_images):
            for second in green_images[first_index + 1 :]:
                if _camera_distance(image_meta_by_id[first], image_meta_by_id[second]) < MIN_CAMERA_SEPARATION_M:
                    continue
                first_pattern = image_runs.get(first)
                second_pattern = image_runs.get(second)
                if first_pattern and first_pattern == second_pattern:
                    agreeing = [first, second]
                    pattern = first_pattern
                    break
            if pattern:
                break
        if pattern:
            styles.append(
                {
                    "wayId": way.way_id,
                    "colour": "greenpaint",
                    "pattern": pattern,
                    "evidenceImages": agreeing,
                    "status": "coarse-image-supported",
                    "positionUncertaintyM": 8.0,
                }
            )
        else:
            unresolved.append(way.way_id)
    return {
        "schema": "infer_cycle_paint_v21",
        "status": "coarse-image-supported" if styles else "unresolved",
        "paintStyles": styles,
        "unresolvedWayIds": unresolved,
        "limitations": [
            "2.5 m camera height and provider pose are coarse assumptions; no metric registration is claimed.",
            "Samples use projected ground rays and exclude v>=0.70 vehicle/branding overlay; foliage false positives remain unverified.",
            "A style is emitted only after same-pattern agreement from two cameras separated by at least 3 m.",
        ],
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="selection JSON with local cached image metadata")
    parser.add_argument("--osm", required=True, help="OSM XML input")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args(argv)
    with open(args.selection, encoding="utf-8") as handle:
        selection = json.load(handle)
    result = infer_cycle_paint(selection, args.osm)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
