#!/usr/bin/env python3
"""Small, evidence-only probe for Mapillary per-capture mesh protobufs.

The format follows the public MapillaryJS ``readMeshPbf`` implementation:
field 1 is repeated float vertex data and field 2 is repeated varint face
indices.  This module deliberately does not assign the camera-frame mesh to a
world coordinate system or alter any imagery/rendering inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
import sys
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Iterable


MAX_VALUES = 50_000_000


def _records_from_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        # Entity files are normally one object; accepting common list wrappers
        # keeps this reader useful for exported batches without assumptions.
        if isinstance(value.get("entities"), list):
            yield from (x for x in value["entities"] if isinstance(x, dict))
        else:
            yield value
    elif isinstance(value, list):
        yield from (x for x in value if isinstance(x, dict))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    files = sorted(root.rglob("*.json")) if root.is_dir() else [root]
    out: list[dict[str, Any]] = []
    for filename in files:
        try:
            with filename.open("r", encoding="utf-8") as stream:
                out.extend(_records_from_json(json.load(stream)))
        except (OSError, ValueError, TypeError):
            continue
    return out


def _position(record: dict[str, Any]) -> tuple[float, float] | None:
    geometry = record.get("computed_geometry") or record.get("geometry")
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    try:
        lon, lat = float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    return lon, lat


def _mesh_url(record: dict[str, Any]) -> str | None:
    mesh = record.get("mesh")
    url = mesh.get("url") if isinstance(mesh, dict) else None
    return url if isinstance(url, str) and url.startswith(("https://", "http://")) else None


def select_spatial(records: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Select far-apart valid camera captures deterministically.

    Candidate order follows camera coordinates and capture time.  Farthest-point sampling then spreads
    the small probe over the available camera footprint without location- or
    capture-specific rules.
    """
    candidates = [r for r in records if _position(r) is not None and _mesh_url(r)]
    candidates.sort(key=lambda r: (_position(r), r.get("captured_at", 0)))
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return candidates
    mean_lat = sum(_position(r)[1] for r in candidates) / len(candidates)  # type: ignore[index]
    scale = max(0.1, math.cos(math.radians(mean_lat)))

    def xy(record: dict[str, Any]) -> tuple[float, float]:
        lon, lat = _position(record)  # type: ignore[misc]
        return lon * scale, lat

    # Start at one footprint extreme, then choose the point farthest from the
    # already-selected set.  Ties remain deterministic through sorted order.
    selected = [min(candidates, key=lambda r: (xy(r)[0], xy(r)[1], str(r.get("id", ""))))]
    while len(selected) < limit:
        remaining = [r for r in candidates if r not in selected]
        chosen = max(
            remaining,
            key=lambda r: (
                min((xy(r)[0] - xy(s)[0]) ** 2 + (xy(r)[1] - xy(s)[1]) ** 2 for s in selected),
                str(r.get("id", "")),
            ),
        )
        selected.append(chosen)
    return selected


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift <= 63:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def _read_packed_varints(payload: bytes, out: list[int]) -> None:
    offset = 0
    while offset < len(payload):
        value, offset = _read_varint(payload, offset)
        out.append(value)
        if len(out) > MAX_VALUES:
            raise ValueError("protobuf value limit exceeded")


def decode_mesh_pbf(data: bytes) -> dict[str, Any]:
    """Decode MapillaryJS mesh fields, accepting packed or unpacked repeats."""
    raw = data
    compression = "none"
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
        compression = "gzip"
    else:
        try:
            candidate = zlib.decompress(raw)
        except zlib.error:
            candidate = None
        if candidate is not None and candidate[:1] in (b"\x08", b"\x11", b"\x12"):
            raw = candidate
            compression = "zlib"

    vertices: list[float] = []
    faces: list[int] = []
    offset = 0
    while offset < len(raw):
        key, offset = _read_varint(raw, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = _read_varint(raw, offset)
            if field == 2:
                faces.append(value)
        elif wire == 1:
            if offset + 8 > len(raw):
                raise ValueError("truncated fixed64 protobuf field")
            offset += 8
        elif wire == 2:
            length, offset = _read_varint(raw, offset)
            end = offset + length
            if end > len(raw):
                raise ValueError("truncated protobuf bytes field")
            payload = raw[offset:end]
            if field == 1:
                if len(payload) % 4:
                    raise ValueError("packed vertex field is not float aligned")
                vertices.extend(struct.unpack("<%df" % (len(payload) // 4), payload))
            elif field == 2:
                _read_packed_varints(payload, faces)
            offset = end
        elif wire == 5:
            if offset + 4 > len(raw):
                raise ValueError("truncated fixed32 protobuf field")
            if field == 1:
                vertices.append(struct.unpack_from("<f", raw, offset)[0])
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if len(vertices) > MAX_VALUES or len(faces) > MAX_VALUES:
            raise ValueError("protobuf value limit exceeded")

    if len(vertices) % 3:
        raise ValueError("vertex field is not divisible by three")
    if len(faces) % 3:
        raise ValueError("face field is not divisible by three")
    vertex_count = len(vertices) // 3
    invalid_faces = sum(index >= vertex_count for index in faces)
    if vertex_count == 0 or not faces:
        raise ValueError("mesh has no vertices or faces")

    points = [vertices[i : i + 3] for i in range(0, len(vertices), 3)]
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    radial = [math.sqrt(sum(value * value for value in point)) for point in points]
    edge_lengths: list[float] = []
    for offset in range(0, len(faces), 3):
        a, b, c = faces[offset : offset + 3]
        if max(a, b, c) >= vertex_count:
            continue
        for left, right in ((a, b), (b, c), (c, a)):
            edge_lengths.append(
                math.sqrt(sum((points[left][axis] - points[right][axis]) ** 2 for axis in range(3)))
            )
    edge_lengths.sort()

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, int(round(fraction * (len(values) - 1))))
        return values[index]

    return {
        "compression": compression,
        "vertexCount": vertex_count,
        "triangleCount": len(faces) // 3,
        "invalidFaceIndexCount": invalid_faces,
        "axisExtentsMeters": {"min": mins, "max": maxs},
        "radialDepthMeters": {"min": min(radial), "max": max(radial)},
        "triangleEdgeLengthMeters": {
            "median": percentile(edge_lengths, 0.50),
            "p90": percentile(edge_lengths, 0.90),
            "p95": percentile(edge_lengths, 0.95),
            "zeroLengthEdgeCount": sum(length == 0.0 for length in edge_lengths),
        },
        # The protobuf contains coordinates, not a depth-map sampling pitch.
        "depthResolutionMeters": None,
        "depthResolutionStatus": "not_encoded_in_mesh_pbf",
    }


def _download(url: str, destination: Path, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mapillary-mesh-probe/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    destination.write_bytes(data)
    return data


def probe(entities: str | Path, output: str | Path, limit: int = 3, timeout: float = 60.0) -> dict[str, Any]:
    records = load_records(entities)
    available = [r for r in records if _position(r) is not None and _mesh_url(r)]
    selected = select_spatial(records, limit)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        destination = out_dir / f"mesh-{index:02d}.pbf"
        item: dict[str, Any] = {
            "entityId": record.get("id"),
            "meshId": (record.get("mesh") or {}).get("id"),
            "downloadFile": destination.name,
            "cameraPosition": _position(record),
        }
        try:
            data = _download(_mesh_url(record) or "", destination, timeout)
            item["downloadedBytes"] = len(data)
            item["sha256"] = hashlib.sha256(data).hexdigest()
            try:
                item.update(decode_mesh_pbf(data))
                item["decodeStatus"] = "decoded"
            except Exception as exc:  # report type only; never echo signed URLs
                item["decodeStatus"] = "decode_failed"
                item["decodeErrorType"] = type(exc).__name__
        except Exception as exc:
            item["decodeStatus"] = "download_failed"
            item["downloadErrorType"] = type(exc).__name__
        results.append(item)

    decoded = sum(x.get("decodeStatus") == "decoded" for x in results)
    report = {
        "schema": "mapillary-mesh-probe-v12",
        "entitiesPath": str(entities),
        "entityRecordCount": len(records),
        "recordsWithCameraPositionAndMesh": len(available),
        "selectedCount": len(selected),
        "selection": "deterministic farthest-point sampling over camera positions",
        "results": results,
        "interpretation": {
            "officialFormat": "MapillaryJS readMeshPbf: field 1 repeated float vertices, field 2 repeated varint triangle indices",
            "cameraFrame": "Meshes are documented in the corresponding camera reference frame; no world transform is inferred here.",
            "approximateCameraDepth": decoded > 0,
            "approximateOcclusionCandidate": decoded > 0,
            "worldRegistration": False,
            "viewerTransitionOnly": "undetermined from mesh bytes alone",
            "photoOrFacadeAccuracy": "unverified; mesh geometry alone does not establish image-to-surface accuracy",
            "depthResolution": "not available because the protobuf carries mesh coordinates, not a depth-map sampling pitch",
            "scope": "diagnostic evidence only; no admission, validation, renderer, or facade output is changed",
        },
        "sources": [
            "https://mapillary.github.io/mapillary-js/docs/extension/procedural-data-provider/",
            "https://github.com/mapillary/mapillary-js/blob/0ab0a161/src/api/Common.ts",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", required=True, help="entity JSON file or directory")
    parser.add_argument("--output", required=True, help="directory for raw probes and report")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    report = probe(args.entities, args.output, args.limit, args.timeout)
    print(json.dumps({"selectedCount": report["selectedCount"], "decodedCount": sum(x.get("decodeStatus") == "decoded" for x in report["results"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
