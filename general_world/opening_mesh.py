"""Meshing of measured wall apertures.

This module deliberately consumes geometry and measurements only.  It does not
infer openings, choose a facade style, or add unmeasured architectural parts.
Coordinates supplied to this API are (horizontal, vertical) metres; returned
meshes use (X horizontal, Y depth, Z vertical).
"""
from __future__ import annotations

from typing import Any, Iterable
import numpy as np
import trimesh
from shapely.geometry import Polygon, LineString, MultiLineString, shape
from shapely.ops import unary_union

from .geometry import extruded


class ApertureError(ValueError):
    """Measured aperture cannot be represented without silently changing it."""


def _poly(value: Any) -> Polygon:
    if isinstance(value, Polygon):
        return value
    if isinstance(value, dict):
        value = shape(value)
    return Polygon(value)


def _solid(poly: Polygon, y0: float, y1: float, label: str) -> trimesh.Trimesh:
    """Use the project's manifold extrusion, then map its extrusion axis to Y."""
    meshes = extruded(poly, y0, y1)
    if len(meshes) != 1:
        raise ApertureError(f"{label}: disconnected or empty solid")
    m = meshes[0]
    # extruded makes (horizontal, vertical, depth); document frame is (X,Y,Z).
    m.vertices = m.vertices[:, [0, 2, 1]]
    m._cache.clear()
    # The odd coordinate permutation reverses handedness; restore outward winding.
    m.invert()
    m._cache.clear()
    m.metadata["label"] = label
    if not m.is_watertight or not m.is_volume:
        raise ApertureError(f"{label}: non-manifold generated solid")
    return m


def _add(parts: list[dict], poly: Polygon, y0: float, y1: float, label: str,
         aperture_index: int | None = None) -> None:
    if poly.is_empty or poly.area <= 1e-8:
        return
    geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    for n, piece in enumerate(geoms):
        if piece.area <= 1e-8:
            continue
        mesh = _solid(piece, y0, y1, label if len(geoms) == 1 else f"{label}_{n}")
        parts.append({"label": label if len(geoms) == 1 else f"{label}_{n}",
                      "mesh": mesh, "aperture_index": aperture_index})


def _segments(value: Iterable) -> Iterable[LineString]:
    for pair in value or []:
        line = LineString(pair)
        if not line.is_empty and line.length > 1e-8:
            yield line


def compile_wall(wall_polygon: Any, apertures: Iterable[dict], wall_thickness_m: float) -> dict:
    """Compile a measured wall and its measured openings into closed mesh parts.

    ``wall_polygon`` and each aperture ``boundary_polygon`` use (X,Z).  An
    aperture boundary is the actual hole, rather than a bounding rectangle.
    Each aperture record may contain ``mullion_segments`` (pairs of (X,Z)),
    ``leaf_polygon`` and ``threshold_polygon``.  The latter two are emitted
    only when explicitly measured.  The result contains ``parts`` and explicit
    area/topology checks for downstream validation.
    """
    t = float(wall_thickness_m)
    if t <= 0:
        raise ValueError("wall_thickness_m must be positive")
    wall = _poly(wall_polygon)
    if wall.is_empty or not wall.is_valid or wall.area <= 1e-8:
        raise ApertureError("wall_polygon is degenerate or invalid")
    records = list(apertures or [])
    holes: list[Polygon] = []
    for i, rec in enumerate(records):
        if rec.get("kind") not in {"window", "door", "open_passage"}:
            raise ApertureError(f"aperture {i}: kind must be window, door, or open_passage")
        p = _poly(rec.get("boundary_polygon"))
        if p.is_empty or not p.is_valid or p.area <= 1e-8:
            raise ApertureError(f"aperture {i}: degenerate or invalid boundary")
        if not wall.covers(p):
            raise ApertureError(f"aperture {i}: boundary lies outside wall")
        if any(p.intersection(q).area > 1e-8 for q in holes):
            raise ApertureError(f"aperture {i}: overlaps another aperture")
        holes.append(p)

    opening_union = unary_union(holes) if holes else None
    wall_region = wall.difference(opening_union) if opening_union is not None else wall
    parts: list[dict] = []
    _add(parts, wall_region, -t / 2, t / 2, "wall")
    aperture_reports = []
    for i, rec in enumerate(records):
        aperture = holes[i]
        width = float(rec.get("frame_width_m", 0.0) or 0.0)
        depth = float(rec.get("frame_depth_m", t) or t)
        recess = float(rec.get("recess_m", 0.0) or 0.0)
        if width < 0 or depth <= 0:
            raise ApertureError(f"aperture {i}: invalid frame measurement")
        inner = aperture.buffer(-width, join_style=2) if width else aperture
        if inner.is_empty or inner.area <= 1e-8:
            raise ApertureError(f"aperture {i}: frame buffer collapses opening")
        ring = aperture.difference(inner)
        y0 = -t / 2 + recess
        y1 = y0 + depth
        _add(parts, ring, y0, y1, f"aperture_{i}_frame", i)
        if rec["kind"] == "window":
            _add(parts, inner, y0 + depth / 2 - 0.001, y0 + depth / 2 + 0.001,
                 f"aperture_{i}_glazing", i)
            glazing = inner
        else:
            glazing = inner
        for j, line in enumerate(_segments(rec.get("mullion_segments"))):
            clipped = line.intersection(glazing)
            for k, piece in enumerate(getattr(clipped, "geoms", [clipped])):
                if piece.geom_type != "LineString" or piece.length <= 1e-8:
                    continue
                bar_width = float(rec.get("mullion_width_m", 0.03))
                bar = piece.buffer(bar_width / 2, cap_style=2, join_style=2).intersection(glazing)
                _add(parts, bar,
                     y0 + depth / 2 - bar_width / 2, y0 + depth / 2 + bar_width / 2,
                     f"aperture_{i}_mullion_{j}_{k}", i)
        leaf = rec.get("leaf_polygon", rec.get("measured_leaf_polygon"))
        if leaf is not None:
            lp = _poly(leaf)
            if not inner.covers(lp):
                raise ApertureError(f"aperture {i}: measured leaf lies outside interior")
            leaf_depth = float(rec.get("leaf_depth_m", 0.05))
            _add(parts, lp, y0 + depth / 2 - leaf_depth / 2,
                 y0 + depth / 2 + leaf_depth / 2, f"aperture_{i}_door_leaf", i)
        threshold = rec.get("threshold_polygon")
        if threshold is not None:
            tp = _poly(threshold)
            _add(parts, tp, -t / 2, t / 2, f"aperture_{i}_threshold", i)
        aperture_reports.append({"index": i, "kind": rec["kind"],
                                 "hole_area_m2": aperture.area,
                                 "interior_area_m2": inner.area})
    checks = {"wall_area_m2": wall.area, "opening_area_m2": sum(p.area for p in holes),
              "remaining_wall_area_m2": wall_region.area,
              "parts_watertight": all(x["mesh"].is_watertight for x in parts),
              "parts_positive_volume": all(x["mesh"].is_volume for x in parts),
              "closed_manifold": all(x["mesh"].is_watertight and x["mesh"].is_volume for x in parts)}
    return {"parts": parts, "apertures": aperture_reports, "checks": checks,
            "frame": "Xhorizontal,Ydepth,Zvertical", "inference": False}
