"""Compile evidence-approved, measured planar wall details.

This is a deliberately small connection between the evidence gate and
``opening_mesh``.  It accepts no identity, location, style, or template
fields, and never creates an aperture from a width/height or a named type.

Input records use this schema (all polygons are local ``(horizontal,
vertical)`` coordinates in metres):

``{"evidence_approved": True, "wall_polygon": [...],
"wall_thickness_m": 0.30, "coordinate_frame": {"origin": [x,y,z],
"horizontal": [..], "inward": [..], "vertical": [..]},
"apertures": [{"kind": ..., "boundary_polygon": [...], ...}]}``

The three frame vectors are unit, mutually perpendicular world-space vectors;
the generated opening mesh uses ``horizontal`` as X, ``inward`` as Y and
``vertical`` as Z.  Optional aperture components are passed through only when
present and measured (mullions, leaves, and thresholds).  No image extraction
is performed here; callers must supply the evidence-approved measurements.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import numpy as np
import trimesh
from shapely.geometry import Polygon

from .evidence import contains_authored_override
from .geometry import extruded
from .opening_mesh import ApertureError, compile_wall
from .surface_coverage import SurfaceCoverageError, audit_mesh_projection


class DetailCompilerError(ValueError):
    """A detail record is not a complete evidence-approved wall measurement."""


_REQUIRED = ("wall_polygon", "wall_thickness_m", "coordinate_frame", "apertures")
_FRAME = ("origin", "horizontal", "inward", "vertical")


def _vector(value, name, size):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise DetailCompilerError(f"{name} must be a finite vector") from None
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise DetailCompilerError(f"{name} must be a finite vector of length {size}")
    return result


def _frame(value):
    if not isinstance(value, Mapping) or any(k not in value for k in _FRAME):
        raise DetailCompilerError("coordinate_frame requires origin, horizontal, inward, and vertical")
    origin = _vector(value["origin"], "coordinate_frame.origin", 3)
    axes = [_vector(value[k], f"coordinate_frame.{k}", 3)
            for k in ("horizontal", "inward", "vertical")]
    for key, axis in zip(("horizontal", "inward", "vertical"), axes):
        if not math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1e-6):
            raise DetailCompilerError(f"coordinate_frame.{key} must be unit length")
    if any(abs(float(np.dot(a, b))) > 1e-6 for i, a in enumerate(axes) for b in axes[i + 1:]):
        raise DetailCompilerError("coordinate_frame axes must be perpendicular")
    if float(np.linalg.det(np.column_stack(axes))) <= 0:
        raise DetailCompilerError("coordinate_frame must be right-handed")
    return origin, axes


def _worldize(compiled, frame):
    origin, (horizontal, inward, vertical) = frame
    basis = np.column_stack((horizontal, inward, vertical))
    for part in compiled["parts"]:
        mesh = part["mesh"]
        mesh.vertices = origin + mesh.vertices @ basis.T
        mesh._cache.clear()
    compiled["frame"] = "world: origin + X*horizontal + Y*inward + Z*vertical"
    return compiled


def compile_detail_records(records: Iterable[Mapping]) -> list[dict]:
    """Compile each accepted measured wall record into transformed mesh parts.

    ``evidence_approved`` must be exactly ``True``.  Records lacking required
    geometry are rejected instead of being filled from defaults.  Unresolved
    cases include non-planar walls, missing measurements, and unsupported
    aperture topology; these remain for a future measured compiler.
    """
    if isinstance(records, Mapping) or isinstance(records, (str, bytes)):
        raise DetailCompilerError("records must be an iterable of wall records")
    output = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DetailCompilerError(f"record {index} must be an object")
        if record.get("evidence_approved") is not True:
            raise DetailCompilerError(f"record {index} is not evidence-approved")
        if contains_authored_override(record):
            raise DetailCompilerError(f"record {index} contains authored identity or override")
        missing = [key for key in _REQUIRED if key not in record]
        if missing:
            raise DetailCompilerError(f"record {index} missing required fields: {', '.join(missing)}")
        if not isinstance(record["apertures"], (list, tuple)):
            raise DetailCompilerError(f"record {index} apertures must be a list")
        frame = _frame(record["coordinate_frame"])
        for ap_index, aperture in enumerate(record["apertures"]):
            if not isinstance(aperture, Mapping) or "kind" not in aperture or "boundary_polygon" not in aperture:
                raise DetailCompilerError(f"record {index} aperture {ap_index} missing kind or boundary_polygon")
        try:
            compiled = compile_wall(record["wall_polygon"], record["apertures"], record["wall_thickness_m"])
        except ApertureError as exc:
            raise DetailCompilerError(f"record {index}: {exc}") from exc
        output.append(_worldize(compiled, frame))
    return output


compile_details = compile_detail_records


def replace_measured_panel(proxy_mesh: trimesh.Trimesh, record: Mapping) -> dict:
    """Boolean-replace a measured wall panel in a closed proxy solid.

    The complete wall envelope is removed first, then the measured wall and
    opening parts are returned for attachment.  A wall with an open passage
    is refused until a full through-depth passage measurement is supported.
    Boolean or support failures are explicit errors; this helper never overlays
    a wall on an uncut proxy.
    """
    if not isinstance(proxy_mesh, trimesh.Trimesh) or not proxy_mesh.is_watertight or not proxy_mesh.is_volume:
        raise DetailCompilerError("proxy_mesh must be a closed positive-volume solid")
    if not isinstance(record, Mapping):
        raise DetailCompilerError("record must be an object")
    for aperture in record.get("apertures", []):
        if aperture.get("kind") == "open_passage":
            raise DetailCompilerError("open_passage replacement requires full passage depth measurement")
    compiled = compile_detail_records([record])[0]
    frame = _frame(record["coordinate_frame"])
    # Audit the measured front skin before touching the proxy.  Only wall-labelled
    # parts on the front plane count; frames, glazing, and depth cues are recessed
    # or separate evidence and must not hide a missing wall triangle.
    wall_parts = [part["mesh"] for part in compiled["parts"] if part["label"].startswith("wall")]
    try:
        wall_polygon = Polygon(record["wall_polygon"])
        openings = [{"polygon": Polygon(ap["boundary_polygon"]), "owned": True}
                    for ap in record.get("apertures", [])]
        origin, (horizontal, inward, vertical) = frame
        front_frame = {"origin": origin - inward * float(record["wall_thickness_m"]) / 2,
                       "horizontal": horizontal, "vertical": vertical, "normal": -inward}
        surface_check = audit_mesh_projection(
            trimesh.util.concatenate(wall_parts), wall_polygon, front_frame, openings,
            tolerance=1e-6)
    except (SurfaceCoverageError, ValueError, TypeError) as exc:
        raise DetailCompilerError(f"measured wall surface audit failed: {exc}") from exc
    floating = 1e-7
    if any(surface_check[key] > floating for key in ("missing_area", "overlap_area", "outside_area")):
        raise DetailCompilerError(
            "measured wall surface audit failed: "
            f"missing={surface_check['missing_area']:.9g}, "
            f"overlap={surface_check['overlap_area']:.9g}, "
            f"outside={surface_check['outside_area']:.9g}")
    from .opening_mesh import _solid
    t=float(record["wall_thickness_m"])
    envelope = _solid(Polygon(record["wall_polygon"]), -t/2, t/2, "replacement-envelope")
    _worldize({"parts": [{"mesh": envelope}]}, frame)
    try:
        common = trimesh.boolean.intersection([proxy_mesh, envelope], engine="manifold", check_volume=True)
        common_volume = float(common.volume) if common is not None else 0.0
        if common_volume <= 1e-8:
            raise DetailCompilerError("wall envelope does not intersect proxy solid")
        replacement = trimesh.boolean.difference([proxy_mesh, envelope], engine="manifold", check_volume=True)
    except DetailCompilerError:
        raise
    except Exception as exc:
        raise DetailCompilerError(f"proxy boolean replacement failed: {exc}") from exc
    if replacement is None or not replacement.is_watertight or not replacement.is_volume:
        raise DetailCompilerError("proxy boolean replacement is not a closed positive solid")
    return {"proxy": replacement, "wall_parts": compiled["parts"], "checks": compiled["checks"],
            "surface_coverage": surface_check}
