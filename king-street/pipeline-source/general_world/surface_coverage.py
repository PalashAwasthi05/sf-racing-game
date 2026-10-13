"""Coverage contracts for replacing measured planar building surfaces.

All polygons are two-dimensional coordinates in one declared surface frame.  No
gap is widened or filled by tolerance: tolerances are used only for mesh plane
classification.
"""
from itertools import combinations
import math

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


class SurfaceCoverageError(ValueError):
    pass


def _polygons(value, label):
    if isinstance(value, (Polygon, MultiPolygon)):
        values = [value] if isinstance(value, Polygon) else list(value.geoms)
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise SurfaceCoverageError(f"{label} must contain polygons") from exc
    out = []
    for p in values:
        if not isinstance(p, Polygon) or p.is_empty or not p.is_valid:
            raise SurfaceCoverageError(f"{label} contains unknown or invalid geometry")
        if not math.isfinite(p.area) or p.area <= 0:
            raise SurfaceCoverageError(f"{label} contains degenerate geometry")
        out.append(p)
    return out


def _union(values, label):
    items = _polygons(values, label)
    return unary_union(items) if items else Polygon()


def _parts(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [g for g in geometry.geoms if isinstance(g, Polygon) and g.area > 0]
    # Boundary-only contacts are not surface overlap.
    if getattr(geometry, "area", 0) == 0:
        return []
    raise SurfaceCoverageError("operation produced unsupported geometry")


def _opening_polygons(openings):
    """Parse explicit opening records and enforce ownership at the API boundary."""
    values = []
    for item in openings or ():
        if not isinstance(item, dict) or item.get("owned") is not True:
            raise SurfaceCoverageError("intentional opening must be an explicitly owned record")
        item = item.get("polygon")
        values.append(item)
    return _polygons(values, "intentional_openings")


def audit_surface_coverage(expected_surface, emitted_surfaces, intentional_openings=()):
    """Compare emitted polygons to an expected surface minus owned openings."""
    expected = _union(expected_surface, "expected_surface")
    emitted = _union(emitted_surfaces, "emitted_surfaces")
    openings = _union(_opening_polygons(intentional_openings), "intentional_openings") if intentional_openings else Polygon()
    if not openings.is_empty and not openings.difference(expected).is_empty:
        raise SurfaceCoverageError("intentional opening is outside expected surface")
    target = expected.difference(openings)
    missing = target.difference(emitted)
    outside = emitted.difference(expected)
    overlaps = []
    source = _polygons(emitted_surfaces, "emitted_surfaces")
    for left, right in combinations(source, 2):
        common = left.intersection(right)
        overlaps.extend(_parts(common))
    overlap_union = unary_union(overlaps) if overlaps else Polygon()
    return {
        "expected": expected,
        "target": target,
        "missing": _parts(missing),
        "overlap": _parts(overlap_union),
        "outside": _parts(outside),
        "missing_area": float(missing.area),
        "overlap_area": float(overlap_union.area),
        "outside_area": float(outside.area),
    }


def replacement_transaction(host_surface, emitted_surfaces, intentional_openings=()):
    """Return exact replacement and the host remainder for an atomic update.

    Only emitted geometry inside the host is replaceable.  Missing host area is
    retained, and openings are informational exclusions from the coverage target;
    this helper never invents a fill for them.
    """
    host = _union(host_surface, "host_surface")
    audit = audit_surface_coverage(host, emitted_surfaces, intentional_openings)
    emitted = _union(emitted_surfaces, "emitted_surfaces")
    replaceable = emitted.intersection(host)
    owned_openings = _union(_opening_polygons(intentional_openings), "intentional_openings") if intentional_openings else Polygon()
    cut_domain = replaceable.union(owned_openings)
    retained = host.difference(cut_domain)
    return {
        "cut_domain": cut_domain,
        "replaceable_union": replaceable,
        "retained_host_remainder": retained,
        "coverage": audit,
    }


def audit_mesh_projection(mesh, expected_surface, frame, intentional_openings=(), tolerance=1e-3,
                          normal_alignment=0.999):
    """Audit front-facing triangles that lie on a declared planar frame.

    Triangles at another depth or facing backwards are rejected from the emitted
    union and counted, preventing unrelated geometry from hiding a gap.
    """
    try:
        origin = np.asarray(frame["origin"], dtype=float).copy()
        horizontal = np.asarray(frame["horizontal"], dtype=float).copy()
        vertical = np.asarray(frame["vertical"], dtype=float).copy()
        normal = np.asarray(frame["normal"], dtype=float).copy()
    except (KeyError, TypeError, ValueError) as exc:
        raise SurfaceCoverageError("frame requires origin, horizontal, vertical, normal") from exc
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (origin, horizontal, vertical, normal)):
        raise SurfaceCoverageError("frame vectors must be finite 3D values")
    norms = [np.linalg.norm(v) for v in (horizontal, vertical, normal)]
    if any(n <= 1e-12 for n in norms):
        raise SurfaceCoverageError("frame vectors must be nonzero")
    if any(abs(n - 1.0) > 1e-6 for n in norms):
        raise SurfaceCoverageError("frame vectors must be unit length")
    horizontal /= norms[0]; vertical /= norms[1]; normal /= norms[2]
    if (abs(horizontal @ vertical) > 1e-6 or abs(horizontal @ normal) > 1e-6 or abs(vertical @ normal) > 1e-6
            or not np.allclose(np.cross(horizontal, vertical), normal, atol=1e-6)):
        raise SurfaceCoverageError("frame vectors must be orthonormal with horizontal x vertical = normal")
    triangles = np.asarray(mesh.triangles, dtype=float)
    faces = np.asarray(mesh.face_normals, dtype=float)
    emitted = []
    rejected_depth = rejected_normal = 0
    for tri, face_normal in zip(triangles, faces):
        if np.max(np.abs((tri - origin) @ normal)) > tolerance:
            rejected_depth += 1
            continue
        if float(face_normal @ normal) < normal_alignment:
            rejected_normal += 1
            continue
        points = np.c_[(tri - origin) @ horizontal, (tri - origin) @ vertical]
        poly = Polygon(points)
        if poly.is_valid and poly.area > 0:
            emitted.append(poly)
    result = audit_surface_coverage(expected_surface, emitted, intentional_openings)
    result.update({"accepted_triangles": len(emitted), "rejected_depth_triangles": rejected_depth,
                   "rejected_normal_triangles": rejected_normal})
    return result
