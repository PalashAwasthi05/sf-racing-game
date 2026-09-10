"""Conservative recovery of small, measured gaps in planar roof coverage."""
import math
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union


class RoofCoverageError(ValueError):
    pass


def _polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for q in getattr(g, "geoms", []) for p in _polys(q)]


def _plane(tri):
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.linalg.norm(n) < 1e-9 or n[2] <= 1e-6:
        return None
    n /= np.linalg.norm(n)
    a, b = -n[0] / n[2], -n[1] / n[2]
    return np.array([a, b, tri[0, 2] - a * tri[0, 0] - b * tri[0, 1]])


def _plane_groups(meshes):
    groups = []
    for parent, mesh in enumerate(meshes):
        tri = np.asarray(mesh.triangles, float)
        for t in tri:
            c = _plane(t)
            if c is None:
                continue
            p = Polygon(t[:, :2])
            if p.area <= 1e-9:
                continue
            match = next((g for g in groups if np.linalg.norm(c[:2] - g["plane"][:2]) < 1e-5 and abs(c[2] - g["plane"][2]) < 1e-3), None)
            if match is None:
                match = {"plane": c, "footprints": [], "parent": parent}
                groups.append(match)
            match["footprints"].append(p)
    for g in groups:
        g["surface"] = unary_union(g["footprints"])
    return groups


def _cells(points, region, cell=1.0):
    if not len(points):
        return {}
    keys = np.floor(np.asarray(points)[:, :2] / cell).astype(int)
    out = {}
    for key, point in zip(map(tuple, keys), points):
        if region.covers(__import__("shapely").geometry.Point(*point[:2])):
            out.setdefault(key, []).append(point)
    return out


def recover_roof_coverage(roof_meshes, footprint, point_cloud, minimum_roof_z, max_cap_distance=4.0,
                          residual_m=.25, cell_m=1.0, thickness_m=.02):
    """Recover only point-supported gap components adjacent to an existing plane."""
    meshes = list(roof_meshes)
    if not meshes or not all(isinstance(m, trimesh.Trimesh) for m in meshes):
        raise RoofCoverageError("roof_meshes must contain trimesh meshes")
    if not isinstance(footprint, (Polygon, MultiPolygon)) or footprint.is_empty or not footprint.is_valid:
        raise RoofCoverageError("footprint must be a valid polygon")
    points = np.asarray(point_cloud, float)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        raise RoofCoverageError("point_cloud must be finite XYZ rows")
    groups = _plane_groups(meshes)
    observed = unary_union([g["surface"] for g in groups]) if groups else Polygon()
    expected = footprint
    missing = expected.difference(observed)
    report = {"expectedAreaM2": float(expected.area), "observedAreaM2": float(observed.intersection(expected).area),
              "gapAreaM2": float(missing.area), "supportedAreaM2": 0.0, "unresolvedAreaM2": float(missing.area),
              "components": [], "status": "no-overlap-existing" if observed.intersection(expected).is_empty else "unresolved"}
    patches = []
    for component in _polys(missing):
        candidates = [g for g in groups if g["surface"].distance(component) <= max_cap_distance and
                      g["surface"].boundary.distance(component) <= max_cap_distance]
        accepted = None
        for g in sorted(candidates, key=lambda x: x["surface"].distance(component)):
            plane = g["plane"]
            if float(np.min([plane[0] * x + plane[1] * y + plane[2] for x, y in component.exterior.coords])) <= minimum_roof_z:
                continue
            cloud = points[points[:, 2] >= minimum_roof_z]
            cloud = cloud[np.abs(cloud[:, 2] - (plane[0] * cloud[:, 0] + plane[1] * cloud[:, 1] + plane[2])) <= residual_m]
            cell_points = _cells(cloud, component, cell_m)
            support = unary_union([box(k[0] * cell_m, k[1] * cell_m, (k[0] + 1) * cell_m, (k[1] + 1) * cell_m) for k in cell_points]).intersection(component) if cell_points else Polygon()
            if len(cell_points) < 3 or support.is_empty:
                continue
            accepted = (g, support)
            break
        if accepted is None:
            report["components"].append({"areaM2": float(component.area), "status": "unsupported"})
            continue
        g, support = accepted
        # Never interpolate beyond the measured cell union; each emitted piece is
        # clipped to both the gap and the supported 1 m cells.
        for piece in _polys(support):
            if piece.area <= 1e-9:
                continue
            z = float(g["plane"][0] * piece.centroid.x + g["plane"][1] * piece.centroid.y + g["plane"][2])
            raw = trimesh.creation.extrude_polygon(piece, mid_plane := thickness_m, engine="earcut")
            raw.apply_translation([0, 0, z - mid_plane / 2])
            raw.metadata.update({"coverage": "measured-planar-gap", "parent_index": g["parent"], "plane": g["plane"].tolist()})
            parent = meshes[g["parent"]]
            if hasattr(parent.visual, "material"):
                raw.visual.material = parent.visual.material
            patches.append(raw)
        area = float(support.area)
        report["supportedAreaM2"] += area
        report["unresolvedAreaM2"] -= area
        report["components"].append({"areaM2": float(component.area), "supportedAreaM2": area, "status": "supported", "parent_index": g["parent"]})
    report["status"] = "supported" if report["supportedAreaM2"] > 0 and report["unresolvedAreaM2"] <= 1e-8 else report["status"]
    return patches, report
