"""Read-only audit for supported open downward sloped roof caps.

The detector returns original face indices and evidence reports.  It never
changes mesh topology, shape, materials, UVs, or orientation itself.
"""

from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

try:
    from .surface_orientation import _components, contains_region
except ImportError:  # pragma: no cover - allows direct module use
    from surface_orientation import _components, contains_region


class SlopedSurfaceOrientationError(ValueError):
    pass


def _region(triangles: np.ndarray):
    polygons = [Polygon(triangle[:, :2]) for triangle in triangles]
    polygons = [polygon for polygon in polygons if polygon.is_valid and polygon.area > 1e-8]
    return unary_union(polygons) if polygons else Polygon()


def _fit_plane(triangles: np.ndarray, face_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = np.unique(np.round(triangles.reshape(-1, 3), 7), axis=0)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    normal = vh[-1]
    mean_normal = np.mean(face_normals, axis=0)
    if np.dot(normal, mean_normal) < 0:
        normal = -normal
    residual = np.abs((vertices - center) @ normal)
    return center, normal / max(np.linalg.norm(normal), 1e-12), float(np.max(residual))


def _split_by_normal(face_indices: list[int], faces: np.ndarray, normals: np.ndarray, max_angle_deg: float = 5.0) -> list[list[int]]:
    """Split a connected patch at a roof ridge or other normal discontinuity."""

    wanted = set(int(index) for index in face_indices)
    edges: dict[tuple[int, int], list[int]] = {}
    for index in wanted:
        face = faces[index]
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    links = {index: set() for index in wanted}
    limit = np.cos(np.radians(max_angle_deg))
    for values in edges.values():
        for left in values:
            for right in values:
                if left != right and float(np.dot(normals[left], normals[right])) >= limit:
                    links[left].add(right)
    components = []
    while links:
        root = next(iter(links)); stack = [root]; component = set()
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index); stack.extend(links[index] - component)
        for index in component:
            links.pop(index, None)
        components.append(sorted(component))
    return components


def _z_at_xy(point: np.ndarray, normal: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if abs(normal[2]) < 1e-8:
        return np.full(len(xy), point[2])
    return point[2] - ((xy - point[:2]) @ normal[:2]) / normal[2]


def _overlying_surface(region, plane_point: np.ndarray, plane_normal: np.ndarray, meshes: list[trimesh.Trimesh], connected_shell: trimesh.Trimesh, max_clearance_m: float = 2.0) -> str | None:
    """Reject a roof/skin above the candidate, including split neighboring meshes."""

    candidate_up = -plane_normal
    candidate_polygons = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        normals = mesh.face_normals
        for index in np.where((normals @ candidate_up > 0.2) & (normals[:, 2] > 0.05))[0]:
            triangle = mesh.triangles[index]
            polygon = Polygon(triangle[:, :2])
            if polygon.is_empty or not polygon.is_valid:
                continue
            centroid = triangle.mean(axis=0)
            candidate_z = float(_z_at_xy(plane_point, plane_normal, centroid[None, :2])[0])
            clearance = float(centroid[2] - candidate_z)
            if 0.05 <= clearance <= max_clearance_m:
                candidate_polygons.append(polygon)
    if candidate_polygons:
        overlap = unary_union(candidate_polygons).intersection(region).area / max(region.area, 1e-9)
        if overlap >= 0.8:
            return "overlying-upward-surface"

    # A connected wall/parapet above the sloped patch can hide the topology
    # boundary.  Only inspect the candidate's connected shell here.
    for index in np.where(np.abs(connected_shell.face_normals @ candidate_up) < 0.2)[0]:
        triangle = connected_shell.triangles[index]
        polygon = Polygon(triangle[:, :2]).buffer(1e-6)
        centroid = triangle.mean(axis=0)
        candidate_z = float(_z_at_xy(plane_point, plane_normal, centroid[None, :2])[0])
        if triangle[:, 2].max() > candidate_z + 0.05 and polygon.intersects(region.boundary):
            return "adjoining-wall-above-plane"
    return None


def audit_sloped_caps(
    mesh: trimesh.Trimesh,
    other_meshes: list[trimesh.Trimesh] | None,
    footprint: Polygon | MultiPolygon,
    point_cloud: np.ndarray,
    minimum_roof_z: float,
    min_face_area: float = 2.0,
    residual_m: float = 0.15,
    side_tolerance_m: float = 0.02,
    min_downward_z: float = 0.15,
    max_downward_z: float = 0.98,
) -> dict:
    """Audit connected downward sloped patches with distributed LiDAR support.

    ``reverse_face_indices`` is safe only for callers that review the returned
    report.  In particular, points must be close to the patch and on its
    roof-facing side; this avoids treating a supported canopy underside as roof.
    """

    if not isinstance(mesh, trimesh.Trimesh) or not isinstance(footprint, (Polygon, MultiPolygon)) or not footprint.is_valid:
        raise SlopedSurfaceOrientationError("mesh and valid polygon footprint are required")
    cloud = np.asarray(point_cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or not np.isfinite(cloud[:, :3]).all():
        raise SlopedSurfaceOrientationError("point_cloud must be finite XYZ rows")
    meshes = [mesh] + list(other_meshes or [])
    triangles = np.asarray(mesh.triangles)
    areas = np.asarray(mesh.area_faces)
    normals = np.asarray(mesh.face_normals)
    downward = np.where((normals[:, 2] <= -min_downward_z) & (normals[:, 2] >= -max_downward_z) & (areas >= min_face_area))[0]
    reports = []
    repair = []
    all_components = _components(range(len(mesh.faces)), mesh.faces, mesh.vertices) if len(downward) else []
    for connected_group in _components(downward, mesh.faces, mesh.vertices):
        for group in _split_by_normal(connected_group, mesh.faces, normals):
            patch_triangles = triangles[group]
            region = _region(patch_triangles)
            center, normal, plane_residual = _fit_plane(patch_triangles, normals[group])
            report = {"face_indices": group, "areaM2": float(np.sum(areas[group])), "planePoint": center.tolist(), "planeNormal": normal.tolist(), "planeResidualM": plane_residual, "status": "rejected"}
            if plane_residual > 0.05:
                report["reason"] = "nonplanar-sloped-patch"; reports.append(report); continue
            if region.is_empty or region.intersection(footprint).area < 0.7 * region.area:
                report["reason"] = "outside-footprint"; reports.append(report); continue
            owner = next((component for component in all_components if group[0] in component), group)
            shell = mesh.submesh([owner], append=True, repair=False)
            shell.merge_vertices(digits_vertex=7, merge_tex=True, merge_norm=True)
            if shell.is_watertight:
                report["reason"] = "closed-solid"; reports.append(report); continue
            in_region = cloud[contains_region(cloud, region)]
            in_region = in_region[in_region[:, 2] >= minimum_roof_z]
            signed_roof_side = (in_region - center) @ (-normal)
            in_region = in_region[signed_roof_side >= -side_tolerance_m]
            in_region = in_region[np.abs((in_region - center) @ normal) <= residual_m]
            cells = {tuple(np.floor(point[:2]).astype(int)) for point in in_region}
            report.update({"sampleCount": int(len(in_region)), "cellCount": int(len(cells)), "regionAreaM2": float(region.area), "supportCellFraction": float(len(cells) / max(region.area, 1e-9))})
            if len(in_region) < 12 or len(cells) < 6 or len(cells) < 0.25 * region.area:
                report["reason"] = "insufficient-distributed-lidar"; reports.append(report); continue
            overlying = _overlying_surface(region, center, normal, meshes, shell)
            if overlying:
                report["reason"] = overlying; reports.append(report); continue
            report["status"] = "repairable"
            repair.extend(group)
            reports.append(report)
    return {"reverse_face_indices": sorted(set(repair)), "reports": reports, "repairable": bool(repair), "candidateCount": int(len(downward))}


def find_reversed_sloped_caps(mesh, other_meshes, footprint, point_cloud, minimum_roof_z=0.0):
    """Small adapter matching the existing read-only orientation helper style."""

    result = audit_sloped_caps(mesh, other_meshes, footprint, point_cloud, minimum_roof_z)
    return result["reverse_face_indices"], result
