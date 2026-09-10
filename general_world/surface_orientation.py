"""Read-only audit for downward-facing open roof caps."""
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


class SurfaceOrientationError(ValueError):
    pass


def _edges(faces, vertices=None):
    edges = {}
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            if vertices is None:
                key = tuple(sorted((int(a), int(b))))
            else:
                key = tuple(sorted((tuple(np.round(vertices[int(i)], 7)) for i in (a, b))))
            edges.setdefault(key, []).append(i)
    return edges


def _components(face_indices, faces, vertices=None):
    by_edge = _edges(faces, vertices)
    wanted = set(int(i) for i in face_indices)
    links = {i: set() for i in wanted}
    for values in by_edge.values():
        values = [i for i in values if i in wanted]
        for i in values:
            links[i].update(values)
    result = []
    while links:
        root = next(iter(links)); stack = [root]; group = set()
        while stack:
            i = stack.pop()
            if i in group: continue
            group.add(i); stack.extend(links[i] - group)
        for i in group: links.pop(i, None)
        result.append(sorted(group))
    return result


def _polygon(triangles):
    return unary_union([Polygon(t[:, :2]) for t in triangles if Polygon(t[:, :2]).area > 1e-8])


def _plane_z(triangles):
    return float(np.mean(triangles[:, :, 2]))


def audit_downward_caps(mesh, other_meshes, footprint, point_cloud, minimum_roof_z,
                        min_face_area=2.0, residual_m=.15):
    """Return indices of supported open downward caps safe to reverse.

    No mesh is modified. Indices refer to ``mesh.faces`` in the original input.
    """
    if not isinstance(mesh, trimesh.Trimesh) or not isinstance(footprint, (Polygon,)) or not footprint.is_valid:
        raise SurfaceOrientationError("mesh and valid polygon footprint are required")
    cloud = np.asarray(point_cloud, float)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or not np.isfinite(cloud[:, :3]).all():
        raise SurfaceOrientationError("point_cloud must be finite XYZ rows")
    meshes = [mesh] + list(other_meshes or [])
    tri = np.asarray(mesh.triangles)
    area = np.asarray(mesh.area_faces)
    downward = np.where((mesh.face_normals[:, 2] < -.98) & (area >= min_face_area))[0]
    reports = []; repair = []
    for group in _components(downward, mesh.faces, mesh.vertices):
        triangles = tri[group]; region = _polygon(triangles); z = _plane_z(triangles)
        report = {"face_indices": group, "areaM2": float(sum(area[group])), "planeZ": z, "status": "rejected"}
        if region.is_empty or not region.intersects(footprint):
            report["reason"] = "outside-footprint"; reports.append(report); continue
        component_faces = _components(range(len(mesh.faces)), mesh.faces, mesh.vertices)
        owner = next((c for c in component_faces if group[0] in c), group)
        sub = mesh.submesh([owner], append=True, repair=False)
        sub.merge_vertices(digits_vertex=7)
        if sub.is_watertight and sub.is_volume:
            report["reason"] = "closed-solid"; reports.append(report); continue
        in_cloud = cloud[contains_region(cloud, region)]
        in_cloud = in_cloud[in_cloud[:, 2] >= minimum_roof_z]
        in_cloud = in_cloud[np.abs(in_cloud[:, 2] - z) <= residual_m]
        cells = {tuple(np.floor(p[:2]).astype(int)) for p in in_cloud}
        report["sampleCount"] = int(len(in_cloud)); report["cellCount"] = int(len(cells))
        if len(in_cloud) < 12 or len(cells) < 6 or len(cells) < .25 * region.area:
            report["reason"] = "insufficient-distributed-lidar"; reports.append(report); continue
        reject = _overlying_roof(region, z, meshes, footprint)
        if reject:
            report["reason"] = reject; reports.append(report); continue
        report["status"] = "repairable"; repair.extend(group); reports.append(report)
    return {"reverse_face_indices": sorted(set(repair)), "reports": reports,
            "repairable": bool(repair), "candidateCount": len(downward)}


def find_reversed_caps(mesh, other_meshes, footprint, point_cloud, minimum_roof_z=0.0):
    """Adapter returning original face IDs and a JSON-safe audit report."""
    result = audit_downward_caps(mesh, other_meshes, footprint, point_cloud, minimum_roof_z)
    return result["reverse_face_indices"], result


def contains_region(cloud, region):
    return np.array([region.covers(Point(float(p[0]), float(p[1]))) for p in cloud], dtype=bool)


def _overlying_roof(region, z, meshes, footprint):
    for m in meshes:
        up = np.where(m.face_normals[:, 2] > .98)[0]
        tops = [Polygon(m.triangles[i][:, :2]) for i in up
                if np.all(m.triangles[i][:, 2] >= z) and np.all(m.triangles[i][:, 2] < z + 1.0)]
        if tops:
            coverage = unary_union(tops).intersection(region).area / max(region.area, 1e-9)
            if coverage >= .8:
                return "overlying-upward-surface"
        vertical = np.where(np.abs(m.face_normals[:, 2]) < .1)[0]
        for i in vertical:
            t = m.triangles[i]; p = Polygon(t[:, :2]).buffer(1e-6)
            if t[:, 2].max() > z + 1e-5 and p.intersects(region.boundary):
                return "adjoining-wall-above-plane"
    return None
