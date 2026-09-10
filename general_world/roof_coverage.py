"""Conservative recovery of small, measured gaps in planar roof coverage."""
import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, box
from shapely.ops import unary_union
from shapely import intersects_xy
from shapely.strtree import STRtree
from .geometry import extruded


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
    if np.linalg.norm(n) < 1e-9:
        return None
    n /= np.linalg.norm(n)
    if n[2] < .5:
        return None
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
            match = next((g for g in groups if g["parent"] == parent and np.linalg.norm(c[:2] - g["plane"][:2]) < 1e-5 and abs(c[2] - g["plane"][2]) < 1e-3), None)
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
    points=np.asarray(points)
    points=points[intersects_xy(region,points[:,0],points[:,1])]
    keys = np.floor(points[:, :2] / cell).astype(int)
    out = {}
    for key, point in zip(map(tuple, keys), points):
        out.setdefault(key, []).append(point)
    return out


def recover_roof_coverage(roof_meshes, footprint, point_cloud, minimum_roof_z, max_cap_distance=4.0,
                          residual_m=.25, cell_m=1.0, thickness_m=.02, occupied_domains=(), occupied_surfaces=(), wall_top_segments=()):
    """Recover only point-supported gap components adjacent to an existing plane."""
    if not all(np.isfinite(v) and v > 0 for v in (max_cap_distance, residual_m, cell_m, thickness_m)) or not np.isfinite(minimum_roof_z):
        raise RoofCoverageError("coverage thresholds must be positive and finite")
    meshes = list(roof_meshes)
    if not meshes or not all(isinstance(m, trimesh.Trimesh) for m in meshes):
        raise RoofCoverageError("roof_meshes must contain trimesh meshes")
    if not isinstance(footprint, (Polygon, MultiPolygon)) or footprint.is_empty or not footprint.is_valid:
        raise RoofCoverageError("footprint must be a valid polygon")
    points = np.asarray(point_cloud, float)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        raise RoofCoverageError("point_cloud must be finite XYZ rows")
    groups = _plane_groups(meshes)
    occupancy=list(occupied_surfaces)
    occupancy_tree=STRtree([domain for domain,height in occupancy])
    observed = unary_union([g["surface"] for g in groups] + list(occupied_domains))
    expected = footprint
    missing = expected.difference(observed)
    overlap = observed.intersection(expected)
    report = {"expectedAreaM2": float(expected.area), "observedAreaM2": float(overlap.area),
              "gapAreaM2": float(missing.area), "supportedAreaM2": 0.0, "unresolvedAreaM2": float(missing.area),
              "components": [], "status": "no-overlap-existing" if overlap.is_empty else "unresolved"}
    patches = []
    if overlap.is_empty:
        return patches, report
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
            cells = [box(k[0] * cell_m, k[1] * cell_m, (k[0] + 1) * cell_m, (k[1] + 1) * cell_m) for k in cell_points]
            support = unary_union(cells).intersection(g["surface"].buffer(max_cap_distance)).intersection(component) if cells else Polygon()
            # Existing faces within the measurement tolerance already own this
            # surface, even if they lie slightly below the fitted plane. Adding
            # a second sheet there changes materials and can cause z-fighting.
            # Only a distinctly lower surface leaves a possible upper-roof gap.
            blocked=[]
            for i in occupancy_tree.query(support,predicate='intersects'):
                domain,top_z=occupancy[i]
                q=domain.representative_point()
                if top_z >= plane[0]*q.x+plane[1]*q.y+plane[2]-residual_m-thickness_m:blocked.append(domain)
            if blocked:support=support.difference(unary_union(blocked))
            connected = []
            for piece in _polys(support):
                supported_cells = sum(any(piece.covers(Point(*q[:2])) for q in values) for values in cell_points.values())
                if piece.distance(g["surface"].boundary) <= 1e-6 and supported_cells >= 3:
                    connected.append(piece)
            support = unary_union(connected) if connected else Polygon()
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
        emitted_area=0.
        for piece in _polys(support):
            if piece.area < .001:
                continue
            # A ground footprint does not describe an upper storey's setbacks.
            # Every new perimeter must meet an existing roof or measured wall
            # top at this height. Sparse roof points alone cannot establish a
            # supported overhang, and sample-cell borders are not architecture.
            edge_support=[other['surface'].buffer(.05) for other in groups
                          if np.linalg.norm(other['plane'][:2]-g['plane'][:2])<1e-5
                          and abs(other['plane'][2]-g['plane'][2])<.001]
            for segment in wall_top_segments:
                q=np.asarray(segment,float)
                if q.shape!=(2,3) or not np.isfinite(q).all():
                    raise RoofCoverageError('wall top segments must contain two finite XYZ endpoints')
                predicted=q[:,:2]@g['plane'][:2]+g['plane'][2]
                if np.max(np.abs(q[:,2]-predicted))<=thickness_m+.001:
                    edge_support.append(LineString(q[:,:2]).buffer(.05))
            unsupported_edge=piece.boundary.difference(unary_union(edge_support))
            if unsupported_edge.length>.05:
                report.setdefault('rejectedUnsupportedBoundaryPatches',0)
                report['rejectedUnsupportedBoundaryPatches']+=1
                continue
            solids=extruded(piece,0,thickness_m)
            if len(solids)!=1:continue
            raw=solids[0]
            raw.vertices=np.array(raw.vertices,copy=True)
            raw.vertices[:, 2] = g["plane"][0] * raw.vertices[:, 0] + g["plane"][1] * raw.vertices[:, 1] + g["plane"][2] + (raw.vertices[:, 2] - thickness_m)
            # Validate at the precision actually written into the game asset.
            # A valid double-precision sliver can collapse during GLB export.
            raw.vertices=raw.vertices.astype(np.float32).astype(float)
            raw.merge_vertices()
            raw.update_faces(raw.area_faces>0)
            if not raw.is_volume or not np.isfinite(raw.face_normals).all() or np.any(np.linalg.norm(raw.face_normals,axis=1)<.5):
                report.setdefault('rejectedUnrenderablePatches',0)
                report['rejectedUnrenderablePatches']+=1
                continue
            raw.metadata.update({"coverage": "measured-planar-gap", "parent_index": g["parent"], "plane": g["plane"].tolist()})
            parent = meshes[g["parent"]]
            if hasattr(parent.visual, "material"):
                raw.visual.material = parent.visual.material
            patches.append(raw)
            emitted_area+=float(piece.area)
        area = emitted_area
        report["supportedAreaM2"] += area
        report["unresolvedAreaM2"] -= area
        report["components"].append({"areaM2": float(component.area), "supportedAreaM2": area, "status": "supported", "parent_index": g["parent"]})
    report["status"] = "supported" if report["supportedAreaM2"] > 0 and report["unresolvedAreaM2"] <= 1e-8 else report["status"]
    return patches, report
