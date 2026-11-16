"""Clip structural wall pieces where they sit in front of a validated bay skin."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def replacement_record(*, origin, tangent, normal, plane, zBase, xRange,
                       zRange, support, sourceId, profileId, bayIndex):
    """Plain data record for one validated bay assembly and its source frame."""
    return {"origin": np.asarray(origin, dtype=float).tolist(),
            "tangent": np.asarray(tangent, dtype=float).tolist(),
            "normal": np.asarray(normal, dtype=float).tolist(),
            "plane": float(plane), "zBase": float(zBase),
            "xRange": list(map(float, xRange)), "zRange": list(map(float, zRange)),
            "support": np.asarray(support, dtype=float).tolist(),
            "sourceId": str(sourceId), "profileId": str(profileId),
            "bayIndex": int(bayIndex)}


def _cutter_plan(replacement, maximum_depth, face_thickness=.12):
    """Plan regions from the inner skin surface to a wall's outward extent."""
    points = np.asarray(replacement["support"], dtype=float)
    if len(points) < 2:
        return []
    runs = []
    current = []
    for a, b in zip(points[:-1], points[1:]):
        da = max(.01, float(a[1]) - face_thickness)
        db = max(.01, float(b[1]) - face_thickness)
        ta, tb = 0., 1.
        if da >= maximum_depth and db >= maximum_depth:
            if current:
                runs.append(current); current = []
            continue
        if da >= maximum_depth:
            ta = float(np.clip((maximum_depth - da) / (db - da), 0., 1.))
        if db >= maximum_depth:
            tb = float(np.clip((maximum_depth - da) / (db - da), 0., 1.))
        if tb <= ta + 1e-10:
            continue
        lower_a = np.array([a[0], da], dtype=float)
        lower_b = np.array([b[0], db], dtype=float)
        p0 = lower_a + ta * (lower_b - lower_a)
        p1 = lower_a + tb * (lower_b - lower_a)
        if not current or np.linalg.norm(current[-1] - p0) > 1e-7:
            if current:
                runs.append(current)
            current = [p0]
        current.append(p1)
        if tb < 1. - 1e-10:
            runs.append(current); current = []
    if current:
        runs.append(current)
    polygons = []
    for run in runs:
        if len(run) < 2:
            continue
        ring = [*run, [run[-1][0], maximum_depth], [run[0][0], maximum_depth]]
        polygon = Polygon(ring)
        if polygon.is_valid and polygon.area > 1e-6:
            polygons.append(polygon)
    return polygons


def clip_structural_mesh(mesh, replacements):
    """Subtract only the part of structural walls at or in front of bay skins.

    Each replacement has a source facade frame and the bay's plan support
    polyline. The source-plane backing remains: clipping starts just inside
    the bay's exterior skin, not at the host wall plane. Wall pieces with no
    overlapping x/z bounds are returned untouched.
    """
    if mesh is None or not len(getattr(mesh, "faces", ())):
        return mesh
    import trimesh

    current = mesh
    for replacement in replacements or ():
        vertices = np.asarray(current.vertices, dtype=float)
        origin = np.asarray(replacement["origin"], dtype=float)
        tangent = np.asarray(replacement["tangent"], dtype=float)
        normal = np.asarray(replacement["normal"], dtype=float)
        local_xy = (vertices[:, :2] - origin) @ np.column_stack((tangent, normal))
        local_z = vertices[:, 2] - float(replacement["zBase"])
        x0, x1 = map(float, replacement["xRange"])
        z0, z1 = map(float, replacement["zRange"])
        if (local_xy[:, 0].max() < x0 - .2 or local_xy[:, 0].min() > x1 + .2
                or local_z.max() < z0 - .02 or local_z.min() > z1 + .02):
            continue
        if (local_z.max() < z0 - .02 or local_z.min() > z1 + .02):
            continue
        near = ((local_xy[:, 0] >= x0 - .2) & (local_xy[:, 0] <= x1 + .2)
                & (local_z >= z0 - .02) & (local_z <= z1 + .02))
        # Long wall slabs can cross the bay's x/z rectangle without placing a
        # mesh vertex inside it. Their plane depth is still represented by all
        # their vertices, so use that only for this bbox-overlapping solid.
        maximum_depth = float(local_xy[near, 1].max() if near.any()
                              else local_xy[:, 1].max())
        cutters = _cutter_plan(replacement, maximum_depth)
        if not cutters:
            continue
        try:
            cutters_mesh = []
            for plan in cutters:
                local = trimesh.creation.extrude_polygon(
                    plan, height=z1 - z0 + .02, engine="earcut")
                transform = np.eye(4)
                transform[:3, 0] = [*tangent, 0.]
                transform[:3, 1] = [*normal, 0.]
                transform[:3, 2] = [0., 0., 1.]
                transform[:3, 3] = [*origin, float(replacement["zBase"]) + z0 - .01]
                local.apply_transform(transform)
                cutters_mesh.append(local)
            result = trimesh.boolean.difference([current, *cutters_mesh], engine="manifold")
            if result is not None and len(result.faces):
                current = result
            elif result is not None and not len(result.faces):
                return None
        except Exception:
            # A wall remains intact if the mesh kernel cannot make a reliable
            # solid difference. The bay itself is still emitted as measured.
            continue
    return current
