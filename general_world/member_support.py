"""Geometry-only cleanup for long elevated members lacking support evidence.

The caller must explicitly attest that support is missing.  This module does not
inspect labels, materials, scene nodes, or site-specific identifiers.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Point

from .geometry import extruded


def _components(mesh: trimesh.Trimesh):
    """Return face-index components using welded geometry and shared edges."""
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return []
    adjacency = np.asarray(mesh.face_adjacency)
    if len(adjacency) == 0:
        return [np.array([i], dtype=int) for i in range(len(faces))]
    rows = np.r_[adjacency[:, 0], adjacency[:, 1]]
    cols = np.r_[adjacency[:, 1], adjacency[:, 0]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(faces), len(faces)))
    count, labels = connected_components(graph.tocsr(), directed=False)
    return [np.flatnonzero(labels == i) for i in range(count)]


def _welded_copy(mesh: trimesh.Trimesh):
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        process=False,
    )
    out.merge_vertices(digits_vertex=8)
    return out


def _original_face_map(source: trimesh.Trimesh, welded: trimesh.Trimesh):
    """Map welded triangle rows back to source face indices by XYZ coordinates."""
    def key(tri):
        return tri[np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))].tobytes()

    source_map = {}
    source_vertices = np.asarray(source.vertices)
    for index, face in enumerate(np.asarray(source.faces)):
        source_map.setdefault(key(source_vertices[face]), []).append(index)
    welded_vertices = np.asarray(welded.vertices)
    result = []
    for face in np.asarray(welded.faces):
        values = source_map.get(key(welded_vertices[face]), [])
        result.append(values.pop(0) if values else None)
    return result


def _max_overreach(points, footprint):
    distances = []
    for x, y in np.asarray(points)[:, :2]:
        point = Point(float(x), float(y))
        if not footprint.covers(point):
            distances.append(point.distance(footprint))
    return float(max(distances, default=0.0))


def _candidate(component, footprint):
    bounds = np.asarray(component.bounds)
    dims = bounds[1] - bounds[0]
    horizontal = np.sort(dims[:2])
    return {
        "elevated": bool(bounds[0, 2] > 5.0),
        "long": bool(horizontal[1] > 12.0),
        "thin": bool(dims[2] < 0.30),
        "narrow": bool(horizontal[0] < 2.0),
        "overreachM": _max_overreach(component.vertices, footprint),
        "bounds": bounds.tolist(),
    }


def clip_long_members(
    mesh: trimesh.Trimesh,
    footprint,
    margin_m: float = 2.0,
    *,
    missingSupportEvidence: bool = False,
    measuredSupport: bool = False,
):
    """Clip unsupported, long elevated members to a footprint support envelope.

    Returns ``(replacements, report)``.  Each replacement has the source face
    indices and a geometry-only mesh.  Components that fail the closed,
    positive-volume and dimensional gates are returned unchanged.  The input
    mesh and its UV/material arrays are never modified.
    """
    if margin_m < 0:
        raise ValueError("margin_m must be non-negative")
    if measuredSupport or not missingSupportEvidence:
        reason = "measured-support" if measuredSupport else "missing-support-evidence-required"
        return [], {"changed": False, "reason": reason, "components": 0, "offending": 0}

    source = mesh
    welded = _welded_copy(source)
    face_map = _original_face_map(source, welded)
    replacements = []
    report = {"changed": False, "reason": "audited", "components": 0, "offending": 0, "failedClips": 0}

    for component_faces in _components(welded):
        component = trimesh.Trimesh(
            vertices=np.asarray(welded.vertices),
            faces=np.asarray(welded.faces)[component_faces],
            process=False,
        )
        report["components"] += 1
        source_indices = [face_map[i] for i in component_faces if face_map[i] is not None]
        metrics = _candidate(component, footprint)
        closed_positive = bool(component.is_watertight and component.is_volume and component.volume > 0)
        offending = closed_positive and all(metrics[k] for k in ("elevated", "long", "thin", "narrow")) and metrics["overreachM"] > 5.0
        if not offending:
            replacements.append({"originalFaceIndices": source_indices, "mesh": component, "changed": False, "metrics": metrics})
            continue
        report["offending"] += 1
        z0, z1 = component.bounds[:, 2]
        support_parts = extruded(footprint.buffer(margin_m), z0 - 0.01, z1 + 0.01)
        try:
            clipped = trimesh.boolean.intersection([component, *support_parts], engine="manifold", check_volume=True)
            if isinstance(clipped, list):
                clipped = trimesh.util.concatenate(clipped)
            if clipped is None or len(clipped.faces) == 0 or not clipped.is_volume:
                raise ValueError("intersection did not produce a positive solid")
            replacements.append({"originalFaceIndices": source_indices, "mesh": clipped, "changed": True, "metrics": metrics})
            report["changed"] = True
        except Exception as exc:
            report["failedClips"] += 1
            replacements.append({"originalFaceIndices": source_indices, "mesh": component, "changed": False, "metrics": metrics, "error": str(exc)[:160]})
    return replacements, report

