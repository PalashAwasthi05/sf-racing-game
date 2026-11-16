"""Preserve authored surface coordinates when clipping generates new vertices."""
import numpy as np
import trimesh


def transfer_metric_uv(source, target):
    """Interpolate source metric UVs on the original surface, including cut rims.

    Unchanged vertices retain their original chart exactly. New clipping vertices
    lie on source triangles; new cap interiors continue the nearest boundary
    chart. No per-face world-axis projection restarts the texture phase.
    """
    if target is None or source is target or 'uvMeters' not in source.metadata:
        return target
    uv = np.asarray(source.metadata['uvMeters'], float)
    if uv.shape != (len(source.vertices), 2) or not np.isfinite(uv).all():
        raise ValueError('authored metric UVs must be finite per-vertex pairs')
    if not len(target.vertices):
        return target
    # Surface builders emit small component meshes. Bound the pairwise work so
    # this remains portable without an optional native spatial-index package.
    chunk = max(1, 250_000 // max(1, len(source.faces)))
    matches = [trimesh.proximity.closest_point_naive(source, target.vertices[i:i+chunk])
               for i in range(0, len(target.vertices), chunk)]
    closest = np.concatenate([item[0] for item in matches])
    face_ids = np.concatenate([item[2] for item in matches])
    triangles = source.triangles[face_ids]
    weights = trimesh.triangles.points_to_barycentric(triangles, closest)
    values = np.einsum('ij,ijk->ik', weights, uv[source.faces[face_ids]])
    if not np.isfinite(values).all():
        raise ValueError('cannot interpolate UVs from degenerate source triangles')
    target.metadata['uvMeters'] = values
    return target
