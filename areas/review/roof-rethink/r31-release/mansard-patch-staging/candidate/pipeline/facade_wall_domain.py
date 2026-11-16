"""Project roof-shell boundary samples into photographed facade frames and subtract owned wall faces from roof skirts."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from shapely import constrained_delaunay_triangles
from shapely.affinity import affine_transform


def project_boundary_samples(samples, normal, frame_left, plane, z_base, x_range, plane_tolerance=.1,
                             endpoint_tolerance=.75):
    """Project world XYZ samples near one source facade plane to ordered local (x,z) points.

    `normal` and `frame_left` are from the source profile frame. Samples must span the entire compiled segment, with
    at most `endpoint_tolerance` of unsampled margin at either end; this prevents a neighboring roof or a partial shell
    boundary from being extrapolated into a wall domain.
    """
    points = np.asarray(samples, dtype=float)
    n = np.asarray(normal, dtype=float)[:2]
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all(): return None
    norm = float(np.linalg.norm(n))
    if norm < 1e-9: return None
    n /= norm
    u = np.array([-n[1], n[0]])
    left, plane, base = float(frame_left), float(plane), float(z_base)
    lo, hi = sorted(map(float, x_range))
    signed = points[:, :2] @ n - plane
    x = points[:, :2] @ u - left
    keep = (np.abs(signed) <= float(plane_tolerance)) & (x >= lo - endpoint_tolerance) & (x <= hi + endpoint_tolerance)
    if keep.sum() < 2: return None
    out = np.c_[x[keep], points[keep, 2] - base]
    out = out[np.argsort(out[:, 0])]
    # Roof-shell triangulation can repeat an x at adjacent face boundaries. Keep the highest boundary sample at each
    # repeated x so a seam cannot create a downward spike in the facade's top outline.
    merged = []
    for xx in np.unique(np.round(out[:, 0], 6)):
        same = out[np.abs(out[:, 0] - xx) <= 1e-6]
        merged.append((float(xx), float(np.max(same[:, 1]))))
    out = np.asarray(merged, dtype=float)
    if out[0, 0] > lo + endpoint_tolerance or out[-1, 0] < hi - endpoint_tolerance: return None
    return out


def wall_domain_from_top_samples(top_samples, x_range, base_z, coverage_tolerance=.75):
    """Close a source-frame roof-boundary trace to the wall foot, only when it covers the whole segment."""
    pts = np.asarray(top_samples, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or not np.isfinite(pts).all(): return None
    lo, hi = sorted(map(float, x_range)); base = float(base_z)
    pts = pts[np.argsort(pts[:, 0])]
    merged = []
    for xx in np.unique(np.round(pts[:, 0], 6)):
        same = pts[np.abs(pts[:, 0] - xx) <= 1e-6]
        merged.append((float(xx), float(np.max(same[:, 1]))))
    pts = np.asarray(merged, dtype=float)
    if len(pts) < 2 or pts[0, 0] > lo + coverage_tolerance or pts[-1, 0] < hi - coverage_tolerance:
        return None
    inside = pts[(pts[:, 0] > lo + 1e-8) & (pts[:, 0] < hi - 1e-8)]
    trace = np.vstack(([lo, np.interp(lo, pts[:, 0], pts[:, 1])], inside,
                       [hi, np.interp(hi, pts[:, 0], pts[:, 1])]))
    if np.any(trace[:, 1] <= base + .05): return None
    domain = Polygon([(lo, base), *map(tuple, trace), (hi, base)]).buffer(0)
    if domain.is_empty or domain.geom_type != 'Polygon' or not domain.is_valid or domain.area < .1:
        return None
    return domain


def clip_owned_skirt(skirt, owners, plane_tolerance=.08, area_tolerance=1e-7):
    """Remove shell-skirt triangles covered by compiled facade wall domains, retaining all unmatched skirt geometry.

    An owner stores the facade's world-space normal/plane/frame and its local x-z wall polygon. Only triangles whose
    three vertices lie close to that actual facade plane are considered; neighboring or recessed skirt faces survive.
    """
    import trimesh
    if skirt is None or not len(getattr(skirt, 'faces', ())): return skirt.copy() if skirt is not None else None
    if not owners: return skirt.copy()
    kept_meshes = []
    for face in np.asarray(skirt.faces, dtype=int):
        tri = np.asarray(skirt.vertices[face], dtype=float)
        matched = []
        for owner in owners:
            n = np.asarray(owner['normal'], dtype=float)[:2]
            n /= max(float(np.linalg.norm(n)), 1e-12)
            plane = float(owner['plane'])
            if np.max(np.abs(tri[:, :2] @ n - plane)) > plane_tolerance: continue
            u = np.asarray(owner['tangent'], dtype=float)[:2]
            u /= max(float(np.linalg.norm(u)), 1e-12)
            local = np.c_[tri[:, :2] @ u - float(owner['left']), tri[:, 2] - float(owner['zBaseM'])]
            tri_poly = Polygon(local)
            domain = owner['domain']
            if tri_poly.area > area_tolerance and tri_poly.intersection(domain).area > area_tolerance:
                matched.append((owner, tri_poly))
        if not matched:
            kept_meshes.append(trimesh.Trimesh(vertices=tri, faces=[[0, 1, 2]], process=False))
            continue
        # Multiple coplanar facade segments may own different parts of one skirt triangle.
        owner = matched[0][0]
        owner_tangent = np.asarray(owner['tangent'], dtype=float)[:2]
        owner_tangent /= max(float(np.linalg.norm(owner_tangent)), 1e-12)
        union = owner['domain']
        for other, _ in matched[1:]:
            # A single skirt triangle can meet two wall pieces whose metric
            # frames start at different `left`/`zBaseM` origins (or run in
            # opposite directions). Put every domain into the first owner's
            # frame before taking the union.
            other_tangent = np.asarray(other['tangent'], dtype=float)[:2]
            other_tangent /= max(float(np.linalg.norm(other_tangent)), 1e-12)
            other_normal = np.asarray(other['normal'], dtype=float)[:2]
            other_normal /= max(float(np.linalg.norm(other_normal)), 1e-12)
            scale_x = float(np.dot(other_tangent, owner_tangent))
            offset_x = float(np.dot(float(other['left']) * other_tangent + float(other['plane']) * other_normal,
                                    owner_tangent) - float(owner['left']))
            offset_z = float(other['zBaseM']) - float(owner['zBaseM'])
            converted = affine_transform(other['domain'], [scale_x, 0., 0., 1., offset_x, offset_z])
            union = union.union(converted)
        local = np.c_[tri[:, :2] @ owner_tangent - float(owner['left']),
                      tri[:, 2] - float(owner['zBaseM'])]
        remainder = Polygon(local).difference(union)
        polys = [remainder] if remainder.geom_type == 'Polygon' else list(getattr(remainder, 'geoms', []))
        for poly in polys:
            if poly.is_empty or poly.area <= area_tolerance: continue
            # Constrained triangulation honors concave boundaries and holes;
            # centroid filtering of unconstrained Delaunay triangles can leave
            # triangles that cross a notch or bridge an interior hole.
            for sub in constrained_delaunay_triangles(poly).geoms:
                if not poly.covers(sub) or sub.area <= area_tolerance: continue
                coords = np.asarray(sub.exterior.coords[:3], dtype=float)
                n = np.asarray(owner['normal'], float)[:2]; n /= max(float(np.linalg.norm(n)), 1e-12)
                u = owner_tangent
                world = np.c_[np.outer(float(owner['left']) + coords[:, 0], u) + float(owner['plane']) * n,
                                 float(owner['zBaseM']) + coords[:, 1]]
                # Shapely's exterior winding is fixed in local x-z, which need not
                # match the roof mesh face winding. Preserve the source face normal
                # so clipped, coplanar pieces shade exactly like the original skirt.
                source_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                piece_normal = np.cross(world[1] - world[0], world[2] - world[0])
                if float(np.dot(source_normal, piece_normal)) < 0.:
                    world = world[[0, 2, 1]]
                kept_meshes.append(trimesh.Trimesh(vertices=world, faces=[[0, 1, 2]], process=False))
    if not kept_meshes:
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False)
    result = trimesh.util.concatenate(kept_meshes)
    result.remove_unreferenced_vertices()
    return result


def project_profile_domain(samples, profile, x_range, z_base, plane, domain_base_z=0., plane_tolerance=.1):
    """Convenience projection from raw shell boundary XYZ into a source reading frame."""
    frame = profile.get('frameU')
    if not isinstance(frame, (list, tuple)) or len(frame) != 2: return None
    projected = project_boundary_samples(samples, profile.get('normal', (0., 1.)), frame[0], plane, z_base,
                                         x_range, plane_tolerance=plane_tolerance)
    if projected is None: return None
    return wall_domain_from_top_samples(projected, x_range, domain_base_z)


def project_boundary_chains(chains, normal, frame_left, plane, z_base, x_range, segment_a, segment_b,
                            plane_tolerance=.1, endpoint_tolerance=.12, max_gap_m=.8):
    """Select a shell boundary chain that lies on the actual compiled facade segment.

    A roof shell can have nearby parallel footprint edges (bays, recesses, or
    returns). Proximity to the facade plane alone cannot identify which one
    owns a wall. This selector requires both endpoints/tangent alignment with
    the actual segment and a tight plane residual, then returns that chain in
    the source profile's local (x,z) frame.
    """
    n = np.asarray(normal, dtype=float)[:2]
    norm = float(np.linalg.norm(n))
    if norm < 1e-9: return None
    n /= norm
    u = np.array([-n[1], n[0]])
    a = np.asarray(segment_a, dtype=float)[:2]
    b = np.asarray(segment_b, dtype=float)[:2]
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length < .1: return None
    tangent = edge / length
    if abs(float(np.dot(tangent, u))) < .96: return None
    lo, hi = sorted(map(float, x_range))
    left, plane, base = float(frame_left), float(plane), float(z_base)
    candidates = []
    for chain in chains or ():
        points = np.asarray(chain.get('points', chain) if isinstance(chain, dict) else chain, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) < 2 or not np.isfinite(points[:, :3]).all():
            continue
        delta = points[:, :2] - a
        t = delta @ tangent / length
        perpendicular = np.abs(delta[:, 0] * tangent[1] - delta[:, 1] * tangent[0])
        plane_error = np.abs(points[:, :2] @ n - plane)
        keep = ((t >= -endpoint_tolerance / length) & (t <= 1. + endpoint_tolerance / length)
                & (perpendicular <= plane_tolerance) & (plane_error <= plane_tolerance))
        if keep.sum() < 2: continue
        order = np.argsort(t[keep]); kept_t = np.clip(t[keep][order], 0., 1.)
        if kept_t[0] > endpoint_tolerance / length or kept_t[-1] < 1. - endpoint_tolerance / length:
            continue
        if np.max(np.diff(kept_t), initial=0.) * length > max_gap_m:
            continue
        selected = points[keep][order]
        local_x = selected[:, :2] @ u - left
        local_z = selected[:, 2] - base
        projected = np.c_[local_x, local_z]
        # The source profile's frame and actual segment must describe the same
        # wall interval; reject mismatches rather than borrowing a nearby edge.
        if projected[:, 0].min() > lo + endpoint_tolerance or projected[:, 0].max() < hi - endpoint_tolerance:
            continue
        score = (float(plane_error[keep].max()), float(perpendicular[keep].max()), -int(keep.sum()))
        candidates.append((score, projected))
    if not candidates: return None
    candidates.sort(key=lambda q: q[0])
    return candidates[0][1]


def project_profile_chain_domain(chains, profile, x_range, z_base, plane, segment_a, segment_b,
                                 domain_base_z=0., plane_tolerance=.1):
    """Project only the exact aligned shell-boundary chain for a compiled facade segment."""
    frame = profile.get('frameU')
    if not isinstance(frame, (list, tuple)) or len(frame) != 2: return None
    projected = project_boundary_chains(chains, profile.get('normal', (0., 1.)), frame[0], plane, z_base,
                                        x_range, segment_a, segment_b, plane_tolerance=plane_tolerance)
    if projected is None: return None
    return wall_domain_from_top_samples(projected, x_range, domain_base_z)


def observed_gable_domain(profile, x_range, frame_width, frame_height, base_z, shell_domain,
                          endpoint_tolerance=.75, minimum_rise=.5):
    """Use an explicitly traced gable outline when its eaves meet a mapped shell boundary chain.

    Some source profiles encode a visible gable outline under `parapets`. Treat
    it as a wall boundary only when it spans this segment, has a clear interior
    peak, and both endpoints agree with the exact shell-chain eaves. The check
    avoids interpreting ordinary parapet outlines as wall geometry.
    """
    if shell_domain is None: return None, set(), []
    lo, hi = sorted(map(float, x_range))
    height = float(frame_height)
    for index, parapet in enumerate(profile.get('parapets', [])):
        raw = np.asarray(parapet.get('top', []), dtype=float)
        if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 3 or not np.isfinite(raw).all(): continue
        pts = np.c_[raw[:, 0] * float(frame_width), raw[:, 1] * height]
        pts = pts[np.argsort(pts[:, 0])]
        if pts[0, 0] > lo + .12 or pts[-1, 0] < hi - .12: continue
        # Profile z and projected shell z are both metres above this facade's
        # world-space foot; base_z is only the polygon's lower closure level.
        source_local = pts[:, 1]
        trace = np.c_[np.interp([lo, hi], pts[:, 0], pts[:, 0]),
                      np.interp([lo, hi], pts[:, 0], source_local)]
        if float(np.max(pts[1:-1, 1])) < max(float(pts[0, 1]), float(pts[-1, 1])) + minimum_rise: continue
        # The shell chain anchors the eave line at the actual perimeter. Only
        # extend it to a photographed gable peak if those two boundary heights
        # agree within the profile calibration tolerance.
        def top_at(x):
            intersections = []
            coords = list(shell_domain.exterior.coords)
            for (xa, za), (xb, zb) in zip(coords, coords[1:]):
                if min(xa, xb) - 1e-8 <= x <= max(xa, xb) + 1e-8 and abs(xb - xa) > 1e-8:
                    if za > .05 and zb > .05:
                        intersections.append(float(za + (zb - za) * (x - xa) / (xb - xa)))
            return max(intersections) if intersections else None
        shell_top = [top_at(x) for x in (lo, hi)]
        if any(z is None for z in shell_top) or max(abs(float(trace[j, 1]) - shell_top[j]) for j in range(2)) > endpoint_tolerance:
            continue
        # Matching eaves alone do not authorize replacing a measured shell
        # ridge with a source parapet triangle. Keep the mapped shell whenever
        # it has real interior rise above its endpoint chord.
        shell_vertex_x = [float(x) for x, _ in shell_domain.exterior.coords if lo < x < hi]
        probe_x = sorted({lo, hi, *[float(x) for x in pts[:, 0] if lo < x < hi], *shell_vertex_x})
        shell_samples = [(x, top_at(x)) for x in probe_x]
        if any(z is None for _, z in shell_samples):
            continue
        chord_error = max(z - (shell_top[0] + (shell_top[1]-shell_top[0])*(x-lo)/(hi-lo))
                          for x, z in shell_samples)
        if chord_error > .15:
            # A source outline may still describe trim, but only if its entire
            # profile matches the authoritative shell trace. Otherwise suppress
            # the obsolete fill/trim and leave a reviewable audit note.
            source_shell_error = max(abs(float(np.interp(x, pts[:, 0], source_local)) - z)
                                     for x, z in shell_samples)
            if source_shell_error > .15:
                return None, {index}, [f"parapets[{index}] gable outline disagrees with variable shell boundary; shell boundary retained"]
            return None, set(), []
        domain = wall_domain_from_top_samples(pts, (lo, hi), base_z)
        if domain is not None: return domain, {index}, []
    return None, set(), []
