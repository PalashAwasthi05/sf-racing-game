"""A continuous provisional roof shell from filtered LiDAR samples.

The top is triangulated as one clipped grid, so adjacent cells share vertices.
Only the footprint's exterior and hole boundaries receive skirt faces. This is
geometry continuity, not a claim that unsupported spans have been measured.
"""
import numpy as np
import trimesh
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points
from shapely.ops import triangulate
from shapely.ops import split as split_polygon


def _polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    else:
        for part in getattr(geometry, "geoms", ()):
            yield from _polygons(part)


def _polygonal_footprint(geometry):
    """Retain every area component and courtyard; discard zero-area remnants.

    Footprint clipping can return a GeometryCollection containing polygons and
    shared-edge lines. Shapely exposes no boundary for that collection, even
    though its polygonal area is valid roof input.
    """
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    from shapely.ops import unary_union
    polygons = [p for p in _polygons(geometry) if not p.is_empty and p.area > 0]
    return unary_union(polygons) if polygons else Polygon()


def _aggregate_cells(points, cell_m):
    """Return robust cell medians, their grid keys, and sample counts."""
    if not len(points):
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int), np.empty(0, int), np.zeros(2)
    origin = points[:, :2].min(axis=0)
    keys = np.floor((points[:, :2] - origin) / cell_m).astype(int)
    groups = {}
    for key, point in zip(map(tuple, keys), points):
        groups.setdefault(key, []).append(point)
    out, out_keys, counts = [], [], []
    for key, values in sorted(groups.items()):
        values = np.asarray(values)
        z = values[:, 2]
        med = np.median(z)
        mad = np.median(np.abs(z - med))
        keep = np.abs(z - med) <= max(.15, 3.5 * 1.4826 * mad)
        q = values[keep]
        out.append(np.r_[np.median(q[:, :2], axis=0), np.median(q[:, 2])])
        out_keys.append(key)
        counts.append(len(values))
    return np.asarray(out, dtype=float), np.asarray(out_keys, dtype=int), np.asarray(counts), origin


def _aggregate(points, cell_m):
    """Median one robust height sample per occupied XY cell."""
    return _aggregate_cells(points, cell_m)[0]


def _filter_cells(samples, keys, counts, cell_m, residual_m=1.8, neighbor_radius_m=1.25,
                  minimum_component_cells=6, minimum_component_points=10):
    """Drop unsupported height outliers while retaining connected roof levels."""
    if len(samples) < 4:
        return np.ones(len(samples), dtype=bool), 0, 0, 0
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    values = samples[:, 2]
    adjacency = [[] for _ in samples]
    for i, (kx, ky) in enumerate(keys):
        radius_cells = int(np.ceil(neighbor_radius_m / cell_m))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx == dy == 0 or np.hypot(dx, dy) * cell_m > neighbor_radius_m + 1e-9:
                    continue
                j = lookup.get((kx + dx, ky + dy))
                if j is not None:
                    adjacency[i].append(j)

    # Connected cells at similar height represent a real patch or step. A
    # lone cell several metres off its neighbours has no such spatial support.
    component = np.full(len(samples), -1, dtype=int)
    component_sizes, component_points = [], []
    label = 0
    for start in range(len(samples)):
        if component[start] >= 0:
            continue
        stack = [start]
        component[start] = label
        members = []
        while stack:
            i = stack.pop(); members.append(i)
            for j in adjacency[i]:
                if component[j] < 0 and abs(values[i] - values[j]) <= 1.5:
                    component[j] = label; stack.append(j)
        component_sizes.append(len(members))
        component_points.append(int(counts[members].sum()))
        label += 1
    supported_component = np.array([
        component_sizes[component[i]] >= minimum_component_cells
        and component_points[component[i]] >= minimum_component_points
        for i in range(len(samples))
    ])

    residual = np.zeros(len(samples), dtype=float)
    has_neighbors = np.zeros(len(samples), dtype=bool)
    for i, neighbors in enumerate(adjacency):
        if neighbors:
            residual[i] = values[i] - np.median(values[neighbors])
            has_neighbors[i] = True
    isolated = has_neighbors & (np.abs(residual) > residual_m) & ~supported_component
    keep = ~isolated
    retained_patches = sum(size >= minimum_component_cells and points >= minimum_component_points
                           for size, points in zip(component_sizes, component_points))
    return keep, int(isolated.sum()), int(np.sum(isolated & (residual < 0))), int(retained_patches)


def _limit_mesh_slope(xy, faces, z, eligible_vertices=None, maximum_grade=2.5, sweeps=24):
    """Limit extrapolated transitions; preserve measured, supported step edges."""
    if not len(faces):
        return z, 0., 0., 0, 0.
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
    before = float(np.max(np.abs(z[edges[:, 1]] - z[edges[:, 0]]) / np.maximum(lengths, 1e-9)))
    movable = (np.ones(len(xy), dtype=bool) if eligible_vertices is None
               else np.asarray(eligible_vertices, dtype=bool))
    eligible = movable[edges[:, 0]] | movable[edges[:, 1]]
    # Project only provisional heights onto the graph grade constraints. A
    # pairwise clipping sweep can oscillate between incompatible fixed anchors
    # and even increase the worst boundary tooth. The sparse L1 projection keeps
    # the measured anchors exact and changes the harmonic continuation minimally.
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    original = np.asarray(z, float)
    ids = np.flatnonzero(movable)
    if not len(ids) or not eligible.any():
        return original.copy(), before, before, 0, 0.
    index = np.full(len(xy), -1, int); index[ids] = np.arange(len(ids))
    active_edges = edges[eligible]; active_lengths = lengths[eligible]
    count = len(ids)
    def solve(minimax=False):
        rr, cc, vv, rhs = [], [], [], []
        row = 0
        for (a, b), length in zip(active_edges, active_lengths):
            for sign in (1., -1.):
                constant = 0.
                for vertex, coefficient in ((b, sign), (a, -sign)):
                    if movable[vertex]:
                        rr.append(row); cc.append(index[vertex]); vv.append(coefficient)
                    else:
                        constant += coefficient * original[vertex]
                if minimax:
                    rr.append(row); cc.append(2 * count); vv.append(-max(length, 1e-9))
                    rhs.append(-constant)
                else:
                    rhs.append(maximum_grade * length - constant)
                row += 1
        for j, vertex in enumerate(ids):
            for sign in (1., -1.):
                rr.extend((row, row)); cc.extend((j, count + j)); vv.extend((sign, -1.))
                rhs.append(sign * original[vertex]); row += 1
        columns = 2 * count + int(minimax)
        matrix = coo_matrix((vv, (rr, cc)), shape=(row, columns)).tocsr()
        objective = np.r_[np.zeros(count), np.full(count, 1. / count), [1.] if minimax else []]
        if minimax:
            objective[count:2 * count] *= 1e-5
        bounds = ([(float(original.min()), float(original.max()))] * count
                  + [(0., None)] * count + ([(maximum_grade, None)] if minimax else []))
        return linprog(objective, A_ub=matrix, b_ub=np.asarray(rhs), bounds=bounds, method='highs')
    result = solve()
    if not result.success:
        # A genuine measured step can make the requested grade impossible.
        # Find the smallest feasible unknown grade without moving either roof.
        result = solve(minimax=True)
    z = original.copy()
    if result.success:
        z[ids] = result.x[:count]
    grades = np.abs(z[edges[:, 1]] - z[edges[:, 0]]) / np.maximum(lengths, 1e-9)
    after = float(np.max(grades))
    eligible_after = float(np.max(grades[eligible])) if eligible.any() else 0.
    adjusted = np.any(np.abs(z[edges] - original[edges]) > 1e-7, axis=1)
    return z, before, after, int(adjusted.sum()), eligible_after


def _project_supported_planes(samples, radius_m=2.0, residual_m=.15, minimum_support=8):
    """Project samples onto locally supported robust planes, retaining ridges."""
    if len(samples) < minimum_support:
        return samples.copy(), {"candidateCenters": 0, "projectedSamples": 0}
    tree = cKDTree(samples[:, :2])
    projected = samples.copy()
    rng = np.random.default_rng(1741)
    candidate_centers = 0
    accepted = 0
    residuals = []
    for i, point in enumerate(samples):
        inds = np.asarray(tree.query_ball_point(point[:2], radius_m), dtype=int)
        if len(inds) < minimum_support:
            continue
        q = samples[inds]
        # Center the fit at the sample being projected to keep the intercept
        # stable for projected-coordinate city data.
        design = np.c_[q[:, :2] - point[:2], np.ones(len(q))]
        best = None
        best_inliers = None
        for _ in range(min(32, max(12, len(inds)))):
            pick = rng.choice(len(q), 3, replace=False)
            a = design[pick]
            if np.linalg.matrix_rank(a) < 3:
                continue
            coef = np.linalg.solve(a, q[pick, 2])
            err = np.abs(design @ coef - q[:, 2])
            inliers = err <= residual_m
            score = int(inliers.sum())
            if best is None or score > best:
                best, best_inliers = score, inliers
        candidate_centers += 1
        if best_inliers is None or best < minimum_support or best < .65 * len(q):
            continue
        coef, *_ = np.linalg.lstsq(design[best_inliers], q[best_inliers, 2], rcond=None)
        err = np.abs(design[best_inliers] @ coef - q[best_inliers, 2])
        p90 = float(np.quantile(err, .90))
        center_error = abs(float(coef[2] - point[2]))
        if p90 > .30 or center_error > .30:
            continue
        projected[i, 2] = float(coef[2])
        accepted += 1
        residuals.append(p90)
    return projected, {
        "candidateCenters": int(candidate_centers),
        "projectedSamples": int(accepted),
        "medianPlaneP90ResidualM": float(np.median(residuals)) if residuals else None,
        "neighborhoodRadiusM": float(radius_m),
        "planeInlierResidualM": float(residual_m),
    }


def _split_faces_at_points(xy, faces, vertex_map, points):
    """Insert boundary constraint endpoints into existing mesh edges."""
    for point in points:
        key = (round(float(point[0]), 9), round(float(point[1]), 9))
        if key in vertex_map:
            continue
        v = len(xy)
        vertex_map[key] = v
        xy.append(key)
        updated = []
        for face in faces:
            split = False
            for j in range(3):
                a, b, c = face[j], face[(j + 1) % 3], face[(j + 2) % 3]
                pa, pb = np.asarray(xy[a]), np.asarray(xy[b])
                edge = pb - pa
                denom = float(np.dot(edge, edge))
                if denom <= 1e-16:
                    continue
                t = float(np.dot(point - pa, edge) / denom)
                closest = pa + np.clip(t, 0, 1) * edge
                if 1e-8 < t < 1 - 1e-8 and np.linalg.norm(closest - point) <= 1e-7:
                    tri1 = [a, v, c]
                    tri2 = [v, b, c]
                    for tri in (tri1, tri2):
                        coords = np.asarray(xy)[tri]
                        u, w = coords[1] - coords[0], coords[2] - coords[0]
                        cross = u[0] * w[1] - u[1] * w[0]
                        updated.append(tri if cross > 0 else tri[::-1])
                    split = True
                    break
            if not split:
                updated.append(face)
        faces[:] = updated
    return xy, faces, vertex_map


def _split_mesh_at_ridge(xy, faces, z, vertex_map, ridge_line, host_zone, model):
    """Insert a validated plane-intersection breakline into the shared mesh."""
    from general_world.roof_planes import evaluate as evaluate_roof

    xy = [tuple(map(float, p)) for p in np.asarray(xy)]
    z_values = list(map(float, np.asarray(z)))
    vertex_map = dict(vertex_map)
    a, b, c = map(float, ridge_line)
    norm2 = a * a + b * b
    if norm2 <= 1e-16:
        return xy, np.asarray(faces, dtype=np.int64).tolist(), z_values, vertex_map, 0
    minx, miny = np.min(np.asarray(xy), axis=0)
    maxx, maxy = np.max(np.asarray(xy), axis=0)
    center = np.array([(minx + maxx) / 2., (miny + maxy) / 2.])
    closest = center - (a * center[0] + b * center[1] + c) / norm2 * np.array([a, b])
    direction = np.array([-b, a]) / np.sqrt(norm2)
    extent = 4. * max(maxx - minx, maxy - miny, 1.)
    ridge = LineString([closest - direction * extent, closest + direction * extent])

    zone_geometry = host_zone.get('geometry')
    if zone_geometry is None:
        zone_a = np.asarray(host_zone["a"], dtype=float)
        tangent = np.asarray(host_zone["tangent"], dtype=float)
        inward = np.asarray(host_zone["inward"], dtype=float)
        frontage_length = float(host_zone["length"])
        zone_depth = float(host_zone["depth"])

    def top_height(point, face):
        # New intersections lie on an original triangle edge. Barycentric
        # interpolation is a stable fallback for numerical corner cases.
        tri_xy = np.asarray([xy[i] for i in face], dtype=float)
        tri_z = np.asarray([z_values[i] for i in face], dtype=float)
        point = np.asarray(point, dtype=float)
        for j in range(3):
            p0, p1 = tri_xy[j], tri_xy[(j + 1) % 3]
            edge = p1 - p0
            denom = float(edge @ edge)
            if denom <= 1e-16:
                continue
            t = float((point - p0) @ edge / denom)
            if -1e-8 <= t <= 1. + 1e-8 and np.linalg.norm(point - (p0 + t * edge)) <= 1e-7:
                return float(tri_z[j] + np.clip(t, 0., 1.) * (tri_z[(j + 1) % 3] - tri_z[j]))
        matrix = np.vstack((tri_xy.T, np.ones(3)))
        weights = np.linalg.solve(matrix, np.r_[point, 1.])
        return float(weights @ tri_z)

    def vertex_for(point, source_face):
        key = (round(float(point[0]), 9), round(float(point[1]), 9))
        existing = vertex_map.get(key)
        if existing is not None:
            return existing
        index = len(xy)
        vertex_map[key] = index
        xy.append(key)
        height = top_height(point, source_face)
        if zone_geometry is not None:
            in_zone = zone_geometry.buffer(1e-8).covers(Point(point))
        else:
            rel = np.asarray(point, dtype=float) - zone_a
            u, depth = float(rel @ tangent), float(rel @ inward)
            in_zone = (-.05 <= u <= frontage_length + .05 and -1e-6 <= depth <= zone_depth + 1e-6)
        if in_zone:
            height = float(np.asarray(evaluate_roof(model, np.asarray(point).reshape(1, 2))).reshape(-1)[0])
        z_values.append(height)
        return index

    output_faces = []
    for original in np.asarray(faces, dtype=np.int64):
        poly = Polygon([xy[i] for i in original])
        try:
            pieces = list(split_polygon(poly, ridge).geoms)
        except (ValueError, RuntimeError):
            pieces = [poly]
        pieces = [piece for piece in pieces if piece.geom_type == "Polygon" and piece.area > 1e-12]
        if len(pieces) <= 1:
            output_faces.append(list(map(int, original)))
            continue
        for piece in pieces:
            try:
                tri_xy, tri_faces = trimesh.creation.triangulate_polygon(piece, engine="earcut")
                triangles = [tri_xy[tri] for tri in tri_faces]
            except (ValueError, RuntimeError):
                triangles = [np.asarray(t.exterior.coords)[:3] for t in triangulate(piece)
                             if piece.covers(t.representative_point()) and t.difference(piece).area < 1e-8]
            for triangle in triangles:
                tri = [vertex_for(p, original) for p in triangle]
                coords = np.asarray([xy[i] for i in tri])
                if Polygon(coords).exterior.is_ccw:
                    output_faces.append(tri)
                else:
                    output_faces.append(tri[::-1])
    return xy, output_faces, z_values, vertex_map, len(z_values) - len(z)


def _boundary_points(footprint, spacing):
    result = []
    for poly in _polygons(footprint):
        for ring in (poly.exterior, *poly.interiors):
            count = max(1, int(np.ceil(ring.length / spacing)))
            result.extend(ring.interpolate(i * ring.length / count).coords[0]
                          for i in range(count))
    return np.asarray(result, dtype=float).reshape((-1, 2))


def _register_frontage_line(footprint, raw_line, max_displacement=1.5):
    """Match a measured frontage to one continuous, tangent-compatible ring run."""
    raw_line = raw_line if isinstance(raw_line, LineString) else LineString(raw_line)
    raw_length = float(raw_line.length)
    report = {"status": "rejected", "rawLengthM": raw_length}
    if raw_length < .5:
        report["reason"] = "frontage-too-short"
        return None, report

    raw_count = max(9, int(np.ceil(raw_length / .2)) + 1)
    raw_xy = np.asarray([raw_line.interpolate(t, normalized=True).coords[0]
                         for t in np.linspace(0., 1., raw_count)])
    raw_direction = (raw_xy[-1] - raw_xy[0]) / raw_length
    candidates = []
    for poly in _polygons(footprint):
        for ring in (poly.exterior, *poly.interiors):
            ring_line = LineString(ring.coords)
            perimeter = float(ring_line.length)
            if perimeter <= .05:
                continue
            start = float(ring_line.project(Point(raw_xy[0])))
            end = float(ring_line.project(Point(raw_xy[-1])))
            for direction in (1., -1.):
                arc = ((end - start) % perimeter if direction > 0
                       else (start - end) % perimeter)
                if arc < .05 or arc / raw_length < .6 or arc / raw_length > 1.5:
                    continue
                n = max(5, int(np.ceil(arc / .2)) + 1)
                distances = (start + direction * np.linspace(0., arc, n)) % perimeter
                coords = np.asarray([ring_line.interpolate(float(s)).coords[0] for s in distances])
                candidate_line = LineString(coords)
                displacement = np.asarray([candidate_line.distance(Point(p)) for p in raw_xy])
                segment_vec = np.diff(coords, axis=0)
                segment_len = np.linalg.norm(segment_vec, axis=1)
                valid_seg = segment_len > 1e-8
                alignment = np.zeros(len(segment_vec))
                alignment[valid_seg] = (segment_vec[valid_seg] @ raw_direction) / segment_len[valid_seg]
                # A kink may turn briefly, but the run must follow the measured
                # facade tangent for most of its length.
                tangent_fraction = float(segment_len[valid_seg & (alignment >= np.cos(np.deg2rad(35)))].sum()
                                         / max(segment_len[valid_seg].sum(), 1e-9))
                max_residual = float(displacement.max())
                mean_residual = float(displacement.mean())
                ratio = arc / raw_length
                score = mean_residual + .25 * max_residual + .5 * abs(np.log(ratio))
                candidates.append((score, candidate_line, {
                    "status": "accepted" if max_residual <= max_displacement and tangent_fraction >= .72 else "rejected",
                    "reason": None if max_residual <= max_displacement and tangent_fraction >= .72
                    else ("frontage-too-far-from-boundary" if max_residual > max_displacement
                          else "frontage-tangent-mismatch"),
                    "rawLengthM": raw_length, "registeredLengthM": arc,
                    "lengthRatio": float(ratio), "rawToBoundaryMeanM": mean_residual,
                    "rawToBoundaryMaxM": max_residual, "tangentAgreementFraction": tangent_fraction,
                    "registeredResidualM": 0.0, "maximumAcceptedDisplacementM": max_displacement,
                    "minimumTangentAgreementFraction": .72,
                }))
    if not candidates:
        report["reason"] = "no-continuous-boundary-run"
        return None, report
    _, line, best_report = min(candidates, key=lambda item: item[0])
    if best_report["status"] != "accepted":
        return None, best_report
    return line, best_report


def _host_fit_samples(samples, footprint, spec):
    """Exclude observed roof components from host fitting, not shell sampling.

    A strip's normal and planeV define its frontage line; u0/u1 bound the
    perpendicular frame coordinate and maxDepthM bounds its inward reach.
    """
    samples = np.asarray(samples, dtype=float)
    excluded = np.zeros(len(samples), dtype=bool)
    strips = spec.get("excludeStrips", ()) if isinstance(spec, dict) else ()
    for strip in strips or ():
        try:
            normal = np.asarray(strip["normal"], dtype=float).reshape(2)
            norm = float(np.linalg.norm(normal))
            plane_v = float(strip["planeV"])
            u0, u1 = sorted((float(strip["u0"]), float(strip["u1"])))
            max_depth = float(strip.get("maxDepthM", 6.))
            if (not np.isfinite([*normal, plane_v, u0, u1, max_depth]).all()
                    or norm < 1e-8 or max_depth <= 0):
                continue
            normal /= norm
            tangent = np.array([-normal[1], normal[0]])
            representative = np.asarray(footprint.representative_point().coords[0], dtype=float)
            inward_sign = 1. if normal @ representative - plane_v >= 0 else -1.
            signed_depth = inward_sign * (samples[:, :2] @ normal - plane_v)
            along = samples[:, :2] @ tangent
            # Allow small registration error between a frontage trace and the
            # mapped wall while keeping the component's exclusion bounded.
            excluded |= ((along >= u0 - .2) & (along <= u1 + .2)
                         & (signed_depth >= -.5) & (signed_depth <= max_depth))
        except (KeyError, TypeError, ValueError):
            continue
    return samples[~excluded], int(excluded.sum())


def _frontage_inward(footprint, a, b):
    tangent = (np.asarray(b, float) - np.asarray(a, float))
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    inward = np.array([-tangent[1], tangent[0]])
    midpoint = (np.asarray(a, float) + np.asarray(b, float)) / 2
    representative = np.asarray(footprint.representative_point().coords[0], float)
    if float((representative - midpoint) @ inward) < 0:
        inward = -inward
    return inward


def _fit_front_gable(footprint, points, xy, z, base_z, spec, lines, cell_m):
    """Fit two measured roof slopes for a gable whose ridge runs inward.

    The fitted field is a pair of planes meeting on a frontage-normal ridge.
    Ownership is limited to contiguous two-sided LiDAR support and ends at a
    measured residual changepoint; no ridge-crossing heuristic is used.
    """
    from scipy.optimize import least_squares
    from general_world.roof_planes import evaluate as evaluate_roof

    rect = footprint.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return z, {"status": "front-gable-needs-rectangular-orientation"}
    corners = np.asarray(rect.exterior.coords, dtype=float)[:4]
    edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    frontage = None
    for line, a, b in sorted(lines, key=lambda item: item[0].length, reverse=True):
        tangent = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        best = None
        for edge_id in range(4):
            ea, eb = corners[edge_id], corners[(edge_id + 1) % 4]
            eu = (eb - ea) / max(float(np.linalg.norm(eb - ea)), 1e-9)
            alignment = abs(float(tangent @ eu))
            offset = float(LineString([ea, eb]).distance(Point((a + b) / 2)))
            ratio = line.length / max(edge_lengths[edge_id], .1)
            score = 1. - alignment + .15 * offset + .1 * abs(np.log(max(ratio, 1e-6)))
            if best is None or score < best[0]:
                best = (score, edge_id, alignment, offset, edge_lengths[edge_id], ratio)
        _, edge_id, alignment, offset, edge_len, ratio = best
        if alignment >= np.cos(np.deg2rad(25.)) and offset <= 1.5 and .55 <= ratio <= 1.5:
            frontage = (line, np.asarray(a), np.asarray(b), tangent, edge_id, offset)
            break
    if frontage is None:
        return z, {"status": "frontage-not-gable-end-orientation"}
    line, a, b, tangent, edge_id, offset = frontage
    length = float(np.linalg.norm(b - a))
    inward = _frontage_inward(footprint, a, b)
    cells = _aggregate(points, cell_m)
    if len(cells) < 30:
        return z, {"status": "insufficient-front-gable-cells", "sampleCells": int(len(cells))}
    rel = cells[:, :2] - a
    u, depth = rel @ tangent, rel @ inward
    in_span = (u >= -.15) & (u <= length + .15)
    # This split is only an initial coverage balance check. The ridge location
    # is subsequently estimated from the training data, not fixed at midspan.
    left = u < length / 2.
    right = ~left
    depth_bin = np.floor(depth / cell_m).astype(int)
    bin_counts = {}
    for k in np.unique(depth_bin[in_span & (depth >= .35)]):
        m = in_span & (depth_bin == k)
        bin_counts[int(k)] = (int(m.sum()), int((m & left).sum()), int((m & right).sum()))
    # A contiguous run needs actual occupied cells on both gable slopes.
    first_bin = int(np.floor(.35 / cell_m))
    good_bins = []
    for k in range(first_bin, max(bin_counts.keys(), default=first_bin - 1) + 1):
        count, nl, nr = bin_counts.get(k, (0, 0, 0))
        if count < 5 or nl < 2 or nr < 2:
            if good_bins:
                break
            continue
        good_bins.append(k)
    if not good_bins:
        return z, {"status": "no-contiguous-two-slope-front-support", "depthBins": bin_counts}
    support_depth = float((good_bins[-1] + 1) * cell_m)
    # Deterministic spatial holdout by half-meter cells.
    hold = (np.floor((u + 2 * length) / cell_m).astype(int)
            + 2 * np.floor(depth / cell_m).astype(int)) % 4 == 0
    # Select the shallowest contiguous seed that has enough independent cells
    # on both sides. The cutoff depends on coverage only, never fit quality.
    seed_candidates = []
    for k in good_bins:
        depth_limit = float((k + 1) * cell_m)
        candidate_fit = in_span & (depth >= .35) & (depth <= depth_limit + 1e-9)
        candidate_train = candidate_fit & ~hold
        candidate_held = candidate_fit & hold
        counts = (int(candidate_fit.sum()), int(candidate_train.sum()), int(candidate_held.sum()),
                  int((candidate_train & left).sum()), int((candidate_train & right).sum()),
                  int((candidate_held & left).sum()), int((candidate_held & right).sum()))
        seed_candidates.append((depth_limit, candidate_fit, candidate_train, candidate_held, counts))
        if counts[0] >= 30 and counts[1] >= 20 and counts[2] >= 8 and min(counts[3:]) >= 4:
            break
    seed = next((item for item in seed_candidates if item[4][0] >= 30 and item[4][1] >= 20
                 and item[4][2] >= 8 and min(item[4][3:]) >= 4), None)
    if seed is None:
        best = max(seed_candidates, key=lambda item: item[4][0]) if seed_candidates else None
        return z, {"status": "insufficient-independent-front-gable-cells",
                   "contiguousSupportDepthM": support_depth,
                   "bestSeedDepthM": best[0] if best else 0.,
                   "bestSeedCellCounts": list(best[4]) if best else []}
    fit_depth, fit, train, held, seed_counts = seed
    if train.sum() < 20 or held.sum() < 8:
        return z, {"status": "insufficient-independent-front-gable-cells",
                   "trainCells": int(train.sum()), "heldOutCells": int(held.sum())}
    train_u, train_d, y = u[train], depth[train], cells[train, 2]
    # Fit a continuous broken line with a free ridge coordinate. Multi-start
    # keeps the non-smooth hinge objective from inheriting a center-ridge prior.
    def predict(q, uu, dd):
        du = uu - q[4]
        return q[0] + q[1] * np.minimum(du, 0.) + q[2] * np.maximum(du, 0.) + q[3] * dd

    lower_z, upper_z = float(np.min(y) - 5.), float(np.max(y) + 5.)
    lower = np.array([lower_z, .15, -2.5, -.6, .15 * length])
    upper = np.array([upper_z, 2.5, -.15, .6, .85 * length])
    candidates = []
    for ridge_start in (.3 * length, .5 * length, .7 * length):
        initial = np.array([float(np.median(y)), .6, -.6, 0., ridge_start])
        candidate = least_squares(lambda q: predict(q, train_u, train_d) - y,
                                  initial, bounds=(lower, upper), loss="soft_l1",
                                  f_scale=.15, max_nfev=700)
        candidates.append(candidate)
    fit_result = min(candidates, key=lambda item: float(np.sum(
        2. * .15**2 * (np.sqrt(1. + ((predict(item.x, train_u, train_d) - y) / .15)**2) - 1.))))
    params = fit_result.x
    ridge_u = float(params[4])
    model_left = u < ridge_u
    model_right = ~model_left
    if not (.15 <= params[1] <= 2.5 and -2.5 <= params[2] <= -.15 and abs(params[3]) <= .6):
        return z, {"status": "front-gable-fit-slope-out-of-range", "slopes": params[1:4].tolist(),
                   "candidateRidgeU": ridge_u}
    residual = cells[:, 2] - predict(params, u, depth)
    held_error = np.abs(residual[held])
    left_hold = held & model_left; right_hold = held & model_right
    if not left_hold.any() or not right_hold.any():
        return z, {"status": "front-gable-heldout-does-not-cover-both-slopes",
                   "candidateRidgeU": ridge_u, "heldOutCells": int(held.sum())}
    held_side = [float(np.mean(np.abs(residual[q]) <= .30)) for q in (left_hold, right_hold)]
    held_fraction = float(np.mean(held_error <= .30))
    held_p95 = float(np.quantile(held_error, .95))
    fit_inliers = fit & (np.abs(residual) <= .30)
    fit_side = [float(np.mean(fit_inliers[q])) for q in (fit & model_left, fit & model_right)]
    if (held_fraction < .80 or held_p95 > .30 or min(held_side) < .70 or min(fit_side) < .80):
        return z, {"status": "front-gable-independent-validation-failed", "fitCells": int(fit.sum()),
                   "heldOutCells": int(held.sum()), "heldOutFractionWithin0_30M": held_fraction,
                   "heldOutP95M": held_p95, "heldOutPerSlopeInlierFraction": held_side,
                   "fitPerSlopeInlierFraction": fit_side,
                   "candidateRidgeZ": float(params[0]),
                   "candidateSlopes": [float(params[1]), float(params[2])],
                   "candidateAlongDepthSlope": float(params[3]),
                   "candidateRidgeU": ridge_u,
                   "contiguousSupportDepthM": support_depth,
                   "fitDepthM": fit_depth, "frontageLengthM": length,
                   "frontageBoundaryOffsetM": offset, "seedCellCounts": list(seed_counts)}

    ridge_z, slope_left, slope_right, depth_slope = map(float, params[:4])
    eave_left = ridge_z - slope_left * ridge_u
    eave_right = ridge_z + slope_right * (length - ridge_u)
    anchor_uncertainty = float(spec.get("anchorUncertaintyM", .8))
    prior_residual = [eave_left - float(spec["eaveZ"]), eave_right - float(spec["eaveZ"])]
    anchor_role = spec.get("anchorRole")
    eave_prior_valid = (anchor_role in (None, "eave")
                        and spec.get("eaveConfidence", "medium") in ("medium", "high"))
    if eave_prior_valid and max(map(abs, prior_residual)) > max(0., anchor_uncertainty) + .2:
        return z, {"status": "front-gable-eave-prior-conflict", "fittedCornerEaveZ": [eave_left, eave_right],
                   "anchorZ": float(spec["eaveZ"]), "anchorUncertaintyM": anchor_uncertainty,
                   "anchorRole": anchor_role or "eave", "cornerPriorResidualM": prior_residual,
                   "heldOutP95M": held_p95,
                   "ridgeU": ridge_u}

    # Use deeper returns to find the first sustained, spatially broad departure
    # from the locally validated host planes (a real rear step/component).
    zone_depth = support_depth
    bad = []
    for k in good_bins:
        m = in_span & (depth_bin == k)
        if m.sum() < 5:
            break
        fraction = float(np.mean(np.abs(residual[m]) <= .30))
        median = float(np.median(residual[m]))
        side_stats = []
        for side_mask in (model_left, model_right):
            side = m & side_mask
            side_stats.append({
                "support": int(side.sum()),
                "inlierFraction": float(np.mean(np.abs(residual[side]) <= .30)) if side.any() else 0.,
                "medianResidualM": float(np.median(residual[side])) if side.any() else 0.,
            })
        bad.append({"bin": int(k), "inlierFraction": fraction, "medianResidualM": median,
                    "slopes": side_stats})
    changepoint = None
    # A single slope can enter a different roof regime while the ridge and
    # opposite slope remain compatible. Detect that independently before
    # using the whole-cross-section fallback below.
    for i in range(len(bad) - 1):
        one, two = bad[i], bad[i + 1]
        if two["bin"] != one["bin"] + 1:
            continue
        for side_index, side_name in enumerate(("left", "right")):
            first, second = one["slopes"][side_index], two["slopes"][side_index]
            if (first["support"] >= 2 and second["support"] >= 2
                    and first["inlierFraction"] < .65 and second["inlierFraction"] < .65
                    and abs(first["medianResidualM"]) > .30
                    and abs(second["medianResidualM"]) > .30
                    and np.sign(first["medianResidualM"]) == np.sign(second["medianResidualM"])):
                changepoint = {"slope": side_name, "startBin": one["bin"],
                               "endBin": two["bin"],
                               "medianResidualsM": [first["medianResidualM"],
                                                    second["medianResidualM"]]}
                zone_depth = max(.5, one["bin"] * cell_m)
                break
        if changepoint is not None:
            break
    # Preserve the previous cross-section-wide signal for changes that affect
    # both slopes similarly or do not isolate cleanly to a fitted side.
    if changepoint is None:
        for i in range(len(bad) - 1):
            one, two = bad[i], bad[i + 1]
            if (one["inlierFraction"] < .65 and two["inlierFraction"] < .65
                    and abs(one["medianResidualM"]) > .45
                    and abs(two["medianResidualM"]) > .45
                    and np.sign(one["medianResidualM"]) == np.sign(two["medianResidualM"])):
                zone_depth = max(.5, one["bin"] * cell_m)
                changepoint = {"slope": "whole-cross-section", "startBin": one["bin"],
                               "endBin": two["bin"]}
                break
    if zone_depth <= .5:
        return z, {"status": "front-gable-support-too-shallow"}

    slope_l_xy = tangent * slope_left + inward * depth_slope
    slope_r_xy = tangent * slope_right + inward * depth_slope
    model = {"planes": [
        [*slope_l_xy, ridge_z - slope_left * ridge_u - float(slope_l_xy @ a)],
        [*slope_r_xy, ridge_z + slope_right * (length - ridge_u) - slope_right * length
         - float(slope_r_xy @ a)]], "operation": "min"}
    ridge_coefficients = np.asarray(model["planes"][0], dtype=float) - np.asarray(model["planes"][1], dtype=float)
    predicted = np.asarray(evaluate_roof(model, xy), dtype=float).reshape(-1)
    vu, vd = (xy - a) @ tangent, (xy - a) @ inward
    zone = (vu >= -.05) & (vu <= length + .05) & (vd >= -.05) & (vd <= zone_depth + 1e-6)
    out = np.asarray(z, dtype=float).copy()
    out[zone] = np.maximum(base_z, predicted[zone])
    return out, {"status": "applied-observed-front-gable", "kind": "gable", "frontageKind": "gable-end",
                 "application": "measured-frontage-zone", "frontageRectangleEdge": int(edge_id),
                 "frontageLengthM": length, "frontageBoundaryOffsetM": offset,
                 "zoneDepthM": float(zone_depth), "zoneDepthSource": "contiguous-support-to-measured-changepoint",
                 "ridgeZ": ridge_z, "ridgeSource": "robust-two-slope-lidar-fit",
                 "ridgeUFromFrontageStartM": ridge_u,
                 "fittedCornerEaveZ": [eave_left, eave_right], "anchorZ": float(spec["eaveZ"]),
                 "anchorUncertaintyM": anchor_uncertainty, "cornerPriorResidualM": prior_residual,
                 "anchorRole": anchor_role or "eave", "cornerPriorUsedAsEave": eave_prior_valid,
                 "seedDepthM": float(fit_depth), "seedCellCounts": list(seed_counts),
                 "leftSlope": slope_left, "rightSlope": slope_right, "ridgeAlongDepthSlope": depth_slope,
                 "fitCellCount": int(fit.sum()), "fitInlierFractionBySlope": fit_side,
                 "heldOutCellCount": int(held.sum()), "heldOutFractionWithin0_30M": held_fraction,
                 "heldOutP95M": held_p95, "heldOutFractionWithin0_30MBySlope": held_side,
                 "depthBins": {str(k): list(v) for k, v in bin_counts.items()},
                 "depthResidualAudit": bad, "depthChangePoint": changepoint, "model": model,
                 "_ridgeLine": ridge_coefficients.tolist(),
                 "_hostZone": {"a": a.tolist(), "tangent": tangent.tolist(),
                               "inward": inward.tolist(), "length": length,
                               "depth": float(zone_depth)},
                 "_ridgeModel": model}


def _observed_host(footprint, samples, xy, z, base_z, spec, constraints, cell_m, raw_samples=None):
    """Apply a photo-read host form only in the frontage zone it can support.

    The segment gives a measured roof boundary anchor; the transition depth is
    inferred by walking inward to the model's first ridge-height crossing. It
    is audited as inferred, not presented as a measured roof extent.
    """
    audit = {"status": "not-proposed"}
    if not isinstance(spec, dict) or spec.get("kind") not in ("hip", "gable", "mansard"):
        return z, audit
    try:
        eave_z = float(spec["eaveZ"])
    except (KeyError, TypeError, ValueError):
        return z, {"status": "incomplete-host-spec"}
    ridge_value = spec.get("ridgeZ")
    ridge_z = None
    if ridge_value is not None:
        try:
            ridge_z = float(ridge_value)
        except (TypeError, ValueError):
            return z, {"status": "invalid-host-ridge"}
    if (spec.get("confidence") not in ("medium", "high") or not np.isfinite(eave_z)
            or (ridge_z is not None and (not np.isfinite(ridge_z) or ridge_z <= eave_z + .25))):
        return z, {"status": "low-confidence-or-invalid-host-spec"}
    if not constraints:
        return z, {"status": "no-matched-frontage-boundary-anchor"}

    # Explicit dormer/tower observations may contaminate host roof cells. Keep
    # these returns in the reconstructed shell, but omit them from host fitting.
    fit_source = samples if raw_samples is None else _aggregate(np.asarray(raw_samples, dtype=float), cell_m)
    host_samples, excluded_host_samples = _host_fit_samples(fit_source, footprint, spec)

    # Reuse the architectural plane construction, but evaluate it on the shell's
    # existing vertices: there is one top field and therefore no patch skirts.
    from .roof_forms import build as build_roof_form
    from general_world.roof_planes import evaluate as evaluate_roof
    lines = []
    for item in constraints:
        try:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], LineString):
                line = item[0]
                a = np.asarray(line.coords[0], float); b = np.asarray(line.coords[-1], float)
            else:
                a = np.asarray(item["a"], float); b = np.asarray(item["b"], float)
                raw_line = LineString([a, b])
                registered, _ = _register_frontage_line(footprint, raw_line)
                if registered is None:
                    continue
                # Preserve the measured straight photo frontage for its normal;
                # the registered boundary run is for perimeter ownership only.
                line = raw_line
            if line.length > .05 and np.isfinite([*a, *b]).all():
                # A registered frontage can follow a real footprint kink. Use
                # its locally straight runs for inward-normal fit windows;
                # one chord through the whole run may leave a concavity.
                simplified = line.simplify(.1, preserve_topology=False)
                coords = np.asarray(simplified.coords, dtype=float)
                for p0, p1 in zip(coords[:-1], coords[1:]):
                    if np.linalg.norm(p1 - p0) >= .5:
                        lines.append((LineString([p0, p1]), p0, p1))
        except (KeyError, TypeError, ValueError):
            continue
    if not lines:
        return z, {"status": "no-matched-frontage-boundary-anchor"}

    if spec.get("frontageKind") == "gable-end":
        fit_source = samples if raw_samples is None else np.asarray(raw_samples, dtype=float)
        fit_source, excluded = _host_fit_samples(fit_source, footprint, spec)
        out, gable_audit = _fit_front_gable(footprint, fit_source, xy, z, base_z, spec, lines, cell_m)
        gable_audit["excludedHostFitSampleCount"] = int(excluded)
        return out, gable_audit

    form_spec = {"kind": spec["kind"], "riseM": 1. if ridge_z is None else ridge_z - eave_z,
                 "overhangM": 0., "pitchDeg": spec.get("pitchDeg", 72.)}
    patches, form_report = build_roof_form(footprint, form_spec, eave_z, eave_z - base_z)
    if not patches:
        return z, {"status": "host-form-unavailable", "form": form_report}
    model = patches[0]["model"]

    # When the crest is occluded, solve only the scale (rise) of the observed
    # form from the frontage returns. The unit-rise plane envelope supplies the
    # geometric basis; a robust median ratio ignores sparse elevated fixtures.
    inferred_ridge = ridge_z is None
    if inferred_ridge:
        bounds = footprint.bounds
        fit_depth = min(5., float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])))
        picked = np.zeros(len(host_samples), bool)
        for line, a, b in lines:
            inward = _frontage_inward(footprint, a, b)
            vec = b - a; length2 = float(vec @ vec); rel = host_samples[:, :2] - a
            t = rel @ vec / length2; depth = rel @ inward
            picked |= ((t >= -.15 / np.sqrt(length2)) & (t <= 1 + .15 / np.sqrt(length2))
                       & (depth >= .7) & (depth <= fit_depth))
        fit_samples = host_samples[picked]
        basis = np.asarray(evaluate_roof(model, fit_samples[:, :2]), float).reshape(-1) - eave_z
        dz = fit_samples[:, 2] - eave_z
        usable = (basis > .08) & (dz > -.5) & (dz < 7.)
        ratios = dz[usable] / basis[usable]
        ratios = ratios[np.isfinite(ratios) & (ratios >= .4) & (ratios <= 6.)]
        if len(ratios) < 8:
            return z, {"status": "insufficient-frontal-support-for-inferred-ridge",
                       "fitSampleCells": int(len(ratios)), "fitWindowM": fit_depth,
                       "form": form_report}
        rise = float(np.median(ratios))
        ridge_z = eave_z + rise
        # Rebuild the exact scaled architectural model now that LiDAR has
        # supplied a bounded, robust estimate of its rise.
        form_spec["riseM"] = rise
        patches, form_report = build_roof_form(footprint, form_spec, eave_z, eave_z - base_z)
        if not patches:
            return z, {"status": "inferred-host-form-unavailable", "form": form_report}
        model = patches[0]["model"]
        inferred_residual = fit_samples[:, 2] - np.asarray(evaluate_roof(model, fit_samples[:, :2]), float).reshape(-1)
        inliers = np.abs(inferred_residual) <= .35
        compatible = (inferred_residual >= -.65) & (inferred_residual <= 1.2)
        median_inlier_residual = (float(np.median(np.abs(inferred_residual[inliers])))
                                  if inliers.any() else float("inf"))
        if inliers.mean() < .30 or compatible.mean() < .72 or median_inlier_residual > .35:
            return z, {"status": "frontal-returns-do-not-support-inferred-host-form",
                       "fitSampleCells": int(len(fit_samples)), "inlierFraction": float(inliers.mean()),
                       "compatibleFraction": float(compatible.mean()), "inferredRiseM": rise,
                       "medianInlierResidualM": median_inlier_residual,
                       "fitWindowM": fit_depth, "form": form_report}
    model_z = np.asarray(evaluate_roof(model, xy), dtype=float).reshape(-1)

    bounds = footprint.bounds
    max_walk = float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    frontage_zones = []
    from shapely import contains_xy
    covered_footprint = footprint.buffer(1e-5)
    for line, a, b in lines:
        inward = _frontage_inward(footprint, a, b)
        # A frontage midpoint need not align with the ridge of a hip (a bay or
        # partial facade shifts it sideways). Search the actual frontage span,
        # rather than declaring the measured roof impossible on that one ray.
        frontage_positions = np.linspace(0., 1., max(3, int(np.ceil(line.length / .25)) + 1))
        frontage_points = a[None, :] + frontage_positions[:, None] * (b - a)[None, :]
        # First modeled crossing of the observed ridge height, bounded to the
        # actual footprint. No arbitrary building-style depth is inserted.
        depth = None
        steps = max(2, int(np.ceil(max_walk / max(.1, cell_m * .25))))
        entered_footprint = False
        for d in np.linspace(0., max_walk, steps):
            query = frontage_points + inward * d
            on_roof = contains_xy(covered_footprint, query[:, 0], query[:, 1])
            if not on_roof.any():
                if entered_footprint:
                    break
                continue
            entered_footprint = True
            qz = float(np.max(evaluate_roof(model, query[on_roof])))
            if qz >= ridge_z - .12:
                depth = float(d); break
        if depth is not None and depth > .05:
            frontage_zones.append((line, a, b, inward, depth))
    if not frontage_zones:
        return z, {"status": "ridge-not-reached-from-observed-eave", "form": form_report}

    # Score the form on cells between the observed eave and its inferred ridge.
    # Positive sparse residuals are roof fixtures; a broad positive structure or
    # a surface below the photographed host form invalidates the hypothesis.
    point_xy = host_samples[:, :2]
    eligible_sample = np.zeros(len(host_samples), bool)
    for line, a, b, inward, depth in frontage_zones:
        v = b - a; l2 = float(v @ v)
        rel = point_xy - a
        t = rel @ v / l2
        side = rel @ inward
        eligible_sample |= ((t >= -.15 / np.sqrt(l2)) & (t <= 1 + .15 / np.sqrt(l2))
                            & (side >= -.15) & (side <= depth + .4))
    candidate_samples = host_samples[eligible_sample]
    if len(candidate_samples) < 8:
        return z, {"status": "insufficient-host-zone-samples", "sampleCells": int(len(candidate_samples)),
                   "form": form_report}
    predicted = np.asarray(evaluate_roof(model, candidate_samples[:, :2]), float).reshape(-1)
    residual = candidate_samples[:, 2] - predicted
    # Sparse positive returns (dormers/chimneys) are tolerated. Broad tall blocks,
    # unsupported undercuts, or a model that misses the majority of the host fail.
    compatible = (residual >= -.65) & (residual <= 1.2)
    supported_fraction = float(compatible.mean())
    inlier = np.abs(residual) <= .65
    inlier_fraction = float(inlier.mean())
    median_inlier = float(np.median(np.abs(residual[inlier]))) if inlier.any() else None
    if supported_fraction < .72 or inlier_fraction < .35:
        return z, {"status": "host-form-contradicted-by-roof-returns", "sampleCells": int(len(candidate_samples)),
                   "compatibleFraction": supported_fraction, "inlierFraction": inlier_fraction,
                   "medianInlierResidualM": median_inlier, "form": form_report}

    # A regular outline and a model supported across its measured roof can use
    # one continuous host form across the complete footprint. Otherwise keep
    # the inferred form only in the observed frontage-to-ridge zone.
    rect = footprint.minimum_rotated_rectangle
    regular = rect.geom_type == "Polygon" and footprint.area >= .75 * rect.area
    all_pred = np.asarray(evaluate_roof(model, host_samples[:, :2]), float).reshape(-1)
    all_residual = host_samples[:, 2] - all_pred
    global_compatible = (all_residual >= -.65) & (all_residual <= 1.2)
    global_inliers = np.abs(all_residual) <= .65
    full_host_supported = (regular and len(host_samples) >= 30 and global_compatible.mean() >= .72
                           and global_inliers.mean() >= .35)
    if full_host_supported:
        out = model_z.copy()
        application = "whole-regular-footprint"
    else:
        application = "observed-frontage-zone"
        out = z.copy(); weights = np.zeros(len(xy), float)
        for line, a, b, inward, depth in frontage_zones:
            v = b - a; l2 = float(v @ v); rel = xy - a
            t = rel @ v / l2; side = rel @ inward
            zone = ((t >= -.15 / np.sqrt(l2)) & (t <= 1 + .15 / np.sqrt(l2))
                    & (side >= -.05) & (side <= depth + cell_m))
            fade = np.clip((depth + cell_m - side) / max(cell_m, .05), 0., 1.)
            weights = np.maximum(weights, np.where(zone, fade, 0.))
        out = out * (1. - weights) + model_z * weights
    out = np.maximum(out, base_z)
    audit = {"status": "applied-photo-host-form", "kind": spec["kind"],
             "confidence": spec["confidence"], "anchorZ": eave_z,
             "anchorRole": spec.get("anchorRole", "eave"), "ridgeZ": ridge_z,
             "ridgeSource": "inferred-frontal-lidar" if inferred_ridge else "observed",
             "inferredRiseM": float(ridge_z - eave_z) if inferred_ridge else None,
             "application": application,
             "zoneDepthM": [float(q[4]) for q in frontage_zones],
             "zoneDepthSource": ("inferred-first-ridge-crossing-from-matched-eave"
                                 if spec.get("anchorRole", "eave") == "eave"
                                 else "inferred-first-ridge-crossing-from-matched-visible-roof-edge"),
             "sampleCells": int(len(candidate_samples)), "compatibleFraction": supported_fraction,
             "excludedHostFitSampleCount": excluded_host_samples,
             "inlierFraction": inlier_fraction, "medianInlierResidualM": median_inlier,
             "form": form_report}
    if full_host_supported:
        # Private handoff to the mesh builder. Removed before audit serialization.
        audit["_analyticModel"] = model
    return out, audit


def _analytic_roof_components(footprint, model, base_z, spacing):
    """Extract exact plane-envelope top and perimeter walls from one closed solid."""
    from general_world.roof_planes import evaluate as evaluate_roof, roof_solid

    solid = roof_solid(footprint, model, base_z)
    faces = np.asarray(solid.faces, dtype=np.int64)
    centers = np.asarray(solid.triangles_center, dtype=float)
    expected = np.asarray(evaluate_roof(model, centers[:, :2]), dtype=float).reshape(-1)
    on_envelope = np.abs(centers[:, 2] - expected) <= 1e-4
    # Outward-facing top facets have positive vertical normals; this includes
    # steep mansard faces while excluding the underside and vertical perimeter.
    top_mask = on_envelope & (np.asarray(solid.face_normals)[:, 2] > 1e-7)
    bottom_mask = np.max(np.abs(solid.vertices[faces, 2] - float(base_z)), axis=1) <= 1e-6
    wall_mask = ~(top_mask | bottom_mask)
    if not top_mask.any():
        raise ValueError("analytic roof solid could not be partitioned into a top surface")
    surface = trimesh.Trimesh(vertices=np.asarray(solid.vertices).copy(), faces=faces[top_mask], process=False)
    # If the eave lies exactly on base_z, the solid has no vertical perimeter
    # faces: roof and bottom meet along the boundary. This is a valid zero-height
    # skirt, not a failed partition.
    skirt = trimesh.Trimesh(vertices=np.asarray(solid.vertices).copy(), faces=faces[wall_mask], process=False)
    surface.remove_unreferenced_vertices()
    skirt.remove_unreferenced_vertices()

    top_edges = np.sort(np.vstack((faces[top_mask][:, [0, 1]], faces[top_mask][:, [1, 2]],
                                   faces[top_mask][:, [2, 0]])), axis=1)
    edges, counts = np.unique(top_edges, axis=0, return_counts=True)
    boundary_edges = edges[counts == 1]
    samples = []
    for edge in boundary_edges:
        p0, p1 = solid.vertices[edge]
        if footprint.boundary.distance(LineString([p0[:2], p1[:2]])) > 1e-5:
            continue
        length = float(np.linalg.norm(p1[:2] - p0[:2]))
        count = max(1, int(np.ceil(length / max(spacing, .05))))
        samples.extend(p0 + (p1 - p0) * (j / count) for j in range(count))
    return surface, skirt, np.asarray(samples, dtype=float).reshape((-1, 3))


def _boundary_chains_from_samples(footprint, samples):
    """Rebuild ordered perimeter chains from an exact analytic roof boundary."""
    samples = np.asarray(samples, dtype=float).reshape((-1, 3))
    chains = []
    for poly in _polygons(footprint):
        oriented = orient(poly, sign=1.0)
        for ring_index, ring in enumerate((oriented.exterior, *oriented.interiors)):
            ringline = LineString(ring.coords)
            owned = [point for point in samples if ringline.distance(Point(point[:2])) <= 1e-6]
            owned.sort(key=lambda point: ringline.project(Point(point[:2])))
            compact = []
            for point in owned:
                if not compact or np.linalg.norm(point[:2] - compact[-1][:2]) > 1e-8:
                    compact.append(point.copy())
            if len(compact) >= 2:
                chains.append({"ring": "exterior" if ring_index == 0 else "hole",
                               "points": np.asarray(compact, dtype=float)})
    return chains


def _fit_structural_regions(footprint, samples, cell_m, source_feature_support=None):
    """Fit local planes, but grant ownership only to directly supported cells.

    ``fit_patches`` uses regularized neighboring labels to estimate a useful
    region outline. That outline is not itself evidence that the plane exists
    throughout it: only observed cells whose robust height agrees with the
    plane may become Dirichlet anchors in the shell.
    """
    from general_world.roof_planes import RoofPolicy, fit_patches, sample_cells
    from shapely.geometry import box
    from shapely.ops import unary_union

    policy = RoofPolicy(cell_m=cell_m, minimum_points_per_cell=1, minimum_cells=30,
                        minimum_coverage=.15, maximum_p95_error_m=.30,
                        minimum_explained_fraction=.80, maximum_planes=6,
                        minimum_interior_explained=.90)
    patches, report = fit_patches(footprint, samples, policy)
    if not patches:
        return None, {"status": "no-validated-structural-planes", "fit": report}
    cell_samples, _ = sample_cells(footprint, samples, policy)
    if not len(cell_samples):
        return None, {"status": "no-supported-plane-cells", "fit": report}
    origin = np.asarray(footprint.bounds[:2], dtype=float)
    keys = np.floor((cell_samples[:, :2] - origin) / cell_m).astype(int)
    cell_boxes = [box(*(origin + key * cell_m), *(origin + (key + 1) * cell_m))
                  for key in keys]

    candidates = []
    for patch in patches:
        planes = patch["model"].get("planes", ())
        if len(planes) != 1:
            continue
        plane = np.asarray(planes[0], dtype=float).reshape(3)
        if np.isfinite(plane).all():
            candidates.append({"geometry": patch["geometry"].intersection(footprint),
                               "plane": plane,
                               "fitRegionAreaM2": float(patch["geometry"].intersection(footprint).area)})
    if not candidates:
        return None, {"status": "no-finite-structural-planes", "fit": report}

    # Reassign each occupied cell to the closest accepted plane, then keep it
    # only when it is an actual inlier. No joined/regularized blank cell can
    # silently become a measured roof anchor.
    design = np.c_[cell_samples[:, :2], np.ones(len(cell_samples))]
    planes = np.asarray([r["plane"] for r in candidates])
    residual = np.abs(design @ planes.T - cell_samples[:, 2, None])
    owner = np.argmin(residual, axis=1)
    best = residual[np.arange(len(owner)), owner]
    regions = []
    ownership_report = []
    for region_id, candidate in enumerate(candidates):
        eligible = (owner == region_id) & (best <= policy.residual_m)
        seed = eligible & np.asarray([candidate["geometry"].covers(Point(p)) for p in cell_samples[:, :2]])
        perimeter_strip = np.asarray([footprint.boundary.distance(Point(p)) <= max(.75, 1.5 * cell_m)
                                      for p in cell_samples[:, :2]])
        eligible &= seed | perimeter_strip
        inlier = _connected_inlier_cells(seed, eligible, keys)
        inlier, component_report = _retain_owned_components(
            inlier, keys, cell_samples, candidate['plane'], source_feature_support or ())
        seed &= inlier
        has_source_component = any(c['sourceSupported'] for c in component_report)
        if int(inlier.sum()) < 4 and not has_source_component:
            ownership_report.append({"region": region_id, "status": "dropped-insufficient-direct-inliers",
                                     "fitRegionAreaM2": candidate["fitRegionAreaM2"],
                                     "directInlierCellCount": int(inlier.sum()),
                                     "connectedComponents": component_report})
            continue
        support = unary_union([cell_boxes[i] for i in np.flatnonzero(inlier)])
        # A regularized patch outline is a fit hypothesis, not a veto on real
        # neighboring inliers. Recover occupied same-plane cells connected to
        # its validated support, including thin perimeter strips the label
        # morphology erased. Empty/mixed/outlier cells never gain ownership.
        old_geometry = candidate['geometry'].intersection(unary_union(
            [cell_boxes[i] for i in np.flatnonzero(seed)])).intersection(footprint)
        recovered = unary_union([cell_boxes[i] for i in np.flatnonzero(inlier & ~seed)])
        geometry = old_geometry.union(recovered).intersection(footprint)
        geometry = unary_union([p for p in _polygons(geometry) if p.area > 1e-10])
        if geometry.is_empty or geometry.area <= 1e-8:
            ownership_report.append({"region": region_id, "status": "dropped-empty-direct-support",
                                     "fitRegionAreaM2": candidate["fitRegionAreaM2"],
                                     "directInlierCellCount": int(inlier.sum())})
            continue
        regions.append({"geometry": geometry, "plane": candidate["plane"],
                        "inlierXY": cell_samples[inlier, :2].copy()})
        ownership_report.append({
            "region": region_id, "status": "owned-by-direct-inlier-cells",
            "fitRegionAreaM2": candidate["fitRegionAreaM2"],
            "directSupportAreaM2": float(geometry.area),
            "removedUnsupportedExpansionM2": float(candidate['geometry'].difference(geometry).area),
            "recoveredMeasuredSupportOutsideFitOutlineM2": float(geometry.difference(candidate['geometry']).area),
            "directInlierCellCount": int(inlier.sum()),
            "recoveredConnectedInlierCellsOutsideFitOutline": int(np.sum(inlier & ~seed)),
            "connectedComponents": component_report,
            "maximumCellResidualM": float(best[inlier].max()),
        })
    if not regions:
        return None, {"status": "no-finite-structural-planes", "fit": report}
    supported = unary_union([r["geometry"] for r in regions]).intersection(footprint)
    unresolved = footprint.difference(supported)
    candidates = []
    for poly in _polygons(unresolved):
        if not poly.is_empty:
            candidates.extend(np.asarray(poly.exterior.coords))
            candidates.append(np.asarray(poly.representative_point().coords[0]))
    continuation = (max((supported.distance(Point(p)) for p in candidates), default=0.)
                    if not supported.is_empty else float("inf"))
    seams = []
    for i, first in enumerate(regions):
        for j, second in enumerate(regions[i + 1:], i + 1):
            shared = first["geometry"].boundary.intersection(second["geometry"].boundary)
            if shared.is_empty or shared.length < .25:
                continue
            xy = []
            for line in getattr(shared, "geoms", [shared]):
                if line.geom_type == "LineString" and line.length > .05:
                    count = max(2, int(np.ceil(line.length / .5)) + 1)
                    xy.extend(line.interpolate(t, normalized=True).coords[0]
                              for t in np.linspace(0., 1., count))
            if xy:
                delta = np.c_[xy, np.ones(len(xy))] @ (first["plane"] - second["plane"])
                seams.append({"regions": [i, j], "sharedBoundaryLengthM": float(shared.length),
                              "heightDifferenceP90M": float(np.quantile(np.abs(delta), .90)),
                              "heightDifferenceMaxM": float(np.max(np.abs(delta)))})
    audit = {
        "status": "validated-local-structural-regions", "fit": report,
        "acceptedPlaneCount": len(regions), "validatedSupportAreaM2": float(supported.area),
        "unresolvedAreaM2": float(max(0., footprint.area - supported.area)),
        "unresolvedAreaFraction": float(max(0., footprint.area - supported.area) / max(footprint.area, 1e-9)),
        "maximumSampledContinuationDistanceM": float(continuation),
        "maximumAllowedContinuationDistanceM": float(max(2., 3. * cell_m)),
        "continuation": "harmonic interpolation from supported plane/eave anchors; no plane extrapolation into gaps",
        "planes": [r["plane"].tolist() for r in regions], "sharedSeams": seams,
        "supportOwnership": ownership_report,
        "arrangement": "single clipped grid with local plane ownership and shared vertex seams; no patch skirts",
        "representationLimit": "unsupported spans use provisional harmonic interpolation; roof steps are not inferred from plane disagreement alone",
    }
    return regions, audit


def _retain_owned_components(mask, keys, samples, plane, source_supports=(), minimum_cells=9):
    """Reapply fit_patches' measured-component rule after cell reassignment.

    A distant point matching a large plane is not a local roof component.
    Preserve small components only when source-verified bounded feature
    geometry and its height range independently support their actual cells.
    """
    import re
    mask = np.asarray(mask, bool)
    kept = np.zeros(len(mask), bool)
    lookup = {tuple(key): i for i, key in enumerate(keys) if mask[i]}
    pending = set(np.flatnonzero(mask))
    reports = []
    while pending:
        seed = pending.pop(); members = [seed]; stack = [seed]
        while stack:
            index = stack.pop(); x, y = keys[index]
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = lookup.get((x + dx, y + dy))
                if neighbor in pending:
                    pending.remove(neighbor); members.append(neighbor); stack.append(neighbor)
        supported_by_source = False
        if len(members) < minimum_cells:
            for source in source_supports:
                try:
                    proof = source.get('provenance') or {}
                    if proof.get('status') not in ('source-derived', 'source-verified', 'accepted-approximate'):
                        continue
                    if not all(re.fullmatch('[0-9a-f]{64}', str(proof.get(k, '')))
                               for k in ('sourceSha256', 'metadataSha256')):
                        continue
                    low, high = map(float, source['heightRangeZ'])
                    geometry = source['geometry']
                    if not np.isfinite([low, high]).all() or high <= low or geometry.is_empty:
                        continue
                    points = samples[members]
                    levels = np.c_[points[:, :2], np.ones(len(points))] @ plane
                    if (np.all((levels >= low) & (levels <= high))
                            and all(geometry.buffer(1e-8).covers(Point(p)) for p in points[:, :2])):
                        supported_by_source = True
                        break
                except (KeyError, TypeError, ValueError):
                    continue
        accepted = len(members) >= minimum_cells or supported_by_source
        if accepted:
            kept[members] = True
        reports.append({'cells': len(members), 'minimumConnectedCells': minimum_cells,
                        'status': ('retained-source-feature-support' if supported_by_source else
                                   'retained-measured-component' if accepted else 'unresolved-small-disconnected-island'),
                        'sourceSupported': supported_by_source})
    return kept, reports


def _raw_points_for_kept_cells(raw, cell_keys, keep, origin, cell_m):
    """Use the very same grid as measured-cell rejection, not a second origin."""
    raw_keys = np.floor((np.asarray(raw)[:, :2] - np.asarray(origin)) / cell_m).astype(int)
    accepted = {tuple(key) for key in np.asarray(cell_keys)[np.asarray(keep, bool)]}
    return np.asarray(raw)[np.asarray([tuple(key) in accepted for key in raw_keys], bool)]


def _connected_inlier_cells(seed, eligible, keys):
    """Grow only through occupied, measured inlier cells of the same plane."""
    seed, eligible = np.asarray(seed, bool), np.asarray(eligible, bool)
    result = seed.copy()
    lookup = {tuple(key): i for i, key in enumerate(np.asarray(keys, int)) if eligible[i]}
    pending = list(np.flatnonzero(seed))
    while pending:
        index = pending.pop()
        x, y = keys[index]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = lookup.get((x + dx, y + dy))
                if neighbor is not None and not result[neighbor]:
                    result[neighbor] = True
                    pending.append(neighbor)
    return result


def _recover_supported_ridges(footprint, regions, cell_m):
    """Interpolate a bounded crease between independently validated slopes.

    A sample-cell outline has many blank cells. Harmonic absolute heights in
    those gaps do not reproduce the fitted planes, so even a straight measured
    ridge can acquire teeth. Here both slope inlier clouds must bracket the
    same ridge segment. Only interpolation inside their convex hull and one
    cell diagonal from direct support is allowed. Existing plane ownership,
    higher or lower third roofs, and courtyard holes are never overwritten.
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    report = {'status': 'no-supported-opposing-ridge', 'creases': [], 'rejectedPairs': []}
    if not regions or len(regions) < 2:
        return regions, [], report
    original = list(regions)
    direct = unary_union([r['geometry'] for r in original])
    output = list(regions)
    creases = []
    limit = float(np.sqrt(2) * cell_m)
    bounds = footprint.bounds
    extent = max(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]), 1.) * 4
    for i, first in enumerate(original):
        for j, second in enumerate(original[i + 1:], i + 1):
            q0 = np.asarray(first.get('inlierXY', []), float).reshape(-1, 2)
            q1 = np.asarray(second.get('inlierXY', []), float).reshape(-1, 2)
            if min(len(q0), len(q1)) < 6:
                continue
            p0, p1 = first['plane'], second['plane']
            difference = np.asarray(p0) - p1
            norm = float(np.linalg.norm(difference[:2]))
            if norm <= 1e-8:
                continue
            cross = difference[:2] / norm
            along = np.array([-cross[1], cross[0]])
            # A ridge has rising/falling slopes on its measured two sides.
            # Two flat roofs at different heights cannot become a pitched cap.
            if float(p0[:2] @ cross) <= .05 or float(p1[:2] @ cross) >= -.05:
                continue
            d0 = (q0 @ difference[:2] + difference[2]) / norm
            d1 = (q1 @ difference[:2] + difference[2]) / norm
            if np.mean(d0 <= .22 / norm) < .9 or np.mean(d1 >= -.22 / norm) < .9:
                report['rejectedPairs'].append({'regions': [i, j], 'reason': 'observed plane ownership does not bracket a common upper crease'})
                continue
            # Independent cells on both sides must see this same along-ridge
            # interval, close to the actual plane intersection.
            near0 = q0[(d0 < -.05 * cell_m) & (d0 >= -2 * cell_m)]
            near1 = q1[(d1 > .05 * cell_m) & (d1 <= 2 * cell_m)]
            if min(len(near0), len(near1)) < 6:
                report['rejectedPairs'].append({'regions': [i, j], 'reason': 'insufficient direct cells on both sides of the intersection'})
                continue
            lower = max(float((near0 @ along).min()), float((near1 @ along).min()))
            upper = min(float((near0 @ along).max()), float((near1 @ along).max()))
            if upper - lower < 2 * cell_m:
                report['rejectedPairs'].append({'regions': [i, j], 'reason': 'opposing cells have no bounded shared ridge span'})
                continue
            origin = -cross * difference[2] / norm
            slab = Polygon([origin + along * lower + cross * extent, origin + along * upper + cross * extent,
                            origin + along * upper - cross * extent, origin + along * lower - cross * extent])
            hull = MultiPoint(np.vstack([q0, q1])).convex_hull
            third = unary_union([r['geometry'] for k, r in enumerate(original) if k not in (i, j)])
            zone = (hull.intersection(slab).intersection(footprint)
                    .intersection(first['geometry'].union(second['geometry']).buffer(limit))
                    # Leave an unknown transition cell beside another roof.
                    # An inferred crease must not create a new conflicting
                    # shared seam that demotes that roof's measured anchors.
                    .difference(third.buffer(cell_m)))
            # More than two roofs can meet. A second pair cannot overwrite an
            # already attributed interpolation patch or impose a conflicting
            # breakline through it; unresolved triple junctions stay local.
            if len(output) > len(original):
                zone = zone.difference(unary_union([r['geometry'] for r in output[len(original):]]))
            ridge = LineString([origin + along * lower, origin + along * upper])
            observed_ridge = ridge.intersection(zone)
            if observed_ridge.length < 2 * cell_m:
                continue
            # Add only previously unknown gaps, keeping every fixed anchor.
            gaps = zone.difference(direct)
            if gaps.is_empty or gaps.area <= 1e-8:
                continue
            center = origin + along * ((lower + upper) / 2)
            half0 = Polygon([center - along * extent, center + along * extent,
                             center + along * extent - cross * extent,
                             center - along * extent - cross * extent])
            half1 = Polygon([center - along * extent, center + along * extent,
                             center + along * extent + cross * extent,
                             center - along * extent + cross * extent])
            areas = []
            for plane, half in ((p0, half0), (p1, half1)):
                geometry = gaps.intersection(half)
                geometry = unary_union([p for p in _polygons(geometry) if p.area > 1e-8])
                areas.append(float(geometry.area))
                if not geometry.is_empty and geometry.area > 1e-8:
                    output.append({'geometry': geometry, 'plane': plane,
                                   'ownership': 'bounded-opposing-plane-interpolation'})
            record = {'regions': [i, j], 'status': 'bounded-opposing-plane-interpolation',
                      'bothSideNearIntersectionCells': [len(near0), len(near1)],
                      'sharedAlongRidgeSpanM': upper - lower,
                      'boundedRidgeLengthM': float(observed_ridge.length),
                      'addedInterpolationAreaM2': sum(areas),
                      'maximumDirectSupportContinuationM': limit,
                      'otherRoofTransitionGuardM': float(cell_m),
                      'withinObservedInlierConvexHull': True, 'existingAnchorsPreserved': True}
            report['creases'].append(record)
            creases.append({'line': difference.tolist(), 'zone': zone,
                            'model': {'planes': [p0.tolist(), p1.tolist()], 'operation': 'min'},
                            'audit': record})
    if creases:
        report['status'] = 'bounded-supported-creases-recovered'
    return output, creases, report


def _recover_front_planes(footprint, points, spec, constraints, existing, cell_m):
    """Recover thin measured front planes that coarse patch morphology erases.

    This does not accept an architectural template. Each recovered local plane
    needs connected training cells and independent spatial holdout, owns only
    its directly explained cells, and cannot replace an existing higher roof.
    """
    from general_world.roof_planes import RoofPolicy, sample_cells, _largest_region
    from shapely.ops import unary_union

    report = {'status': 'not-requested', 'acceptedPlanes': [], 'rejectedCandidates': 0}
    if (not isinstance(spec, dict) or spec.get('confidence') not in ('medium', 'high')
            or spec.get('kind') not in ('hip', 'mansard', 'gable')
            or spec.get('frontageKind') == 'gable-end' or not constraints):
        return existing, report
    points = np.asarray(points, float)
    selected = np.zeros(len(points), bool)
    for item in constraints:
        try:
            a, b = np.asarray(item['a'], float), np.asarray(item['b'], float)
            line, _ = _register_frontage_line(footprint, LineString([a, b]))
            if line is None:
                continue
            vector = b - a; length = np.linalg.norm(vector)
            tangent = vector / length; inward = _frontage_inward(footprint, a, b)
            rel = points[:, :2] - a
            along, depth = rel @ tangent, rel @ inward
            selected |= ((along >= -.15) & (along <= length + .15)
                         & (depth >= -.1) & (depth <= 3.))
        except (KeyError, ValueError, TypeError):
            continue
    fine = min(.25, cell_m)
    policy = RoofPolicy(cell_m=fine, minimum_points_per_cell=1)
    cells, _ = sample_cells(footprint, points[selected], policy)
    report.update(status='no-independent-local-front-planes', sourcePointCount=int(selected.sum()),
                  sampleCellCount=len(cells), cellM=fine)
    if len(cells) < 24:
        return existing, report
    origin = np.asarray(footprint.bounds[:2])
    keys = np.floor((cells[:, :2] - origin) / fine).astype(int)
    shape = tuple((keys.max(axis=0) + 1)[::-1])
    held = (keys[:, 0] + 2 * keys[:, 1]) % 4 == 0
    available = np.flatnonzero(~held)
    rng = np.random.default_rng(7421)
    regions = [dict(r) for r in (existing or [])]
    replaced_area = 0.
    for _ in range(6):
        if len(available) < 12:
            break
        design = np.c_[cells[:, :2], np.ones(len(cells))]
        candidates = []
        signatures = set()
        for _ in range(192):
            pick = rng.choice(available, 3, replace=False)
            centered = np.c_[cells[pick, :2] - cells[pick[0], :2], np.ones(3)]
            if np.linalg.matrix_rank(centered) < 3:
                continue
            fit = np.linalg.solve(centered, cells[pick, 2])
            if not .10 <= np.linalg.norm(fit[:2]) <= 3.7:
                continue
            plane = np.r_[fit[:2], fit[2] - cells[pick[0], :2] @ fit[:2]]
            signature = tuple(np.round(plane, 2))
            if signature in signatures:
                continue
            signatures.add(signature)
            inliers = available[np.abs(design[available] @ plane - cells[available, 2]) <= .15]
            if len(inliers) < 12:
                continue
            inliers = _largest_region(inliers, keys, shape)
            if len(inliers) >= 12:
                candidates.append((len(inliers), inliers))
        chosen = None
        for _, train in sorted(candidates, key=lambda q: -q[0]):
            plane, *_ = np.linalg.lstsq(design[train], cells[train, 2], rcond=None)
            spread = np.linalg.svd(cells[train, :2] - cells[train, :2].mean(axis=0), compute_uv=False)
            if (len(spread) < 2 or spread[1] / np.sqrt(len(train)) < .06
                    or np.quantile(np.abs(design[train] @ plane - cells[train, 2]), .95) > .22):
                continue  # an edge-on wall line cannot establish roof z(x,y)
            chosen = plane, train
            break
        if chosen is None:
            break
        plane, train = chosen
        # Select the hypothesis using training cells only. A failed holdout
        # leaves this region unresolved; never search the heldout cells for a
        # more convenient plane.
        available = np.setdiff1d(available, train)
        from shapely.geometry import MultiPoint
        # Hold out interior cells of this measured planar region. Buffering its
        # raster outline would test a different roof across a genuine step and
        # incorrectly erase a thin front plane.
        validation_geometry = MultiPoint(cells[train, :2]).convex_hull.intersection(footprint)
        test = np.flatnonzero(held & np.asarray([validation_geometry.covers(Point(p)) for p in cells[:, :2]]))
        if len(test) < 6:
            report['rejectedCandidates'] += 1
            continue
        error = np.abs(design[test] @ plane - cells[test, 2])
        if np.mean(error <= .22) < .8 or np.quantile(error, .95) > .25:
            report['rejectedCandidates'] += 1
            continue
        local = np.asarray([validation_geometry.covers(Point(p)) for p in cells[:, :2]])
        direct = local & (np.abs(design @ plane - cells[:, 2]) <= .22)
        if direct.sum() < 18:
            report['rejectedCandidates'] += 1
            continue
        geometry = unary_union([box(*(origin + keys[i] * fine),
            *(origin + (keys[i] + 1) * fine)) for i in np.flatnonzero(direct)])
        geometry = (geometry.buffer(fine / 2, join_style=2)
                    .intersection(validation_geometry.buffer(fine / 2))
                    .intersection(footprint))
        for old in regions:
            overlap = old['geometry'].intersection(geometry)
            if overlap.is_empty:
                continue
            point = np.r_[overlap.representative_point().coords[0], 1.]
            if float(plane @ point) > float(old['plane'] @ point) + .22:
                old['geometry'] = old['geometry'].difference(overlap)
                replaced_area += overlap.area
            else:
                geometry = geometry.difference(old['geometry'])
        available = np.setdiff1d(available, train)
        if geometry.is_empty or geometry.area < .10:
            continue
        regions.append({'geometry': geometry, 'plane': plane})
        report['acceptedPlanes'].append({'plane': plane.tolist(), 'ownedAreaM2': float(geometry.area),
            'trainingCells': len(train), 'heldOutCells': len(test), 'heldOutP95M': float(np.quantile(error, .95)),
            'heldOutInlierFraction': float(np.mean(error <= .22)), 'boundedContinuationM': fine / 2})
    regions = [r for r in regions if not r['geometry'].is_empty]
    report['replacedCoarseLowerOwnershipM2'] = float(replaced_area)
    if report['acceptedPlanes']:
        report['status'] = 'independently-validated-local-front-planes'
    return regions or None, report


def _harmonic_structural_heights(xy, faces, regions, eave_mask, eave_levels, fallback_z,
                                 seam_tolerance=.75, maximum_transition_grade=1.5):
    """Keep measured plane interiors and harmonically fill only unknown vertices.

    Conflicting shared seams are treated as an unknown band instead of being
    averaged into vertical fins. The graph Laplacian has a maximum principle:
    interpolated values stay between nearby supported/eave anchors.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    xy = np.asarray(xy, dtype=float)
    faces = np.asarray(faces, dtype=int)
    z = np.asarray(fallback_z, dtype=float).copy()
    known = np.zeros(len(xy), dtype=bool)
    seam_conflict = np.zeros(len(xy), dtype=bool)
    seam_audit = []

    # Only a seam whose independently fitted planes disagree beyond tolerance
    # gets an interpolation band; well-matched ridges remain exact shared seams.
    seam_bands = []
    for i, first in enumerate(regions):
        for j, second in enumerate(regions[i + 1:], i + 1):
            shared = first["geometry"].boundary.intersection(second["geometry"].boundary)
            if shared.is_empty or shared.length < .25:
                continue
            xy_samples = []
            for line in getattr(shared, "geoms", [shared]):
                if line.geom_type == "LineString" and line.length > .05:
                    count = max(2, int(np.ceil(line.length / .5)) + 1)
                    xy_samples.extend(line.interpolate(t, normalized=True).coords[0]
                                      for t in np.linspace(0., 1., count))
            if not xy_samples:
                continue
            design = np.c_[xy_samples, np.ones(len(xy_samples))]
            delta = design @ (first["plane"] - second["plane"])
            p90, maxdiff = float(np.quantile(np.abs(delta), .90)), float(np.max(np.abs(delta)))
            report = {"regions": [i, j], "sharedBoundaryLengthM": float(shared.length),
                      "heightDifferenceP90M": p90, "heightDifferenceMaxM": maxdiff,
                      "interpolatedConflict": bool(p90 > seam_tolerance)}
            seam_audit.append(report)
            if p90 > seam_tolerance:
                width = max(1.5, maxdiff / maximum_transition_grade)
                seam_bands.append((shared, width, report))

    # Supported interior vertices are Dirichlet anchors. A conflicting seam
    # becomes a local unknown band, except where a measured eave owns the point.
    memberships = [[] for _ in xy]
    # Mesh XY is stored to nine decimal places. A clipped footprint boundary
    # may lie <1e-9m beyond a direct-support polygon after that serialization;
    # do not turn those genuinely supported vertices into harmonic unknowns.
    membership_geometry = [region['geometry'].buffer(1e-8) for region in regions]
    for vertex_id, p in enumerate(xy):
        point = Point(p)
        for region_id, geometry in enumerate(membership_geometry):
            if geometry.covers(point):
                memberships[vertex_id].append(region_id)
    for shared, width, _ in seam_bands:
        for i, p in enumerate(xy):
            if shared.distance(Point(p)) <= width:
                seam_conflict[i] = True

    plane_spread_max = 0.
    for i, ids in enumerate(memberships):
        if not ids or seam_conflict[i]:
            continue
        estimates = [float(regions[k]["plane"] @ np.r_[xy[i], 1.]) for k in ids]
        spread = float(np.ptp(estimates)) if len(estimates) > 1 else 0.
        plane_spread_max = max(plane_spread_max, spread)
        if spread <= seam_tolerance:
            z[i] = float(np.mean(estimates))
            known[i] = True
        # A disagreement above tolerance is not a step claim: leave the seam
        # unknown so surrounding validated regions interpolate it.

    eave_mask = np.asarray(eave_mask, dtype=bool)
    eave_levels = np.asarray(eave_levels, dtype=float)
    eave_conflicts = int(np.sum(eave_mask & known & (np.abs(z - eave_levels) > seam_tolerance)))
    z[eave_mask] = eave_levels[eave_mask]
    known[eave_mask] = True
    unknown = ~known

    # Positive inverse-length weights on the shared mesh graph.
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    weights = 1. / np.maximum(lengths, 1e-6)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    vals = np.r_[weights, weights]
    adjacency = coo_matrix((vals, (rows, cols)), shape=(len(xy), len(xy))).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    component_count, component_labels = connected_components(adjacency, directed=False)
    supported_values = z[known]
    fallback = float(np.median(supported_values)) if len(supported_values) else float(np.median(z))
    for component in range(component_count):
        ids = np.where((component_labels == component) & unknown)[0]
        if not len(ids):
            continue
        anchors = np.where((component_labels == component) & known)[0]
        if not len(anchors):
            z[ids] = fallback
            continue
        block = laplacian[ids][:, ids]
        rhs = -(laplacian[ids][:, anchors] @ z[anchors])
        solved = np.asarray(spsolve(block, rhs), dtype=float).reshape(-1)
        lo, hi = float(z[anchors].min()), float(z[anchors].max())
        z[ids] = np.clip(solved, lo, hi)

    unknown_edges = (unknown[edges[:, 0]] | unknown[edges[:, 1]])
    grades = np.abs(z[edges[:, 0]] - z[edges[:, 1]]) / np.maximum(lengths, 1e-8)
    return z, known, unknown, {
        "method": "harmonic-graph-interpolation-with-Dirichlet-plane-and-eave-anchors",
        "unknownVertexCount": int(unknown.sum()),
        "conflictingSeamBandVertexCount": int(seam_conflict.sum()),
        "seamToleranceM": float(seam_tolerance),
        "planeSpreadAtSupportedVerticesMaxM": float(plane_spread_max),
        "seamReports": seam_audit,
        "eaveAnchorConflictCount": eave_conflicts,
        "unknownHeightRangeM": ([float(z[unknown].min()), float(z[unknown].max())]
                                if unknown.any() else []),
        "dirichletHeightRangeM": ([float(z[known].min()), float(z[known].max())]
                                  if known.any() else []),
        "maximumUnknownEdgeGrade": float(grades[unknown_edges].max()) if unknown_edges.any() else 0.,
        "disconnectedUnknownFallbackZ": fallback,
    }


def component_cutters(features, z0):
    """Validated convex cells for observed dormer bodies replacing host faces."""
    from .roof_features import dormer_cut_solids
    cutters, records = [], []
    for index, feature in enumerate(features):
        if feature.get('kind') != 'dormer' or feature.get('composeWithHost') is not True:
            continue
        try:
            cells = dormer_cut_solids(feature, z0)
            if not cells or any(not cell.is_volume for cell in cells):
                raise ValueError('source component has no closed outer body')
            for cell in cells:
                normals = np.asarray(cell.face_normals)
                offsets = np.einsum('ij,ij->i', normals, cell.triangles_center)
                values = np.c_[normals, offsets]
                _, unique = np.unique(np.round(values, 8), axis=0, return_index=True)
                cutters.append({'planes': values[np.sort(unique)], 'bounds': cell.bounds})
            records.append({'featureIndex': index, 'status': 'source-front-body-owns-intersection',
                            'convexCells': len(cells), 'frontCoordinatesMoved': False})
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            records.append({'featureIndex': index, 'status': 'unresolved', 'reason': str(error)})
    return cutters, records


def clip_component_intersections(mesh, cutters):
    """Subtract only the intersection of source bodies and opaque host faces.

    Triangle clipping preserves the original host planes. The component's own
    side/front/cap surfaces close the intersection; host geometry above a cap
    or outside its bounded body remains intact.
    """
    if mesh is None or not len(mesh.faces) or not cutters:
        return mesh
    if not any(np.all(mesh.bounds[1] >= c['bounds'][0] - 1e-8)
               and np.all(mesh.bounds[0] <= c['bounds'][1] + 1e-8) for c in cutters):
        return mesh
    def split(poly, plane):
        inside, outside = [], []
        distance = np.asarray(poly) @ plane[:3] - plane[3]
        for i, point in enumerate(poly):
            j = (i + 1) % len(poly)
            in0, in1 = distance[i] <= 1e-8, distance[j] <= 1e-8
            (inside if in0 else outside).append(point)
            if in0 != in1:
                t = distance[i] / (distance[i] - distance[j])
                crossing = point + np.clip(t, 0., 1.) * (poly[j] - point)
                inside.append(crossing); outside.append(crossing)
        def clean(points):
            result = []
            for point in points:
                if not result or np.linalg.norm(point - result[-1]) > 1e-9:
                    result.append(point)
            if len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= 1e-9:
                result.pop()
            return np.asarray(result) if len(result) >= 3 else None
        return clean(inside), clean(outside)
    output = []
    for triangle in np.asarray(mesh.triangles):
        fragments = [triangle]
        for cutter in cutters:
            changed = []
            for poly in fragments:
                if np.any(poly.max(axis=0) < cutter['bounds'][0] - 1e-8) or np.any(poly.min(axis=0) > cutter['bounds'][1] + 1e-8):
                    changed.append(poly); continue
                remaining = poly
                for plane in cutter['planes']:
                    remaining, outside = split(remaining, plane)
                    if outside is not None:
                        changed.append(outside)
                    if remaining is None:
                        break
                # The remaining polygon lies inside the source body and is
                # replaced, not emitted as opaque backing behind its aperture.
            fragments = changed
            if not fragments:
                break
        for poly in fragments:
            for i in range(1, len(poly) - 1):
                tri = np.array([poly[0], poly[i], poly[i + 1]])
                if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-10:
                    output.append(tri)
    if not output:
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), int), process=False)
    vertices, inverse = np.unique(np.round(np.asarray(output).reshape(-1, 3), 9), axis=0, return_inverse=True)
    result = trimesh.Trimesh(vertices=vertices, faces=inverse.reshape(-1, 3), process=False)
    result.metadata.update(mesh.metadata)
    result.metadata['sourceComponentHostCut'] = True
    return result


def build(footprint, points, base_z, cell_m=.5, eave_constraints=None, host_spec=None,
          boundary_points=None, source_feature_support=None, front_priors=None,
          native_baseline_surface=None, source_owned_domains=(),
          gable_priors=None, source_outline_points=()):
    """Build a continuous top mesh and perimeter skirts.

    ``points`` should already be roof-filtered when possible. Unsupported
    cells use a robust local fit constrained by the dense roof-height envelope;
    isolated height outliers are rejected unless a connected patch has enough
    spatial support. ``base_z`` is used only when no roof samples are available.
    """
    input_footprint_type = footprint.geom_type
    footprint = _polygonal_footprint(footprint)
    if footprint.is_empty or footprint.area <= 0 or cell_m <= 0:
        raise ValueError("footprint must contain polygonal area and cell_m must be positive")
    base_z = float(base_z)
    raw = np.asarray(points, dtype=float)
    if raw.size == 0:
        raw = np.empty((0, 3), dtype=float)
    raw = raw.reshape((-1, raw.shape[-1] if raw.ndim > 1 else 3))
    raw = raw[:, :3]
    finite = np.isfinite(raw).all(axis=1)
    raw = raw[finite]
    if len(raw):
        from shapely import contains_xy
        mask = contains_xy(footprint, raw[:, 0], raw[:, 1])
        raw = raw[mask]
    primary_point_count = len(raw)
    boundary_evidence_count = 0
    if boundary_points is not None:
        extra = np.asarray(boundary_points, dtype=float)
        if extra.size:
            extra = extra.reshape((-1, extra.shape[-1]))[:, :3]
            from shapely import contains_xy
            eligible = np.isfinite(extra).all(axis=1) & contains_xy(footprint, extra[:, 0], extra[:, 1])
            extra = extra[eligible]
            # The caller's ordinary interior sample is intentionally eroded.
            # Recover edge evidence without replacing its independently filtered
            # interior statistics or importing a neighboring roof/courtyard.
            near = np.asarray([footprint.boundary.distance(Point(p)) <= 1.0 for p in extra[:, :2]], dtype=bool)
            extra = extra[near]
            before = len(raw)
            raw = np.unique(np.vstack([raw, extra]), axis=0)
            boundary_evidence_count = len(raw) - before
    aggregated, cell_keys, cell_counts, cell_origin = _aggregate_cells(raw, cell_m)
    keep, rejected_cells, rejected_low_cells, retained_patches = _filter_cells(
        aggregated, cell_keys, cell_counts, cell_m)
    samples_in = aggregated[keep]
    structural_raw = _raw_points_for_kept_cells(raw, cell_keys, keep, cell_origin, cell_m)
    samples_in, plane_audit = _project_supported_planes(samples_in)
    structural_regions, structural_plane_audit = _fit_structural_regions(
        footprint, structural_raw, cell_m, source_feature_support)
    structural_regions, front_plane_audit = _recover_front_planes(
        footprint, structural_raw, host_spec, eave_constraints or (), structural_regions, cell_m)
    structural_plane_audit['localFrontPlaneRecovery'] = front_plane_audit
    if front_plane_audit['acceptedPlanes']:
        from shapely.ops import unary_union
        supported = unary_union([r['geometry'] for r in structural_regions]).intersection(footprint)
        structural_plane_audit.update(
            status='validated-local-structural-regions',
            acceptedPlaneCount=len(structural_regions),
            planes=[r['plane'].tolist() for r in structural_regions],
            validatedSupportAreaM2=float(supported.area),
            unresolvedAreaM2=float(footprint.area - supported.area),
            unresolvedAreaFraction=float((footprint.area - supported.area) / footprint.area))
    structural_regions, supported_creases, crease_audit = _recover_supported_ridges(
        footprint, structural_regions, cell_m)
    structural_plane_audit['boundedCreaseInterpolation'] = crease_audit
    if len(samples_in):
        envelope_low, envelope_high = np.quantile(samples_in[:, 2], [.02, .98])
    else:
        envelope_low = envelope_high = base_z

    tree = cKDTree(samples_in[:, :2]) if len(samples_in) else None
    linear = None
    if len(samples_in) >= 3:
        try:
            linear = LinearNDInterpolator(samples_in[:, :2], samples_in[:, 2], fill_value=np.nan)
        except (QhullError, ValueError):
            linear = None

    max_nearest_m = max(1.5 * cell_m, .75)
    max_extrapolation_slope = 1.25
    max_surface_grade = 2.5
    envelope_margin_m = .5

    def local_fit(query, indices):
        support = samples_in[np.asarray(indices, dtype=int)]
        if len(support) < 3:
            return float(np.clip(np.median(support[:, 2]), envelope_low - envelope_margin_m,
                                 envelope_high + envelope_margin_m))
        offsets = support[:, :2] - query
        design = np.c_[offsets, np.ones(len(support))]
        keep_local = np.ones(len(support), dtype=bool)
        coeff = np.zeros(3)
        for _ in range(3):
            if keep_local.sum() < 3:
                break
            coeff, *_ = np.linalg.lstsq(design[keep_local], support[keep_local, 2], rcond=None)
            errors = support[:, 2] - design @ coeff
            med = np.median(errors[keep_local])
            mad = np.median(np.abs(errors[keep_local] - med))
            proposed = np.abs(errors - med) <= max(.65, 3. * 1.4826 * mad)
            if proposed.sum() < 3 or np.array_equal(proposed, keep_local):
                break
            keep_local = proposed
        slope = coeff[:2]
        magnitude = np.linalg.norm(slope)
        if magnitude > max_extrapolation_slope:
            coeff[:2] *= max_extrapolation_slope / magnitude
        return float(np.clip(coeff[2], envelope_low - envelope_margin_m,
                             envelope_high + envelope_margin_m))

    def heights(xy):
        xy = np.asarray(xy, dtype=float).reshape((-1, 2))
        if not len(samples_in):
            return np.full(len(xy), base_z)
        z = (np.asarray(linear(xy), dtype=float).reshape(-1) if linear is not None
             else np.full(len(xy), np.nan))
        missing = ~np.isfinite(z)
        if missing.any():
            dist, indices = tree.query(xy[missing], k=min(12, len(samples_in)))
            if np.ndim(indices) == 1:
                indices = indices[:, None]
                dist = dist[:, None]
            ids = np.where(missing)[0]
            for row in range(len(ids)):
                z[ids[row]] = local_fit(xy[ids[row]], indices[row])
        return z

    vertices_xy = []
    faces = []
    vertex_map = {}

    def vertex(xy):
        key = (round(float(xy[0]), 9), round(float(xy[1]), 9))
        if key not in vertex_map:
            vertex_map[key] = len(vertices_xy)
            vertices_xy.append(key)
        return vertex_map[key]

    minx, miny, maxx, maxy = footprint.bounds
    ix0, ix1 = int(np.floor(minx / cell_m)), int(np.ceil(maxx / cell_m))
    iy0, iy1 = int(np.floor(miny / cell_m)), int(np.ceil(maxy / cell_m))
    for iy in range(iy0, iy1):
        y0, y1 = iy * cell_m, (iy + 1) * cell_m
        for ix in range(ix0, ix1):
            x0, x1 = ix * cell_m, (ix + 1) * cell_m
            clipped = box(x0, y0, x1, y1).intersection(footprint)
            if clipped.is_empty:
                continue
            for poly in _polygons(clipped):
                if poly.area <= 1e-10:
                    continue
                # One clipped grid arrangement spans supported regions and
                # gaps. Plane ownership changes the height field, never the
                # mesh topology, so seams cannot create patch skirts/T-junctions.
                try:
                    tri_xy, tri_faces = trimesh.creation.triangulate_polygon(poly, engine="earcut")
                    triangles = [tri_xy[tri] for tri in tri_faces]
                except (ValueError, RuntimeError):
                    triangles = [np.asarray(t.exterior.coords)[:3] for t in triangulate(poly)
                                 if poly.covers(t.representative_point()) and t.difference(poly).area < 1e-8]
                for coords in triangles:
                    tri_poly = Polygon(coords)
                    if tri_poly.area <= 1e-12 or tri_poly.difference(footprint).area > 1e-8:
                        continue
                    ids = [vertex(p) for p in coords]
                    if Polygon([vertices_xy[j] for j in ids]).exterior.is_ccw:
                        faces.append(ids)
                    else:
                        faces.append(ids[::-1])

    constraints = []
    constraint_endpoints = list(source_outline_points)
    constraint_registration = []
    for item in eave_constraints or ():
        try:
            a = np.asarray(item["a"], dtype=float)
            b = np.asarray(item["b"], dtype=float)
            level = float(item.get("z", item.get("z0")))
            end_level = float(item.get("z1", level))
        except (KeyError, TypeError, ValueError):
            continue
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([*a, *b, level, end_level]).all():
            continue
        raw_line = LineString([a, b])
        snapped_line, registration = _register_frontage_line(footprint, raw_line)
        registration["role"] = str(item.get("role", "eave"))
        constraint_registration.append(registration)
        if snapped_line is None:
            continue
        if snapped_line.length <= .05:
            continue
        role = str(item.get("role", "eave"))
        uncertainty = max(0., float(item.get("uncertaintyM", 0.)))
        if (isinstance(host_spec, dict) and host_spec.get("frontageKind") == "gable-end"
                and role in ("eave", "visible-roof-edge")):
            uncertainty = max(uncertainty, max(0., float(host_spec.get("anchorUncertaintyM", 0.))))
        constraints.append((snapped_line, level, role, end_level, uncertainty))
        for endpoint in (Point(snapped_line.coords[0]), Point(snapped_line.coords[-1])):
            near = nearest_points(footprint.boundary, endpoint)[0]
            if near.distance(endpoint) <= 1.5:
                constraint_endpoints.append(np.asarray(near.coords[0]))

    if constraint_endpoints:
        vertices_xy, faces, vertex_map = _split_faces_at_points(
            vertices_xy, faces, vertex_map, constraint_endpoints)

    xy = np.asarray(vertices_xy, dtype=float)
    z = heights(xy)
    structural_supported = np.zeros(len(xy), dtype=bool)
    if len(samples_in):
        support_distance, _ = tree.query(xy)
        unsupported = support_distance > max_nearest_m
        boundary_distance = np.asarray([footprint.boundary.distance(Point(p)) for p in xy])
        unsupported_interior = unsupported & (boundary_distance > max_nearest_m)
        nearest_fallback = int(np.sum(~np.isfinite(linear(xy)))) if linear is not None else len(xy)
        extrapolated = int(np.sum(unsupported))
    else:
        unsupported = np.ones(len(xy), dtype=bool)
        unsupported_interior = unsupported.copy()
        nearest_fallback = 0
        extrapolated = 0
    eave_levels = np.full(len(xy), np.nan)
    eave_distances = np.full(len(xy), np.inf)
    boundary_mask = np.asarray([footprint.boundary.distance(Point(p)) <= 1e-7 for p in xy])
    constraint_role_by_vertex = np.full(len(xy), "", dtype=object)
    for line, level, role, end_level, uncertainty in constraints:
        # Uncertain photo-derived edge heights are fit priors, not hard
        # Dirichlet values. A validated gable model can own this seam; if it
        # fails, measured roof data remains in control of the boundary.
        if uncertainty > 0.:
            continue
        if line.length <= 1e-8:
            continue
        projected_distance = np.asarray([line.project(Point(p)) for p in xy])
        dist = np.asarray([line.distance(Point(p)) for p in xy])
        t = projected_distance / line.length
        eligible = boundary_mask & (dist <= .25) & (t >= -1e-8) & (t <= 1 + 1e-8)
        choose = eligible & (dist < eave_distances)
        eave_levels[choose] = level + np.clip(t[choose], 0., 1.) * (end_level - level)
        eave_distances[choose] = dist[choose]
        constraint_role_by_vertex[choose] = role
    eave_mask = np.isfinite(eave_levels)
    eave_floor_conflicts = int(np.sum(eave_mask & (eave_levels < base_z)))
    front_ownership_audit = {"status": "not-requested"}
    gable_ownership_audit = {"status": "not-requested"}
    native_support_state = None
    front_adjustable = None
    front_grade_cap = max_surface_grade
    front_provisional_mask = np.zeros(len(xy), dtype=bool)
    if structural_regions:
        z, structural_supported, structural_unknown, harmonic_audit = _harmonic_structural_heights(
            xy, faces, structural_regions, eave_mask, eave_levels, z)
        z = np.maximum(z, base_z)
        structural_plane_audit.update(harmonic_audit)
        if front_priors:
            from . import roof_front_ownership
            if native_baseline_surface is None:
                front_ownership_audit = {"status": "unresolved-native-baseline-required"}
            else:
                baseline_vertices = np.asarray(native_baseline_surface.vertices, float)
                distance, index = cKDTree(baseline_vertices[:, :2]).query(xy)
                if distance.max() > 1e-5:
                    front_ownership_audit = {"status": "unresolved-baseline-grid-mismatch",
                                             "maximumDistanceM": float(distance.max())}
                else:
                    baseline_z = baseline_vertices[index, 2].copy()
                    front_result = roof_front_ownership.solve_front_ownership(
                        xy, faces, baseline_z, structural_supported, structural_unknown,
                        structural_regions, supported_creases, footprint, front_priors,
                        protected_domains=source_owned_domains, cell_m=cell_m)
                    front_ownership_audit = front_result['audit']
                    if front_ownership_audit.get('appliedBoundaryVertices', 0):
                        z = front_result['z']
                        front_provisional_mask = front_result['provisional_boundary_mask']
                        front_adjustable = (front_result['changed_unknown_mask']
                                            & ~front_provisional_mask & ~eave_mask)
                        front_grade_cap = max(max_surface_grade, max(
                            (float(r.get('capSlopeFromAnchorInequalities',0.))
                             for r in front_ownership_audit.get('records',[])), default=0.))
                        # These are numerical locks for one provisional solve,
                        # never additions to the native measured-anchor mask.
                        assert np.array_equal(front_result['measured_anchor_mask'], structural_supported)
        if gable_priors:
            from . import roof_gable_ownership
            if front_priors:
                gable_ownership_audit = {"status": "unresolved-competing-front-ownership-priors"}
            else:
                gable_z, gable_photo, gable_changed, gable_ownership_audit = roof_gable_ownership.solve(
                    xy, faces, z, structural_supported, structural_unknown,
                    gable_priors, footprint, protected_domains=source_owned_domains, cell_m=cell_m)
                if gable_ownership_audit.get('appliedBoundaryVertices', 0):
                    z = gable_z
                    front_adjustable = gable_changed & ~gable_photo & ~eave_mask
        native_support_state = {'xy':xy.copy(), 'faces':np.asarray(faces, np.int64).copy(),
                                'z':z.copy(), 'known':structural_supported.copy(),
                                'unknown':structural_unknown.copy()}
        structural_plane_audit["supportedOwnedFaceCount"] = int(sum(
            bool(any(r["geometry"].covers(Point(np.mean(xy[face], axis=0))) for r in structural_regions))
            for face in faces))
        structural_plane_audit["ownedFaceCount"] = int(len(faces))
        structural_plane_audit["unsupportedOwnedFaceCount"] = int(
            len(faces) - structural_plane_audit["supportedOwnedFaceCount"])
        structural_plane_audit["sharedVertexCount"] = int(sum(
            sum(r["geometry"].covers(Point(p)) for r in structural_regions) > 1 for p in xy))
        structural_plane_audit["maximumSharedPlaneHeightDifferenceM"] = max(
            (s["heightDifferenceMaxM"] for s in harmonic_audit["seamReports"]), default=0.)
        edges = np.unique(np.sort(np.vstack((np.asarray(faces)[:, [0, 1]],
                                             np.asarray(faces)[:, [1, 2]],
                                             np.asarray(faces)[:, [2, 0]])), axis=1), axis=0)
        lengths = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
        grades = np.abs(z[edges[:, 0]] - z[edges[:, 1]]) / np.maximum(lengths, 1e-8)
        z, raw_max_grade, limited_max_grade, clipped_edges, extrapolated_max_grade = _limit_mesh_slope(
            xy, np.asarray(faces, dtype=np.int64), z,
            eligible_vertices=(structural_unknown & ~eave_mask
                               if front_adjustable is None else front_adjustable),
            maximum_grade=front_grade_cap, sweeps=64)
        structural_plane_audit['provisionalSlopeLimit'] = {
            'maximumGrade': front_grade_cap, 'adjustedEdgeCount': clipped_edges,
            'measuredPlaneAndEaveAnchorsPreserved': True,
            'remainingMaximumUnknownGrade': extrapolated_max_grade,
            'method': 'sparse-L1-grade-projection-with-fixed-anchors',
            'fixedAnchorGradeConflict': bool(extrapolated_max_grade > front_grade_cap + 1e-4)}
    else:
        z = np.maximum(z, base_z)
        z, raw_max_grade, limited_max_grade, clipped_edges, extrapolated_max_grade = _limit_mesh_slope(
            xy, np.asarray(faces, dtype=np.int64), z, eligible_vertices=unsupported & ~eave_mask,
            maximum_grade=max_surface_grade)
        if len(samples_in):
            # Preserve measured cells, but keep unsupported continuation inside
            # the global envelope supported by the dense roof returns.
            z[unsupported] = np.clip(z[unsupported], envelope_low - envelope_margin_m,
                                     envelope_high + envelope_margin_m)
            z, _, limited_max_grade, extra_clipped, extrapolated_max_grade = _limit_mesh_slope(
                xy, np.asarray(faces, dtype=np.int64), z, eligible_vertices=unsupported & ~eave_mask,
                maximum_grade=max_surface_grade, sweeps=48)
            clipped_edges += extra_clipped
    z = np.maximum(z, base_z)
    z[eave_mask] = np.maximum(eave_levels[eave_mask], base_z)
    z, host_audit = _observed_host(footprint, samples_in, xy, z, base_z, host_spec,
                                   eave_constraints or (), cell_m, raw_samples=structural_raw)
    # An explicitly validated photographed gable already supplies its own
    # crease and bounded front-zone fit. Preserve that stronger breakline.
    for crease in supported_creases:
        if host_audit.get('status') == 'applied-observed-front-gable' or '_analyticModel' in host_audit:
            crease['audit']['breaklineStatus'] = 'explicit-observed-host-breakline-takes-precedence'
            continue
        zone = crease['zone']
        for line, _, _, _, uncertainty in constraints:
            if uncertainty == 0.:
                zone = zone.difference(line.buffer(.25))
        xy, faces, z, vertex_map, inserted = _split_mesh_at_ridge(
            xy, faces, z, vertex_map, crease['line'], {'geometry': zone}, crease['model'])
        xy, z = np.asarray(xy, float), np.maximum(np.asarray(z, float), base_z)
        faces = np.asarray(faces, np.int64)
        crease['audit']['insertedBreaklineVertexCount'] = int(inserted)
    if isinstance(host_spec, dict) and host_audit.get("status") != "not-proposed":
        host_audit.setdefault("anchorRole", host_spec.get("anchorRole", "eave"))
        if "anchorZ" not in host_audit and host_spec.get("eaveZ") is not None:
            host_audit["anchorZ"] = float(host_spec["eaveZ"])
        if host_audit["anchorRole"] != "eave":
            host_audit.pop("eaveZ", None)
    analytic_host_model = host_audit.pop("_analyticModel", None)
    ridge_line = host_audit.pop("_ridgeLine", None)
    host_zone = host_audit.pop("_hostZone", None)
    ridge_model = host_audit.pop("_ridgeModel", None)
    if ridge_line is not None and host_zone is not None and ridge_model is not None:
        xy, faces, z_values, vertex_map, ridge_vertices = _split_mesh_at_ridge(
            xy, faces, z, vertex_map, ridge_line, host_zone, ridge_model)
        xy = np.asarray(xy, dtype=float)
        z = np.asarray(z_values, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)
        host_audit["ridgeBreakline"] = {
            "status": "inserted", "implicitPlaneDifference": ridge_line,
            "insertedVertexCount": int(ridge_vertices),
            "hostZoneDepthM": float(host_zone["depth"]),
            "triangles": "split along validated plane intersection; existing vertex heights preserved",
        }
    surface = trimesh.Trimesh(vertices=np.c_[xy, z], faces=np.asarray(faces, dtype=np.int64), process=False)

    # Build skirts on the actual polygon rings. Include every top-mesh boundary
    # vertex so each skirt top edge is welded to the corresponding surface edge.
    skirt_xy = []
    skirt_faces = []
    skirt_map = {}

    def skirt_vertex(xy, z):
        key = (round(float(xy[0]), 9), round(float(xy[1]), 9), round(float(z), 8))
        if key not in skirt_map:
            skirt_map[key] = len(skirt_xy)
            skirt_xy.append((key[0], key[1], key[2]))
        return skirt_map[key]

    boundary_samples = []
    boundary_chains = []
    for raw_poly in _polygons(footprint):
        poly = orient(raw_poly, sign=1.0)
        for ring_index, ring in enumerate((poly.exterior, *poly.interiors)):
            ringline = ring
            ring_samples = []
            for p in xy:
                pt = Point(p)
                if ringline.distance(pt) <= 1e-8:
                    ring_samples.append((ringline.project(pt), p))
            ordered = sorted(ring_samples, key=lambda item: item[0])
            compact = []
            for d, p in ordered:
                if not compact or np.linalg.norm(np.asarray(p) - compact[-1][1]) > 1e-8:
                    compact.append((d, np.asarray(p)))
            if len(compact) < 2:
                continue
            ring_chain = []
            for i, (_, p0) in enumerate(compact):
                p1 = compact[(i + 1) % len(compact)][1]
                # Read the limited top surface's exact boundary vertex heights.
                z0, z1 = float(z[vertex_map[(round(float(p0[0]), 9), round(float(p0[1]), 9))]]), \
                    float(z[vertex_map[(round(float(p1[0]), 9), round(float(p1[1]), 9))]])
                a = skirt_vertex(p0, z0); b = skirt_vertex(p1, z1)
                c = skirt_vertex(p1, base_z); d = skirt_vertex(p0, base_z)
                # Ring direction keeps the roof footprint on its left; reverse
                # the quad winding so normals point out of the solid.
                skirt_faces.extend(([a, c, b], [a, d, c]))
                length = float(np.linalg.norm(p1 - p0))
                count = max(1, int(np.ceil(length / min(.5, cell_m))))
                for j in range(count):
                    t = j / count
                    p = p0 + (p1 - p0) * t
                    sample = [p[0], p[1], z0 + (z1 - z0) * t]
                    boundary_samples.append(sample)
                    ring_chain.append(sample)
            if ring_chain:
                boundary_chains.append({"ring": "exterior" if ring_index == 0 else "hole",
                                        "points": np.asarray(ring_chain, dtype=float)})

    skirt = trimesh.Trimesh(vertices=np.asarray(skirt_xy, dtype=float),
                            faces=np.asarray(skirt_faces, dtype=np.int64), process=False)
    boundary_samples = np.asarray(boundary_samples, dtype=float).reshape((-1, 3))
    analytic_mesh_audit = {"status": "not-used"}
    if analytic_host_model is not None:
        try:
            surface, skirt, boundary_samples = _analytic_roof_components(
                footprint, analytic_host_model, base_z, min(.5, cell_m))
            boundary_chains = _boundary_chains_from_samples(footprint, boundary_samples)
            analytic_mesh_audit = {"status": "exact-plane-envelope", "source": "roof_solid",
                                   "surfaceFaces": int(len(surface.faces)),
                                   "perimeterWallFaces": int(len(skirt.faces)),
                                   "boundaryOwnershipSamples": int(len(boundary_samples))}
        except (ImportError, ValueError, RuntimeError) as exc:
            analytic_mesh_audit = {"status": "analytic-solid-failed-grid-retained",
                                   "reason": str(exc)[:240]}
    if analytic_host_model is None and structural_regions:
        analytic_mesh_audit = {"status": "shared-local-plane-arrangement",
                               "source": "spatially-validated-local-planes",
                               "surfaceFaces": int(len(surface.faces)),
                               "perimeterWallFaces": int(len(skirt.faces)),
                               "boundaryOwnershipSamples": int(len(boundary_samples))}
    if len(samples_in) and len(boundary_samples):
        boundary_support_distance, _ = tree.query(boundary_samples[:, :2])
        boundary_unsupported = boundary_support_distance > max_nearest_m
    else:
        boundary_unsupported = np.ones(len(boundary_samples), dtype=bool)
    audit = {
        "status": "continuous-provisional-surface",
        "inputFootprintGeometryType": input_footprint_type,
        "polygonalFootprintGeometryType": footprint.geom_type,
        "inputPointCount": int(len(raw)),
        "primaryInteriorPointCount": int(primary_point_count),
        "additionalBoundaryEvidencePointCount": int(boundary_evidence_count),
        "structuralFitUsesMeasuredCellKeepMask": True,
        "structuralInputPointCount": int(len(structural_raw)),
        "structuralInputPointsRejectedByMeasuredCellFilter": int(len(raw) - len(structural_raw)),
        "aggregatedSampleCount": int(len(samples_in)),
        "surfaceVertexCount": int(len(surface.vertices)),
        "surfaceFaceCount": int(len(surface.faces)),
        "nearestFilledVertexCount": int(nearest_fallback),
        "unsupportedVertexCount": int(np.sum(unsupported)),
        "unsupportedInteriorVertexCount": int(np.sum(unsupported_interior)),
        "extrapolatedVertexCount": int(extrapolated),
        "nearestFillLimitM": float(max_nearest_m),
        "unsupportedHeightFallback": ("harmonic interpolation from validated plane and boundary anchors"
                                       if structural_regions else "bounded robust local fit; base_z only with no samples"),
        "rejectedIsolatedCells": int(rejected_cells),
        "rejectedLowCells": int(rejected_low_cells),
        "retainedSpatialRoofPatches": int(retained_patches),
        "localPlaneFit": plane_audit,
        "boundaryConstraintsAccepted": int(len(constraints)),
        "boundaryConstraintRoles": {role: int(sum(item[2] == role for item in constraints))
                                    for role in sorted({item[2] for item in constraints})},
        "eaveConstraintsAccepted": int(sum(item[2] == "eave" for item in constraints)),
        "boundaryConstraintRegistration": constraint_registration,
        "eaveConstraintRegistration": [q for q in constraint_registration if q.get("role") == "eave"],
        "boundaryConstraintVerticesApplied": int(eave_mask.sum()),
        "uncertainBoundaryConstraintCount": int(sum(item[4] > 0. for item in constraints)),
        "uncertainBoundaryConstraintsAppliedAsHardAnchors": 0,
        "eaveConstraintVerticesApplied": int(np.sum(eave_mask & (constraint_role_by_vertex == "eave"))),
        "visibleRoofEdgeConstraintVerticesApplied": int(np.sum(eave_mask & (constraint_role_by_vertex == "visible-roof-edge"))),
        "eaveConstraintFloorConflicts": int(np.sum(eave_mask & (constraint_role_by_vertex == "eave") & (eave_levels < base_z))),
        "extrapolationSlopeCap": float(max_extrapolation_slope),
        "extrapolatedEdgeGradeCap": float(max_surface_grade),
        "clippedExtrapolationEdgeCount": int(clipped_edges),
        "maximumExtrapolatedMeshGrade": float(extrapolated_max_grade),
        "preLimitMaximumMeshGrade": float(raw_max_grade),
        "maximumMeshGrade": float(limited_max_grade),
        "extrapolationHeightEnvelopeM": [float(envelope_low - envelope_margin_m), float(envelope_high + envelope_margin_m)],
        "boundaryExtrapolatedSampleCount": int(boundary_unsupported.sum()),
        "boundarySampleHeightRangeM": ([float(boundary_samples[:, 2].min()), float(boundary_samples[:, 2].max())]
                                        if len(boundary_samples) else []),
        "observedHostForm": host_audit,
        "analyticHostMesh": analytic_mesh_audit,
        "structuralPlaneSurface": structural_plane_audit,
        "sourceFrontOwnership": front_ownership_audit,
        "sourceGableOwnership": gable_ownership_audit,
        "provisionalFrontBoundaryVertices": int(front_provisional_mask.sum()),
        "footprintHoles": int(sum(len(p.interiors) for p in _polygons(footprint))),
    }
    return {"surface": surface, "skirt": skirt, "samples": boundary_samples,
            "boundaryChains": boundary_chains, "audit": audit,
            "nativeSupportState": native_support_state}
