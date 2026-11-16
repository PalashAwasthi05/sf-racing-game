"""Evidence admission for perimeter heights, independent of interior roof levels.

This module proposes uncertainty intervals; it never changes mesh vertices or
turns a connected interior deck into an observed outer eave. Callers retain
ownership of topology and must preserve fixed anchors when applying priors.
"""
from __future__ import annotations

import re

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def _interval(value):
    a = np.asarray(value, float)
    if a.shape != (2,) or not np.isfinite(a).all() or a[1] < a[0]:
        raise ValueError('finite ordered height interval required')
    return a


def _verified(record):
    proof = record.get('proof') or {}
    return (proof.get('status') == 'source-verified'
            and all(re.fullmatch('[0-9a-f]{64}', str(proof.get(k, '')))
                    for k in ('sourceSha256', 'metadataSha256', 'calibrationSha256')))


def perimeter_priors(xy, current_z, boundary_mask, fixed_mask, footprint, owner,
                     planes=(), source_edges=(), maximum_continuation_m=.75):
    """Return locally supported boundary intervals without modifying any height.

    Plane inputs need direct inlier ownership, an empirical height uncertainty,
    and a finite caller-supplied continuation bound. Source inputs must identify
    an actual wall/roof junction; a visible roof edge is a different role.
    Intersect compatible evidence intervals. Contradictions remain unresolved,
    and every existing fixed anchor is retained regardless of photo evidence.
    """
    xy, z = np.asarray(xy, float), np.asarray(current_z, float)
    boundary, fixed = np.asarray(boundary_mask, bool), np.asarray(fixed_mask, bool)
    if (xy.ndim != 2 or xy.shape[1] != 2 or z.shape != (len(xy),)
            or boundary.shape != z.shape or fixed.shape != z.shape
            or not np.isfinite(xy).all() or not np.isfinite(z).all()
            or not np.isfinite(maximum_continuation_m) or maximum_continuation_m < 0):
        raise ValueError('invalid boundary support arrays or continuation bound')
    reports = []
    domain = footprint.buffer(1e-8)
    for vertex in np.flatnonzero(boundary):
        point = Point(xy[vertex])
        report = {'vertex': int(vertex), 'status': 'unresolved-boundary',
                  'currentZ': float(z[vertex]), 'preferredZ': None, 'heightIntervalZ': None,
                  'exactHardConstraint': False, 'evidence': [], 'rejectedEvidence': []}
        if footprint.boundary.distance(point) > 1e-5:
            report['reason'] = 'sample does not lie on the actual owned footprint perimeter'
            reports.append(report); continue
        for index, plane in enumerate(planes):
            label = {'type': 'adjoining-lidar-plane', 'index': index}
            try:
                if plane.get('owner') != owner or plane.get('status') != 'validated-direct-inliers':
                    raise ValueError('different owner or non-direct plane support')
                support = plane['directSupport'].intersection(footprint)
                if support.is_empty:
                    raise ValueError('plane has no directly owned support')
                distance = float(support.distance(point))
                allowed = min(float(plane.get('maximumContinuationM', maximum_continuation_m)),
                              float(maximum_continuation_m))
                if not np.isfinite(allowed) or allowed < 0 or distance > allowed + 1e-8:
                    raise ValueError('interior plane is too remote to establish this perimeter')
                nearest = nearest_points(point, support)[1]
                if not domain.covers(LineString([point, nearest])):
                    raise ValueError('continuation would cross outside ownership or a courtyard')
                coefficients = np.asarray(plane['plane'], float)
                error = float(plane['heightUncertaintyM'])
                if coefficients.shape != (3,) or not np.isfinite(coefficients).all() or not np.isfinite(error) or error < 0:
                    raise ValueError('plane or height uncertainty missing')
                # Sensitivity to slope is supplied by the fit, not invented by
                # this admission helper. A missing value leaves continuation
                # uncertainty unresolved rather than silently assigning zero.
                sensitivity = plane.get('gradientUncertaintyMPerM')
                if distance > 1e-8 and (sensitivity is None or not np.isfinite(sensitivity) or sensitivity < 0):
                    raise ValueError('extrapolated plane has no quantified slope uncertainty')
                uncertainty = error + distance * float(sensitivity or 0.)
                level = float(coefficients @ np.r_[xy[vertex], 1.])
                report['evidence'].append({**label, 'heightIntervalZ': [level - uncertainty, level + uncertainty],
                    'directSupportDistanceM': distance, 'maximumContinuationM': allowed,
                    'uncertaintyM': uncertainty, 'uncertaintyMeaning': plane.get('uncertaintyMeaning', 'supplied empirical fit envelope'),
                    'sourceConfidence': plane.get('confidence'), 'cameraXYValidated': None})
            except (KeyError, ValueError, TypeError) as error:
                report['rejectedEvidence'].append({**label, 'reason': str(error)})
        for index, edge in enumerate(source_edges):
            label = {'type': 'source-eave-interval', 'index': index}
            try:
                if edge.get('owner') != owner or edge.get('role') not in ('eave', 'wall-roof-junction'):
                    raise ValueError('source is not an owned wall/roof junction')
                if not _verified(edge):
                    raise ValueError('source identities/frame/calibration were not verified')
                line = edge['geometry']
                tolerance = float(edge.get('locationUncertaintyM', 0.))
                if not np.isfinite(tolerance) or tolerance < 0:
                    raise ValueError('invalid boundary location uncertainty')
                if line.distance(point) > tolerance + 1e-7:
                    continue
                interval = _interval(edge['heightIntervalZ'])
                report['evidence'].append({**label, 'heightIntervalZ': interval.tolist(),
                    'sourceConfidence': edge.get('confidence'), 'unknownAlongEdgePosition': bool(edge.get('unknownX', False)),
                    'cameraXYValidated': (edge.get('proof') or {}).get('cameraXYValidated') is True,
                    'sourceProvenance': edge['proof'], 'uncertaintyMeaning': 'source interval, conditional on registered camera frame'})
            except (KeyError, ValueError, TypeError) as error:
                report['rejectedEvidence'].append({**label, 'reason': str(error)})
        intervals = [_interval(e['heightIntervalZ']) for e in report['evidence']]
        if fixed[vertex]:
            report.update(status='fixed-anchor-preserved', preferredZ=float(z[vertex]),
                          heightIntervalZ=[float(z[vertex])] * 2, exactHardConstraint=True)
            report['conflictingEvidenceCount'] = sum(not (a[0] - 1e-8 <= z[vertex] <= a[1] + 1e-8) for a in intervals)
        elif intervals:
            low, high = max(a[0] for a in intervals), min(a[1] for a in intervals)
            if low > high + 1e-8:
                report['reason'] = 'local boundary evidence intervals contradict; no averaging across a roof step'
                report['status'] = 'conflicting-boundary-evidence'
            else:
                high = max(low, high)
                report.update(status='supported-uncertain-boundary-prior', preferredZ=float((low + high) / 2),
                              heightIntervalZ=[float(low), float(high)], uncertaintyM=float((high - low) / 2),
                              cameraXYValidated=all(e.get('cameraXYValidated') is not False for e in report['evidence']),
                              confidenceUpgraded=False)
        else:
            report['reason'] = 'no adjoining direct plane or source-verified eave interval; interior connectivity is not eave evidence'
        reports.append(report)
    return reports


def check_silhouettes(points_xyz, silhouettes):
    """Diagnostic projection into verified, unoccluded source skyline traces.

    A point below a skyline is not validated by it. Occlusion/out-of-frame means
    unknown. With unvalidated camera positions, disagreement is conditional and
    must never move a measured roof anchor automatically.
    """
    points = np.asarray(points_xyz, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('finite world XYZ points required')
    reports = []
    for index, source in enumerate(silhouettes):
        try:
            if not _verified(source) or source.get('skyAboveConfirmed') is not True:
                raise ValueError('skyline source or sky visibility unverified')
            camera = np.asarray(source['cameraXYZ'], float)
            basis = np.asarray(source['basisRightDownForward'], float)
            size = np.asarray(source['sizePx'], float)
            focal = float(source['focalPx'])
            trace = np.asarray(source['pointsPx'], float)
            uncertainty = float(source['uncertaintyPx'])
            if (camera.shape != (3,) or basis.shape != (3, 3) or size.shape != (2,)
                    or trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 2
                    or not np.isfinite([*camera, *basis.flat, *size, focal, *trace.flat, uncertainty]).all()
                    or min(*size, focal) <= 0 or uncertainty < 0
                    or np.any(np.diff(trace[:, 0]) <= 0)
                    or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6)):
                raise ValueError('invalid actual perspective crop projection/trace')
            camera_coordinates = (points - camera) @ basis.T
            depth = camera_coordinates[:, 2]
            pixels = camera_coordinates[:, :2] / np.where(depth[:, None] > 0, depth[:, None], np.nan) * focal + size / 2 - .5
            valid_camera = (source.get('proof') or {}).get('cameraXYValidated') is True
            for point_index, (pixel, d) in enumerate(zip(pixels, depth)):
                item = {'sourceIndex': index, 'pointIndex': point_index, 'status': 'unobserved',
                        'cameraXYValidated': valid_camera, 'confidenceUpgraded': False}
                if not np.isfinite(pixel).all() or d <= 0 or np.any(pixel < 0) or np.any(pixel >= size):
                    item['reason'] = 'point behind camera or outside actual crop'
                elif not trace[0, 0] <= pixel[0] <= trace[-1, 0]:
                    item['reason'] = 'point outside the actually traced skyline span'
                elif any(a <= pixel[0] <= b for a, b in source.get('occludedXIntervalsPx', [])):
                    item['reason'] = 'source skyline is occluded at this horizontal position'
                else:
                    upper = float(np.interp(pixel[0], trace[:, 0], trace[:, 1]))
                    excess = float(upper - pixel[1] - uncertainty)
                    item.update(pixelPx=pixel.tolist(), tracedSkylineYPx=upper,
                                uncertaintyPx=uncertainty, excessIntoVerifiedSkyPx=max(0., excess),
                                status=('skyline-conflict' if valid_camera else 'conditional-skyline-conflict')
                                       if excess > 0 else 'not-rejected-by-skyline',
                                qualifiesRoofHeight=False, movesMeasuredAnchors=False)
                reports.append(item)
        except (KeyError, TypeError, ValueError) as error:
            reports.append({'sourceIndex': index, 'status': 'unresolved-source', 'reason': str(error)})
    return reports

def continue_adjoining_planes(footprint, raw, xy, z, regions, unknown, fixed, cell_m):
    """Experimental bounded affine priors for unknown perimeter strips beside real planes.

    Mixed edge cells can have a high median from a neighbor/fixture although
    several actual returns still lie on the independently validated host
    slope. Use those plane-specific returns as local support, never the high
    cell summary. Six occupied cells and a non-collinear local patch must
    support a candidate within two metres, and its local inlier hull may be
    continued by at most one cell. Interior decks do not own remote eaves.
    These are provisional continuations, not additional measured anchors.
    """
    from shapely.geometry import MultiPoint
    from scipy.spatial import cKDTree
    from .roof_shell import _aggregate
    from shapely.ops import nearest_points

    report = {'status': 'no-adjoining-plane-continuation', 'continuedVertices': 0,
              'maximumLocalSupportRadiusM': max(2., 3 * cell_m),
              'maximumInlierHullExtrapolationM': float(cell_m),
              'fixedAnchorsPreserved': True, 'exactMeasurementsAdded': 0,
              'candidateConflicts': 0, 'planes': []}
    raw = np.asarray(raw, float)
    if not regions or not len(raw):
        return z, report
    direct = [r for r in regions if r.get('ownership') != 'bounded-opposing-plane-interpolation']
    if not direct:
        return z, report
    planes = np.array([r['plane'] for r in direct])
    residual = abs(np.c_[raw[:, :2], np.ones(len(raw))] @ planes.T - raw[:, 2, None])
    owners = np.argmin(residual, axis=1)
    best = residual[np.arange(len(raw)), owners]
    distance = np.array([footprint.boundary.distance(Point(p)) for p in xy])
    targets = np.flatnonzero(np.asarray(unknown, bool) & ~np.asarray(fixed, bool) & (distance <= cell_m + 1e-8))
    candidates = [[] for _ in xy]
    radius = report['maximumLocalSupportRadiusM']
    owned_domain = footprint.buffer(1e-8)
    for index, region in enumerate(direct):
        # Actual raw inliers recover a lower host surface even when an upper
        # return would dominate the ordinary top-half cell statistic.
        cloud = _aggregate(raw[(owners == index) & (best <= .15)], cell_m)
        plane_report = {'plane': region['plane'].tolist(), 'actualInlierCells': len(cloud),
                        'continuedVertices': 0, 'maximumEmpiricalP95ResidualM': None,
                        'maximumInlierHullExtrapolationM': None,
                        'uncertaintyMeaning': 'local empirical residual and bounded support distances; systematic LiDAR/footprint error unquantified'}
        report['planes'].append(plane_report)
        if len(cloud) < 6:
            continue
        tree = cKDTree(cloud[:, :2])
        for vertex in targets:
            ids = tree.query_ball_point(xy[vertex], radius)
            if len(ids) < 6:
                continue
            support = cloud[ids]
            spread = np.linalg.svd(support[:, :2] - support[:, :2].mean(0), compute_uv=False) / np.sqrt(len(ids))
            if spread[-1] < .15 * cell_m:
                continue
            hull = MultiPoint(support[:, :2]).convex_hull.intersection(footprint)
            extrapolation = float(hull.distance(Point(xy[vertex])))
            if extrapolation > cell_m + 1e-8:
                continue
            nearest = nearest_points(Point(xy[vertex]), hull)[1]
            if not owned_domain.covers(LineString([xy[vertex], nearest.coords[0]])):
                continue
            # The local cloud must join this plane's originally validated
            # patch; a remote accidental coplanar group is not a roof anchor.
            if region['geometry'].distance(hull) > cell_m + 1e-8:
                continue
            prediction = float(region['plane'] @ np.r_[xy[vertex], 1.])
            errors = abs(np.c_[support[:, :2], np.ones(len(support))] @ region['plane'] - support[:, 2])
            p95 = float(np.quantile(errors, .95))
            candidates[vertex].append((prediction, index, p95, extrapolation))
    out = np.asarray(z, float).copy()
    for vertex in targets:
        values = candidates[vertex]
        if not values:
            continue
        levels = [v[0] for v in values]
        if np.ptp(levels) > max(.3, 2 * max(v[2] for v in values)):
            report['candidateConflicts'] += 1
            continue
        prediction, index, p95, extrapolation = min(values, key=lambda v: (v[3], v[2], v[1]))
        out[vertex] = prediction
        record = report['planes'][index]
        record['continuedVertices'] += 1
        record['maximumEmpiricalP95ResidualM'] = max(record['maximumEmpiricalP95ResidualM'] or 0., p95)
        record['maximumInlierHullExtrapolationM'] = max(record['maximumInlierHullExtrapolationM'] or 0., extrapolation)
        report['continuedVertices'] += 1
    if report['continuedVertices']:
        report['status'] = 'bounded-local-inlier-plane-priors'
    return out, report

