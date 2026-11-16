"""Bounded roof faces measured in matched, registered source photographs.

This is independent of whole-building roof styles and nadir point density. The
corners are recomputed from source pixels; cached XYZ never drives geometry.
Camera position uncertainty remains explicit even when pixel rays agree.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import trimesh
from scipy.spatial import QhullError
from shapely.geometry import Polygon


def _array(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(label + ' is malformed')
    return a


def _ray(source, pixel):
    frame = source['frame']
    normal = _array(frame['normal'], (2,), 'source normal')
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError('source normal has zero length')
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    u = _array(frame['frameU'], (2,), 'source frameU')
    ppm, pad = float(source['pixelsPerMetre']), float(source['padPx'])
    if (not np.isfinite([ppm, pad, frame['heightM'], frame['lengthM'], frame['groundZ'], frame['planeV']]).all()
            or min(ppm, frame['heightM'], frame['lengthM']) <= 0 or pad < 0
            or not np.isclose(u[1] - u[0], frame['lengthM'], atol=.02)):
        raise ValueError('invalid source metric frame')
    # This is the actual crop sampler, including its 5 cm facade-plane offset.
    x = (pixel[0] - pad + .5) / ppm - float(frame['lengthM']) / 2
    z = float(frame['heightM']) - (pixel[1] - pad + .5) / ppm
    point = np.r_[normal * (float(frame['planeV']) + .05) + tangent * (u.mean() + x),
                  float(frame['groundZ']) + z]
    camera = _array(source['cameraXYZ'], (3,), 'source camera')
    direction = point - camera
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError('source ray has zero length')
    return camera, direction / np.linalg.norm(direction)


def _intersection(ray0, ray1):
    c0, d0 = ray0
    c1, d1 = ray1
    system = np.c_[d0, -d1]
    if np.linalg.cond(system) > 1e4:
        raise ValueError('matched rays are too nearly parallel to bound a roof point')
    distance, *_ = np.linalg.lstsq(system, c1 - c0, rcond=None)
    if np.any(distance <= 0):
        raise ValueError('matched point is behind a source camera')
    p0, p1 = c0 + distance[0] * d0, c1 + distance[1] * d1
    return (p0 + p1) / 2, float(np.linalg.norm(p0 - p1)), distance


def derive(part):
    """Reconstruct one bounded convex planar face from attributed stereo picks.

    ``proof`` is produced only after the adapter verifies on-disk identities.
    Runtime validates its schema and recomputes rays, uncertainty, and planarity.
    No cached vertices, pitch, roof-style, or supplied metric dimensions are used.
    """
    if part.get('kind') != 'bounded-source-plane' or part.get('confidence') not in ('medium', 'high'):
        raise ValueError('roof part is not a confident bounded source plane')
    proof = part.get('proof') or {}
    if proof.get('status') != 'source-verified' or proof.get('sourcePixelsVisuallyInspected') is not True:
        raise ValueError('roof part lacks inspected, source-verified evidence')
    sources = part.get('sources', [])
    by_key = {}
    for source in sources:
        key = source.get('sourceKey')
        if not key or key in by_key:
            raise ValueError('missing or ambiguous source identity')
        for field in ('sourceSha256', 'metadataSha256', 'calibrationSha256'):
            if not re.fullmatch('[0-9a-f]{64}', str(source.get(field, ''))):
                raise ValueError('source proof lacks ' + field)
        if source.get('sourceAlreadyGravityCalibrated') is not True or source.get('sampleConvention') != 'pixel-center':
            raise ValueError('source crop calibration or sample convention unresolved')
        size = _array(source.get('sizePx'), (2,), 'source image size')
        if np.any(size <= 0):
            raise ValueError('source image size must be positive')
        by_key[key] = source
    if len(by_key) < 2 or len({str(s.get('image')) for s in sources}) < 2:
        raise ValueError('two distinct source images are required')
    corners, records = {}, []
    for corner in part.get('corners', []):
        name = corner.get('name')
        if not name or name in corners or not corner.get('role'):
            raise ValueError('corner has missing or ambiguous attributed role')
        matches = corner.get('matches', [])
        if len(matches) != 2 or matches[0].get('sourceKey') == matches[1].get('sourceKey'):
            raise ValueError('each corner needs exactly two different source matches')
        contexts, rays, errors, pixels = [], [], [], []
        for match in matches:
            source = by_key.get(match.get('sourceKey'))
            if source is None:
                raise ValueError('corner references unknown source')
            pixel = _array(match.get('pointPx'), (2,), 'corner pixel')
            error = float(match.get('uncertaintyPx', -1))
            if not np.isfinite(error) or error <= 0:
                raise ValueError('approximate corner must carry positive pixel uncertainty')
            if np.any(pixel < 0) or np.any(pixel >= np.asarray(source['sizePx'])):
                raise ValueError('corner lies outside its actual source crop')
            contexts.append(source); pixels.append(pixel); errors.append(error)
            rays.append(_ray(source, pixel))
        point, gap, ranges = _intersection(*rays)
        cone_radius = 0.
        for source, pixel, error, ray, distance in zip(contexts, pixels, errors, rays, ranges):
            # A source-specific angular cone, not a universal metre threshold.
            angle = max(np.arccos(np.clip(ray[1] @ _ray(source, pixel + np.asarray(offset) * error)[1], -1, 1))
                        for offset in itertools.product((-1, 1), repeat=2))
            cone_radius += distance * np.tan(angle)
        if gap > cone_radius + 1e-6:
            raise ValueError('matched rays disagree beyond their pixel-reading cones')
        samples = []
        for offsets in itertools.product((-1, 1), repeat=4):
            perturbed = [_ray(s, p + np.asarray(offsets[2*i:2*i+2]) * e)
                         for i, (s, p, e) in enumerate(zip(contexts, pixels, errors))]
            samples.append(_intersection(*perturbed)[0])
        samples = np.asarray(samples)
        bounds = np.array([np.minimum(samples.min(0), point), np.maximum(samples.max(0), point)])
        corners[name] = point
        records.append({'name': name, 'role': corner['role'], 'pointXYZ': point.tolist(),
                        'raySeparationM': gap, 'pixelRayConeRadiusSumM': float(cone_radius),
                        'rayAngleDeg': float(np.degrees(np.arccos(np.clip(rays[0][1] @ rays[1][1], -1, 1)))),
                        'pixelOnlyBoundsXYZM': bounds.tolist(),
                        'pixelOnlyUncertaintyXYZM': np.max(abs(bounds - point), axis=0).tolist(),
                        'cameraPositionUncertaintyM': None})
    order = part.get('cornerOrder', [])
    if len(order) < 3 or len(set(order)) != len(order) or set(order) != set(corners):
        raise ValueError('corner order must use every attributed corner exactly once')
    points = np.array([corners[k] for k in order])
    # A wide uncertainty cone must not make a weak stereo baseline easier to
    # accept. Bound pixel-only corner uncertainty relative to the measured
    # face's smallest actual edge before fitting its plane. At a quarter edge
    # radius, adjacent corner ranges already consume half that edge.
    edge_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    shortest_edge = float(edge_lengths.min())
    radii = np.array([np.linalg.norm(next(r for r in records if r['name'] == key)['pixelOnlyUncertaintyXYZM']) for key in order])
    relative_uncertainty = float(radii.max() / max(shortest_edge, 1e-12))
    if relative_uncertainty > .25:
        raise ValueError('weak stereo triangulation: pixel-only corner uncertainty exceeds one quarter of the shortest observed roof edge')
    center = points.mean(0)
    _, singular, vectors = np.linalg.svd(points - center)
    if singular[1] < 1e-5:
        raise ValueError('roof part corners are collinear')
    normal = vectors[-1]
    residual = abs((points - center) @ normal)
    uncertainty = {r['name']: np.asarray(r['pixelOnlyUncertaintyXYZM']) for r in records}
    tolerances = np.array([abs(normal) @ uncertainty[k] for k in order])
    if np.any(residual > tolerances + 1e-6):
        raise ValueError('corner picks do not support one plane within pixel uncertainty')
    if abs(normal[2]) < 1e-4:
        raise ValueError('a vertical source plane is not a roof face')
    projected = points - np.outer((points - center) @ normal, normal)
    polygon = Polygon(projected[:, :2])
    if not polygon.is_valid or polygon.area <= 1e-5 or polygon.convex_hull.area - polygon.area > 1e-6:
        raise ValueError('roof part must be a nonempty convex bounded face')
    edge = part.get('lowerEdgeCornerNames', [])
    if len(edge) != 2 or edge[0] == edge[1] or any(k not in corners for k in edge):
        raise ValueError('roof part needs two attributed lower closure edge corners')
    i0, i1 = map(order.index, edge)
    if (i0 - i1) % len(order) not in (1, len(order) - 1):
        raise ValueError('lower closure edge is not an actual polygon edge')
    plane = np.r_[-normal[:2] / normal[2], (normal @ center) / normal[2]]
    a, b = projected[i0], projected[i1]
    delta = b[:2] - a[:2]
    if delta @ delta < 1e-8:
        raise ValueError('lower edge is vertical or degenerate')
    gradient = delta * (b[2] - a[2]) / (delta @ delta)
    closure = np.r_[gradient, a[2] - gradient @ a[:2]]
    base = np.c_[projected[:, :2], np.c_[projected[:, :2], np.ones(len(order))] @ closure]
    if np.any(base[:, 2] > projected[:, 2] + 1e-6) or np.max(projected[:, 2] - base[:, 2]) < 1e-5:
        raise ValueError('attributed lower edge does not bound a positive roof wedge')
    report = {'status': 'conditional-source-plane', 'confidence': part['confidence'],
              'cameraXYValidated': all(s.get('cameraXYValidated') is True for s in sources),
              'cameraUncertaintyIncluded': False, 'confidenceUpgraded': False,
              'corners': records, 'verticesXYZ': projected.tolist(), 'planeZCoefficients': plane.tolist(),
              'maximumPlanarityResidualM': float(residual.max()),
              'shortestObservedEdgeM': shortest_edge,
              'maximumPixelOnlyCornerRadiusM': float(radii.max()),
              'maximumPixelUncertaintyToShortestEdgeRatio': relative_uncertainty,
              'maximumAcceptedPixelUncertaintyToEdgeRatio': .25,
              'pitchDeg': float(np.degrees(np.arctan(np.linalg.norm(plane[:2])))),
              'horizontalCameraBaselinesM': [float(np.linalg.norm(np.asarray(s0['cameraXYZ'])[:2] - np.asarray(s1['cameraXYZ'])[:2]))
                  for s0, s1 in itertools.combinations(sources, 2)],
              'closure': 'provisional return to datum through observed lower edge; no hidden rear pitch inferred'}
    return {'polygon': polygon, 'vertices': projected, 'baseVertices': base,
            'plane': plane, 'part': part, 'audit': report}


def resolve(observations, profiles, part_id, source_id, footprint):
    """Use the same exact side/part join as independently observed components."""
    from .roof_assemblies import _profiles_for_side
    accepted, audit = [], []
    for observation in observations:
        parts = observation.get('roofParts') or []
        if not parts:
            continue
        joined = _profiles_for_side(profiles, observation.get('side'), part_id, source_id)
        if not joined:
            continue
        for index, part in enumerate(parts):
            record = {'side': observation.get('side'), 'partIndex': index}
            try:
                if len(joined) != 1:
                    raise ValueError('roof part has ambiguous exact profile ownership')
                item = derive(part)
                profile = joined[0]
                for source in part['sources']:
                    if source.get('side') != observation.get('side'):
                        raise ValueError('roof part source side differs from joined observation')
                    for key in ('normal', 'frameU', 'planeV'):
                        if not np.allclose(profile.get(key), source['frame'][key], atol=1e-4, rtol=0):
                            raise ValueError('source roof part and runtime profile frames differ')
                polygon = item['polygon']
                if polygon.intersection(footprint).area <= 1e-6:
                    raise ValueError('source roof part has no owned host-footprint intersection')
                for piece in getattr(footprint, 'geoms', [footprint]):
                    for hole in piece.interiors:
                        if polygon.intersection(Polygon(hole)).area > 1e-6:
                            raise ValueError('source roof part would cover an existing courtyard')
                item['audit'] = {**record, **item['audit']}
                accepted.append(item)
                audit.append(item['audit'])
            except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as error:
                audit.append({**record, 'status': 'unresolved', 'reason': str(error)})
    return accepted, audit


def compose(shell, parts, neighbor_footprints=()):
    """Add only measured faces; keep the native roof outside their solid wedges.

    The lower-edge datum closes the new part without extending wall-top chains.
    Hidden closures are provisional. Existing supported higher surfaces survive.
    """
    from .roof_shell import clip_component_intersections
    reports = []
    for item in parts:
        polygon, vertices, base = item['polygon'], item['vertices'], item['baseVertices']
        record = dict(item['audit'])
        if any(polygon.intersection(g).area > 1e-6 for g in neighbor_footprints):
            reports.append({**record, 'status': 'unresolved', 'reason': 'measured roof face overlaps a separately owned building'})
            continue
        try:
            solid = trimesh.Trimesh(vertices=np.vstack([vertices, base]), faces=[], process=False).convex_hull
        except (ValueError, RuntimeError, QhullError) as error:
            reports.append({**record, 'status': 'unresolved', 'reason': 'bounded roof wedge failed: ' + str(error)[:200]})
            continue
        if not solid.is_volume:
            reports.append({**record, 'status': 'unresolved', 'reason': 'bounded roof wedge is not a closed positive volume'})
            continue
        top_distance = solid.triangles[:, :, 2] - (solid.triangles[:, :, :2] @ item['plane'][:2] + item['plane'][2])
        is_top = np.max(abs(top_distance), axis=1) < 1e-6
        top = solid.submesh([np.flatnonzero(is_top)], append=True)
        returns = solid.submesh([np.flatnonzero(~is_top)], append=True)
        normals = np.asarray(solid.face_normals)
        planes = np.c_[normals, np.einsum('ij,ij->i', normals, solid.triangles_center)]
        _, unique = np.unique(np.round(planes, 8), axis=0, return_index=True)
        cutters = [{'planes': planes[np.sort(unique)], 'bounds': solid.bounds}]
        before = float(shell['surface'].area)
        surface = clip_component_intersections(shell['surface'], cutters)
        skirt = clip_component_intersections(shell.get('skirt'), cutters)
        shell['surface'] = trimesh.util.concatenate([surface, top])
        shell['skirt'] = trimesh.util.concatenate([g for g in (skirt, returns) if g is not None and len(g.faces)])
        shell['surface'].metadata['boundedSourceRoofParts'] = True
        record.update(status='applied-conditional-source-plane', measuredFaceAreaM2=float(top.area),
                      nativeRoofRemovedM2=max(0., before - float(surface.area)),
                      nativeBoundaryChainsUnchanged=True, nativeRoofOutsidePartUnchanged=True,
                      conditionalMeasuredOverhangAllowed=True)
        reports.append(record)
    shell['audit']['boundedSourceRoofParts'] = reports
    shell['audit']['surfaceVertexCount'] = len(shell['surface'].vertices)
    shell['audit']['surfaceFaceCount'] = len(shell['surface'].faces)
    return shell
