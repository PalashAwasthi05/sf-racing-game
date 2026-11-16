"""Photo-space ownership for repeated readings of the same facade windows.

Profiles from different crops can place the same photographed windows on nearby
parallel footprint faces.  This module projects their measured window boxes
through the shared source panorama, requires repeated image-space agreement,
then checks whether the front wall actually blocks the rear candidate.  It does
not merge materials, doors, or windows that remain visible through a setback.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial.transform import Rotation


def _unit(v):
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else None


def _uv(direction, rotation):
    """World direction to equirectangular pano UV using camera=(right,down,forward)."""
    d = np.asarray(direction, dtype=float) @ np.asarray(rotation, dtype=float).T
    x, y, z = d
    lon = math.atan2(float(x), float(z))
    lat = math.atan2(float(-y), math.hypot(float(x), float(z)))
    return (lon / (2 * math.pi) + .5) % 1., .5 - lat / math.pi


def _opening_features(profile):
    """Window rectangles in the profile's normalized [x,z] frame with stable source refs."""
    out = []
    for gi, group in enumerate(profile.get('groups') or []):
        style = group.get('style') or {}
        if style.get('kind', 'window') != 'window':
            continue
        region = group.get('region')
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            continue
        x0, y0, x1, y1 = map(float, region)
        nx, ny = max(1, int(group.get('columns', 1))), max(1, int(group.get('rows', 1)))
        xs = group.get('xCenters') or [(i + .5) / nx for i in range(nx)]
        ys = group.get('yCenters') or [(i + .5) / ny for i in range(ny)]
        dx, dy = float(group.get('spacingX', 1 / nx)), float(group.get('spacingY', 1 / ny))
        ow = dx * (x1 - x0) * float(style.get('openingRatio', .7))
        oh = dy * (y1 - y0) * float(style.get('openingHeightRatio', .7))
        omitted = {tuple(q) for q in group.get('omit') or [] if isinstance(q, (list, tuple)) and len(q) == 2}
        for ix, cx in enumerate(xs):
            for iy, cy in enumerate(ys):
                if (ix, iy) in omitted:
                    continue
                mx, mz = x0 + float(cx) * (x1 - x0), y0 + float(cy) * (y1 - y0)
                out.append({'ref': ('group', gi, ix, iy), 'rect': (mx - ow / 2, mz - oh / 2, mx + ow / 2, mz + oh / 2),
                            'head': style.get('head', 'flat')})
    for oi, opening in enumerate(profile.get('openings') or []):
        style = opening.get('style') or {}
        if style.get('kind', 'window') != 'window':
            continue
        rect = opening.get('rect')
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            out.append({'ref': ('opening', oi), 'rect': tuple(map(float, rect)), 'head': style.get('head', 'flat')})
    # A group and explicit opening can describe the same light. Keep both refs
    # so filtering either representation removes its duplicate emission.
    dedup = []
    for item in out:
        duplicate = next((q for q in dedup if _rect_iou(q['rect'], item['rect']) > .92), None)
        if duplicate:
            duplicate.setdefault('refs', [duplicate['ref']]).append(item['ref'])
        else:
            dedup.append({**item, 'refs': [item['ref']]})
    return dedup


def _rect_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    inter = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    ia = float(np.prod(inter))
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ia)
    return ia / union if union > 1e-10 else 0.


def _world_corners(profile, ground_z, rect):
    normal = _unit(profile.get('normal', ()))
    frame = profile.get('frameU')
    if normal is None or not isinstance(frame, (list, tuple)) or len(frame) != 2:
        return None
    length = float(frame[1]) - float(frame[0])
    height = float(profile.get('heightRefM') or 0.)
    plane = profile.get('planeV')
    if length <= 0 or height <= 0 or not isinstance(plane, (int, float)):
        return None
    tangent = np.array([-normal[1], normal[0]])
    x0, z0, x1, z1 = rect
    return [np.r_[tangent * (float(frame[0]) + x * length) + normal * float(plane), float(ground_z) + z * height]
            for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]


def project_profile_windows(profile, ground_z, camera):
    """Project one profile's measured window footprints into a source pano.

    `camera` requires `center:[x,y,z]`, `rotation:3x3`, and `imageId`. Returns
    normalized panorama boxes; x wraps at the seam. The function is pure so an
    offline adapter and the runtime selector use identical geometry.
    """
    center = np.asarray(camera.get('center'), float)
    rotation = np.asarray(camera.get('rotation'), float)
    if center.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(center).all() or not np.isfinite(rotation).all():
        return []
    result = []
    for feature in _opening_features(profile):
        corners = _world_corners(profile, ground_z, feature['rect'])
        if not corners:
            continue
        uv = np.asarray([_uv(p - center, rotation) for p in corners], float)
        ux = uv[:, 0]
        ux = ux[0] + (ux - ux[0] + .5) % 1. - .5
        box = [float(ux.min()), float(uv[:, 1].min()), float(ux.max()), float(uv[:, 1].max())]
        if box[2] - box[0] <= 1e-7 or box[3] - box[1] <= 1e-7:
            continue
        result.append({'imageId': str(camera['imageId']), 'box': box, 'rect': feature['rect'],
                       'refs': feature['refs'], 'head': feature['head']})
    return result


def _iou_box(a, b):
    return _rect_iou(a, b)


def match_projected_windows(a, b, *, min_matches=2, center_width_ratio=.55, center_height_ratio=.35,
                            size_ratio=(.78, 1.28)):
    """One-to-one matches for same-pano window boxes; reject nearest-neighbour guesses.

    Center offsets are normalized by the mean projected opening size. A 0.4
    width residual is tolerated for registered-camera uncertainty only when
    the opening dimensions also agree. The caller still must prove front-wall
    occlusion before removing any geometry.
    """
    if not a or not b or a[0].get('imageId') != b[0].get('imageId'):
        return []
    candidates = []
    for i, x in enumerate(a):
        ax = np.asarray(x['box'], float)
        aw, ah = ax[2] - ax[0], ax[3] - ax[1]
        ac = (ax[:2] + ax[2:]) / 2
        for j, y in enumerate(b):
            bx = np.asarray(y['box'], float)
            bw, bh = bx[2] - bx[0], bx[3] - bx[1]
            if min(aw, ah, bw, bh) <= 1e-8:
                continue
            rw, rh = aw / bw, ah / bh
            if not size_ratio[0] <= rw <= size_ratio[1] or not size_ratio[0] <= rh <= size_ratio[1]:
                continue
            bc = (bx[:2] + bx[2:]) / 2
            du = abs(float(ac[0] - bc[0]))
            du = min(du, 1. - du)
            dv = abs(float(ac[1] - bc[1]))
            if du > center_width_ratio * (aw + bw) / 2 or dv > center_height_ratio * (ah + bh) / 2:
                continue
            score = (du / ((aw + bw) / 2)) ** 2 + (dv / ((ah + bh) / 2)) ** 2 + math.log(rw) ** 2 + math.log(rh) ** 2
            candidates.append((score, i, j, du, dv, rw, rh))
    matches = []
    used_a, used_b = set(), set()
    for score, i, j, du, dv, rw, rh in sorted(candidates):
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        matches.append({'a': a[i], 'b': b[j], 'centerDeltaU': du, 'centerDeltaV': dv,
                        'widthRatio': rw, 'heightRatio': rh, 'score': score})
    if len(matches) < min_matches:
        return []
    # Two matches must identify distinct vertical features, not two duplicate
    # representations of one window in the same storey.
    heights_a = [float((m['a']['box'][1] + m['a']['box'][3]) / 2) for m in matches]
    if max(heights_a) - min(heights_a) < .5 * float(np.median([m['a']['box'][3] - m['a']['box'][1] for m in matches])):
        return []
    return matches


def _inside_head(rect, head, x, z, style=None):
    x0, z0, x1, z1 = rect
    if not (x0 <= x <= x1 and z0 <= z <= z1):
        return False
    if head == 'flat':
        return True
    xx = (x - x0) / max(x1 - x0, 1e-9)
    zz = (z - z0) / max(z1 - z0, 1e-9)
    if head == 'arch':
        spring = float((style or {}).get('springRatio', .68))
        edge = spring + (1. - spring) * math.sqrt(max(0., 1. - (2. * xx - 1.) ** 2))
        return zz <= edge
    if head == 'segmental':
        rise = float((style or {}).get('riseRatio', .14))
        edge = (1. - rise) + rise * math.sqrt(max(0., 1. - (2. * xx - 1.) ** 2))
        return zz <= edge
    if head == 'pointed':
        spring = float((style or {}).get('springRatio', .68))
        edge = spring + (1. - spring) * (2. * xx if xx <= .5 else 2. * (1. - xx))
        return zz <= edge
    return True


def wall_opening_at(profile, x_world, z_world, ground_z, *, camera_center=None, target=None, plane_v=None):
    """Whether a ray hit falls in a genuinely open portal in the front wall.

    Compiled windows are backed by a deep interior block and a glazing pane
    (see the window emission in facade_grammar.py); their rectangular aperture
    is therefore not a view-through portal for ownership checks. Entries,
    passages, and open recesses remain pass-through.
    """
    normal = _unit(profile.get('normal', ()))
    frame = profile.get('frameU')
    if normal is None or not frame:
        return False
    tangent = np.array([-normal[1], normal[0]])
    x = float(np.asarray(x_world, float)[:2] @ tangent) - float(frame[0])
    z = float(z_world) - float(ground_z)
    length = float(frame[1]) - float(frame[0]); height = float(profile.get('heightRefM') or 0.)
    if length <= 0 or height <= 0:
        return False
    xn, zn = x / length, z / height
    # A door or passage can expose a farther surface, even though doors are
    # never used as evidence that two readings describe the same feature.
    for opening in profile.get('openings') or []:
        style = opening.get('style') or {}
        if style.get('kind') not in ('entry', 'door', 'garage', 'rollup', 'passage'):
            continue
        rect = opening.get('rect')
        if isinstance(rect, (list, tuple)) and len(rect) == 4 and rect[0] <= xn <= rect[2] and rect[1] <= zn <= rect[3]:
            return True
    # A recess has a real back surface. Treat only an explicit glazed back
    # opening as view-through; a blank loggia/recess back wall blocks the ray.
    for recess in profile.get('recesses') or []:
        rect = recess.get('rect')
        if not (isinstance(rect, (list, tuple)) and len(rect) == 4 and rect[0] <= xn <= rect[2] and rect[1] <= zn <= rect[3]):
            continue
        back_xn, back_zn = xn, zn
        if camera_center is not None and target is not None and isinstance(plane_v, (int, float)):
            back_hit = ray_intersection(camera_center, target, normal, float(plane_v) - float(recess.get('depthM') or 0.))
            if back_hit is None:
                continue
            back_u = float(back_hit[:2] @ tangent)
            back_xn = (back_u - float(frame[0])) / length
            back_zn = (float(back_hit[2]) - float(ground_z)) / height
        for opening in recess.get('backOpenings') or []:
            bounds = opening.get('rect')
            if (isinstance(bounds, (list, tuple)) and len(bounds) == 4 and
                    bounds[0] <= back_xn <= bounds[2] and bounds[1] <= back_zn <= bounds[3] and
                    opening.get('kind', 'window') in ('window', 'glazed_door')):
                return True
        return False
    return False


def ray_intersection(camera_center, target, normal, plane_v):
    """Intersect a camera-to-target segment with n dot xy = planeV; None if not between."""
    c, p = np.asarray(camera_center, float), np.asarray(target, float)
    n = _unit(normal)
    if c.shape != (3,) or p.shape != (3,) or n is None:
        return None
    den = float(p[:2] @ n - c[:2] @ n)
    if abs(den) < 1e-9:
        return None
    t = (float(plane_v) - float(c[:2] @ n)) / den
    if not 0. < t < 1.:
        return None
    return c + t * (p - c)


def _feature_center_world(profile, ground_z, feature):
    corners = _world_corners(profile, ground_z, feature['rect'])
    if not corners:
        return None
    return np.mean(np.asarray(corners), axis=0)


def _polygons(geometry):
    if geometry is None:
        return []
    if getattr(geometry, 'geom_type', None) == 'Polygon':
        return [geometry]
    return [p for child in getattr(geometry, 'geoms', []) for p in _polygons(child)]


def _frontmost_host_plane(geometry, front, far, front_features):
    """Find the visible frontmost footprint edge at each matched window x."""
    normal = _unit(front.get('normal', ()))
    frame = front.get('frameU')
    if normal is None or not frame:
        return None
    tangent = np.array([-normal[1], normal[0]])
    length = float(frame[1]) - float(frame[0])
    target_us = [float(frame[0]) + (f['rect'][0] + f['rect'][2]) * .5 * length for f in front_features]
    plane_lo = min(float(front.get('planeV', 0.)), float(far.get('planeV', 0.))) - 2.5
    plane_hi = max(float(front.get('planeV', 0.)), float(far.get('planeV', 0.))) + 2.5
    line_candidates = []
    for poly in _polygons(geometry):
        for ring in (poly.exterior, *poly.interiors):
            coords = list(ring.coords)
            for aa, bb in zip(coords, coords[1:]):
                a, b = np.asarray(aa, float), np.asarray(bb, float)
                edge = b - a
                span = float(np.linalg.norm(edge))
                if span < .25 or abs(float(edge / span @ tangent)) < .98:
                    continue
                outward = np.array([edge[1], -edge[0]]) / span
                mid = (a + b) * .5
                if poly.contains(Point(*(mid + outward * .05))):
                    outward = -outward
                if float(outward @ normal) < .98:
                    continue
                plane = float(mid @ normal)
                if not plane_lo <= plane <= plane_hi:
                    continue
                us = sorted([float(a @ tangent), float(b @ tangent)])
                if any(us[0] - .1 <= target <= us[1] + .1 for target in target_us):
                    line_candidates.append((plane, us))
    owners = []
    for target in target_us:
        options = [plane for plane, us in line_candidates if us[0] - .1 <= target <= us[1] + .1]
        if not options:
            return None
        owners.append(max(options))
    return owners


def duplicate_window_observations(profile_a, profile_b, projected_a, projected_b, ground_a, ground_b,
                                  camera, *, host_geometry=None, min_matches=2):
    """Return a proof record only if repeated windows match and the nearer wall occludes the farther ones.

    Registered camera position uncertainty is handled by the image-space
    residual threshold, but a matched window pair is suppressed only when the
    ray to the farther opening hits opaque material on the front wall for every
    matched storey. If the ray passes through a front opening, the set-back
    window remains an independent visible feature.
    """
    na, nb = _unit(profile_a.get('normal', ())), _unit(profile_b.get('normal', ()))
    if na is None or nb is None or float(na @ nb) < .98:
        return None
    same_image = sorted({f['imageId'] for f in projected_a} & {f['imageId'] for f in projected_b})
    if not same_image:
        return None
    # Projection lists are image-specific; require a repeated match in one exact pano.
    pa = [f for f in projected_a if f['imageId'] == same_image[0]]
    pb = [f for f in projected_b if f['imageId'] == same_image[0]]
    matches = match_projected_windows(pa, pb, min_matches=min_matches)
    if not matches:
        return None
    center = np.asarray(camera.get('center'), float)
    if center.shape != (3,):
        return None
    # Select the candidate wall nearer the common source camera, consistently
    # across the matched openings.
    dist_a, dist_b = [], []
    for m in matches:
        xa = _feature_center_world(profile_a, ground_a, m['a'])
        xb = _feature_center_world(profile_b, ground_b, m['b'])
        if xa is None or xb is None:
            return None
        dist_a.append(float(np.linalg.norm(xa - center)))
        dist_b.append(float(np.linalg.norm(xb - center)))
    a_front = all(a < b - .08 for a, b in zip(dist_a, dist_b))
    b_front = all(b < a - .08 for a, b in zip(dist_a, dist_b))
    if not (a_front or b_front):
        return {'status': 'conflict', 'reason': 'matched windows do not have a consistent source-camera depth order',
                'matches': matches, 'imageId': same_image[0]}
    front, far = (profile_a, profile_b) if a_front else (profile_b, profile_a)
    front_gz, far_gz = (ground_a, ground_b) if a_front else (ground_b, ground_a)
    front_features = [m['a'] if a_front else m['b'] for m in matches]
    far_features = [m['b'] if a_front else m['a'] for m in matches]
    owner_planes = _frontmost_host_plane(host_geometry, front, far, front_features) if host_geometry is not None else None
    if owner_planes is None or len(owner_planes) != len(matches):
        return {'status': 'conflict', 'reason': 'no unambiguous measured frontmost footprint edge covers matched windows',
                'matches': matches, 'imageId': same_image[0], 'frontProfileId': front.get('id'),
                'farProfileId': far.get('id')}
    blocked = []
    visible = []
    for m, near_feature, far_feature, owner_plane in zip(matches, front_features, far_features, owner_planes):
        far_center = _feature_center_world(far, far_gz, far_feature)
        if far_center is None:
            return {'status': 'conflict', 'reason': 'malformed far opening projection', 'matches': matches, 'imageId': same_image[0]}
        hit = ray_intersection(center, far_center, front['normal'], owner_plane)
        if hit is None:
            return {'status': 'conflict', 'reason': 'source ray does not cross the candidate front wall', 'matches': matches, 'imageId': same_image[0]}
        if host_geometry is not None and not host_geometry.buffer(.025).covers(Point(float(hit[0]), float(hit[1]))):
            visible.append({'match': m, 'reason': 'source ray misses the measured host footprint',
                            'frontHitWorld': hit.tolist(), 'ownerPlaneV': float(owner_plane)})
            continue
        fn = _unit(front['normal']); ft = np.array([-fn[1], fn[0]])
        u = float(hit[:2] @ ft)
        f0, f1 = sorted(map(float, front.get('frameU', (0., 0.))))
        x = u - f0
        reading = front.get('reading') or {}
        top_value = front.get('opaqueWallTopM', front.get('observedWallTopM', reading.get('wallTopM')))
        if (reading.get('confidence') not in ('medium', 'high') or not isinstance(top_value, (int, float))
                or not math.isfinite(float(top_value)) or float(top_value) <= 0):
            return {'status': 'conflict', 'reason': 'front wall opaque height is not supported by a medium/high reading',
                    'matches': matches, 'imageId': same_image[0], 'frontProfileId': front.get('id'),
                    'farProfileId': far.get('id')}
        top = float(top_value)
        if not (f0 <= u <= f1 and front_gz <= hit[2] <= front_gz + top):
            return {'status': 'conflict', 'reason': 'front wall does not cover a matched far-window ray', 'matches': matches, 'imageId': same_image[0]}
        if wall_opening_at(front, hit[:2], hit[2], front_gz, camera_center=center,
                           target=far_center, plane_v=owner_plane):
            visible.append({'match': m, 'reason': 'source ray remains open through a portal or recess',
                            'frontHitWorld': hit.tolist(), 'ownerPlaneV': float(owner_plane)})
            continue
        blocked.append({'match': m, 'frontHitWorld': hit.tolist(), 'farCenterWorld': far_center.tolist(),
                        'frontLocalXM': x, 'frontLocalZM': float(hit[2] - front_gz),
                        'frontAperture': 'window-backed', 'ownerPlaneV': float(owner_plane)})
    if len(blocked) < min_matches:
        return {'status': 'setback-visible' if visible else 'conflict',
                'reason': 'fewer than two matched far-window rays are blocked by the front wall',
                'imageId': same_image[0], 'frontProfileId': front.get('id'), 'farProfileId': far.get('id'),
                'matches': matches, 'blockedMatches': blocked, 'visibleMatches': visible}
    return {'status': 'occluded-duplicate', 'imageId': same_image[0], 'frontProfileId': front.get('id'),
            'farProfileId': far.get('id'), 'frontPlaneV': float(np.median(owner_planes)), 'farPlaneV': float(far['planeV']),
            'matches': matches, 'blockedMatches': blocked, 'visibleMatches': visible,
            'blockedRays': [q for q in blocked], 'frontDepthM': float(np.mean(dist_a if a_front else dist_b)),
            'farDepthM': float(np.mean(dist_b if a_front else dist_a))}


def suppress_refs(profile, refs):
    """Copy a profile with only selected window cells/openings omitted."""
    refs = {tuple(r) for r in refs}
    if not refs:
        return profile, []
    out = dict(profile)
    groups = [dict(g) for g in profile.get('groups') or []]
    openings = list(profile.get('openings') or [])
    suppressed = []
    for ref in refs:
        if not ref:
            continue
        if ref[0] == 'group' and len(ref) == 4:
            _, gi, ix, iy = ref
            if 0 <= gi < len(groups):
                group = dict(groups[gi]); omit = [list(v) for v in group.get('omit') or []]
                if [ix, iy] not in omit:
                    omit.append([ix, iy]); group['omit'] = omit; groups[gi] = group; suppressed.append(ref)
        elif ref[0] == 'opening' and len(ref) == 2:
            oi = ref[1]
            if 0 <= oi < len(openings):
                openings[oi] = None; suppressed.append(ref)
    out['groups'] = groups
    out['openings'] = [q for q in openings if q is not None]
    return out, suppressed


def _profile_side_id(profile):
    value = str(profile.get('id') or '')
    return value.removeprefix('regen-')


def _profile_source_images(profile):
    """Exact panoramas used to read this profile, not every side-view panorama."""
    images = set()
    for source in profile.get('sourceIds') or []:
        value = str(source)
        images.add(value.rsplit(':', 1)[-1])
    provenance = profile.get('readingProvenance') or {}
    if provenance.get('image'):
        images.add(str(provenance['image']))
    return images


def _records(doc):
    return doc.get('pieces', []) if isinstance(doc, dict) else doc if isinstance(doc, list) else []


@lru_cache(maxsize=4)
def _area_source_catalog(area_str):
    """Load area-local source side/camera transforms once; ambiguous sources fail closed.

    An already calibrated camera map may be passed through build context as
    `FACADE_SOURCE_CAMERAS`. This fallback uses only the area's own source index
    and registered-camera file, with no fixed CRS or pilot calibration.
    """
    area = Path(area_str)
    source_path = area / 'imagery/source-index-registered.json'
    reg_path = area / 'imagery/registered-cameras.json'
    if not all(p.is_file() for p in (source_path, reg_path)):
        return None
    src = json.loads(source_path.read_text())
    regs = json.loads(reg_path.read_text())
    cameras = {}
    for key, record in src.items():
        camera = record.get('camera') or {}
        image_id = str(key).split(':', 1)[-1]
        reg = regs.get(image_id)
        if not reg or not reg.get('accepted'):
            continue
        try:
            R = Rotation.from_rotvec(np.asarray(camera['rotationAngleAxis'], float)).as_matrix()
            tilt = np.radians(reg.get('tiltDeg') or [0., 0.])
            # Match regen.register_cameras.apply_registration without importing
            # area-specific crop code.
            from regen.register_cameras import _rz, _tilt
            R = R @ _rz(-math.radians(float(reg.get('dyawDeg') or 0.))) @ _tilt(float(tilt[0]), float(tilt[1]))
            xy = np.asarray(reg.get('xy'), float)
            z = float(reg.get('z'))
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(R).all() or not math.isfinite(z):
                continue
            cameras[image_id] = {'imageId': image_id, 'center': [float(xy[0]), float(xy[1]), z],
                                 'rotation': R.tolist(), 'registrationCostDeg': reg.get('costDeg'),
                                 'worldRegistrationValidated': bool(camera.get('worldRegistrationValidated')),
                                 'sourceSha256': record.get('imageSha256')}
        except (KeyError, TypeError, ValueError):
            continue
    side_records = {}
    dirs = [area / 'regen/sides-reg', area / 'regen/sides']
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob('sides*.json')):
            try:
                for piece in _records(json.loads(path.read_text())):
                    if isinstance(piece, dict) and piece.get('id'):
                        side_records.setdefault(piece['id'], []).append(piece)
            except (OSError, ValueError):
                continue
    # Prefer the source-side record with the most usable views. Conflicting
    # records with equal evidence are excluded rather than picked by filename.
    sides = {}
    for side_id, candidates in side_records.items():
        best_n = max(len(q.get('views') or []) for q in candidates)
        best = [q for q in candidates if len(q.get('views') or []) == best_n]
        signatures = {json.dumps({k: q.get(k) for k in ('normal', 'frameU', 'planeV', 'groundZ', 'lengthM')}, sort_keys=True) for q in best}
        if len(signatures) == 1:
            sides[side_id] = best[0]
    return {'sides': sides, 'cameras': cameras, 'area': str(area)}


def area_photo_observations(profile, catalog):
    """Build source-pano opening boxes for a profile from its recorded side/camera metadata."""
    if not catalog:
        return None
    source_images = _profile_source_images(profile)
    if not source_images:
        return None
    side = catalog['sides'].get(_profile_side_id(profile))
    if not side:
        return None
    # Reject metadata mismatches: profile wall coordinates and the camera crop
    # must share the exact registered side frame.
    if not (np.allclose(profile.get('normal'), side.get('normal'), atol=.02) and
            np.allclose(profile.get('frameU'), side.get('frameU'), atol=.05) and
            abs(float(profile.get('planeV', 1e9)) - float(side.get('planeV', -1e9))) <= .1):
        return None
    views = {str(v.get('image')) for v in side.get('views') or [] if v.get('image')}
    # A side's available camera views do not prove that a reading was made in
    # those images. Only project into a panorama that is explicitly recorded
    # as a source for this profile; multi-view evidence needs its own record.
    common = sorted(source_images & views & set(catalog['cameras']))
    if not common:
        return None
    ground_z = float(side['groundZ'])
    projections = []
    for image_id in common:
        projections.extend(project_profile_windows(profile, ground_z, catalog['cameras'][image_id]))
    return {'side': side, 'groundZ': ground_z, 'projections': projections,
            'cameras': {image_id: catalog['cameras'][image_id] for image_id in common}}


def prepare_profile_observations(profiles, ctx):
    """Return per-profile source-photo records; absent/ambiguous metadata is skipped."""
    area = ctx.get('AREA')
    if not area:
        area = os.environ.get('SFW_AREA')
    if not area:
        return {}
    try:
        catalog = _area_source_catalog(str(Path(area).resolve()))
    except (OSError, ValueError, TypeError):
        return {}
    if catalog is None:
        return {}
    supplied = ctx.get('FACADE_SOURCE_CAMERAS') or {}
    if supplied:
        catalog = {'sides': catalog['sides'], 'cameras': dict(catalog['cameras']), 'area': catalog['area']}
        for image_id, camera in supplied.items():
            if isinstance(camera, dict) and all(k in camera for k in ('center', 'rotation')):
                catalog['cameras'][str(image_id)] = {'imageId': str(image_id), 'center': camera['center'],
                                                      'rotation': camera['rotation'],
                                                      'registrationCostDeg': camera.get('registrationCostDeg'),
                                                      'worldRegistrationValidated': camera.get('worldRegistrationValidated', False),
                                                      'sourceSha256': camera.get('sourceSha256')}
    output = {}
    for profile in profiles:
        value = area_photo_observations(profile, catalog)
        if value and value['projections']:
            output[profile['id']] = value
    return output


def match_profile_pairs(profiles, observations, *, host_geometry=None, min_matches=2):
    """Find same-pano, same-facing profiles that describe the same repeated windows."""
    pairs = []
    for i, a in enumerate(profiles):
        oa = observations.get(a.get('id'))
        if not oa:
            continue
        for b in profiles[i + 1:]:
            ob = observations.get(b.get('id'))
            if not ob:
                continue
            na, nb = _unit(a.get('normal', ())), _unit(b.get('normal', ()))
            if na is None or nb is None or float(na @ nb) < .98:
                continue
            common = sorted(set(oa['cameras']) & set(ob['cameras']))
            for image_id in common:
                pa = [f for f in oa['projections'] if f['imageId'] == image_id]
                pb = [f for f in ob['projections'] if f['imageId'] == image_id]
                matches = match_projected_windows(pa, pb, min_matches=min_matches)
                if not matches:
                    continue
                result = duplicate_window_observations(a, b, pa, pb, oa['groundZ'], ob['groundZ'], oa['cameras'][image_id],
                                                       host_geometry=host_geometry, min_matches=min_matches)
                if result:
                    pairs.append({'a': a, 'b': b, 'result': result, 'imageId': image_id,
                                  'groundA': oa['groundZ'], 'groundB': ob['groundZ'], 'camera': oa['cameras'][image_id]})
    return pairs


def edge_plane(a, b, facing_normal):
    mid = (np.asarray(a, float)[:2] + np.asarray(b, float)[:2]) / 2
    n = _unit(facing_normal)
    return float(mid @ n) if n is not None else None


def suppress_pair_features(pair, selected_profile, selected_plane):
    """Omit only the occluded rear duplicate cells on their rear host plane."""
    proof = pair['result']
    if proof.get('status') != 'occluded-duplicate':
        return selected_profile, [], proof
    front_id, far_id = proof['frontProfileId'], proof['farProfileId']
    if selected_profile.get('id') not in (front_id, far_id):
        return selected_profile, [], None
    if selected_profile.get('id') == front_id:
        return selected_profile, [], proof
    # The rear profile only loses the matched windows where the current wall
    # face lies behind the front measured host plane. Other windows in its
    # reading remain intact.
    n = _unit(pair['a']['normal'])
    front_plane = float(proof['frontPlaneV'])
    if n is None or selected_plane is None:
        return selected_profile, [], {'status': 'conflict', 'reason': 'host wall plane unavailable', 'proof': proof}
    if float(selected_plane) >= front_plane - .12:
        return selected_profile, [], proof
    far_refs = []
    for blocked_match in proof.get('blockedMatches') or []:
        match = blocked_match['match']
        feature = match['a'] if selected_profile.get('id') == pair['a'].get('id') else match['b']
        far_refs.extend(feature.get('refs') or [])
    filtered, removed = suppress_refs(selected_profile, far_refs)
    if not removed:
        return selected_profile, [], {'status': 'conflict', 'reason': 'matched windows were not removable from profile schema', 'proof': proof}
    return filtered, removed, proof
