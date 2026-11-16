"""Place observed roof details against a part's measured roof support.

This module adapts review observations to neutral roof-feature input specs. It does
not build meshes or infer building-specific rules. Every accepted placement is
linked to one existing profile and checked against the requested footprint and
roof returns.
"""

from __future__ import annotations

import math


_CONFIDENT = {"high", "medium"}
# roof_features places glazing 8 cm above feature bottom. Preserve 5 mm
# clearance to that sill rather than rejecting a harmless plinth overlap.
_MAX_SUPPORT_ABOVE_BOTTOM = .08 - .005


def _same(a, b):
    return a is not None and b is not None and str(a) == str(b)


def _profile_side(profile):
    value = str(profile.get("id", ""))
    return value.removeprefix("regen-")


def _profiles_for_side(profiles, side, part_id, source_id):
    """Join a reading to its profile, preferring exact part identity."""
    candidates = [p for p in profiles if _profile_side(p) == str(side)]
    exact, fallback = [], []
    for p in candidates:
        match = p.get("match") or {}
        matched_part = match.get("partId")
        matched_feature = match.get("featureId")
        if _same(matched_part, part_id):
            exact.append(p)
        elif matched_part is None and (_same(matched_feature, source_id) or not match):
            fallback.append(p)
    return exact if exact else fallback


def _xy(profile, u, v):
    import numpy as np

    n = np.asarray(profile.get("normal"), dtype=float)[:2]
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        raise ValueError("profile normal is missing or zero")
    n = n / norm
    tangent = np.array([-n[1], n[0]])
    return n * float(v) + tangent * float(u), n, tangent


def _bounds(feature):
    vals = [feature.get(k) for k in ("x0M", "x1M", "bottomM", "topM")]
    if any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in vals):
        return None
    x0, x1, bottom, top = map(float, vals)
    if x1 <= x0 or top <= bottom:
        return None
    return x0, x1, bottom, top


def _sample_support(anchor, n, tangent, width, target_z, footprint, roof_samples,
                    maximum_support_z=None, photo_cap_z=None, body_depth=None):
    """Intersect the observed wall-frame ray with the fitted roof surface.

    ``roof_samples`` are vertices of the actual fitted shell, not the raw scan.
    Candidate points stay on the dormer's lateral centerline; a small local fit
    estimates shell height between vertices without shifting that centerline.
    """
    import numpy as np

    # A fitted trimesh is authoritative: interpolate its actual triangles at
    # exact ray locations. Roof meshes can be intentionally coarse, so nearest
    # vertices alone are not sufficient evidence of the surface height.
    mesh_vertices = getattr(roof_samples, "vertices", None)
    mesh_faces = getattr(roof_samples, "faces", None)
    if mesh_vertices is not None and mesh_faces is not None:
        return _mesh_ray_support(anchor, n, width, target_z, footprint,
                                 np.asarray(mesh_vertices, dtype=float),
                                 np.asarray(mesh_faces, dtype=int), maximum_support_z, photo_cap_z, body_depth)

    ceiling = target_z + _MAX_SUPPORT_ABOVE_BOTTOM if maximum_support_z is None else maximum_support_z

    if roof_samples is None:
        return None
    samples = np.asarray(roof_samples, dtype=float)
    if samples.size == 0:
        return None
    samples = samples.reshape((-1, samples.shape[-1]))
    if samples.shape[1] < 3:
        return None
    # Work in a narrow strip around the ray. This preserves the source-frame
    # lateral location while allowing interpolation across the shell tessellation.
    delta = samples[:, :2] - anchor[None, :]
    lateral = np.abs(delta @ tangent)
    half_width = max(.12, width * .15)
    strip = samples[lateral <= half_width]
    if not len(strip):
        return None

    Point = __import__("shapely.geometry", fromlist=["Point"]).Point
    fallback = []
    preferred = []
    # Facade registration can lie up to 1.5 m inside the mapped eave.
    # Search that bounded outward interval too, but only on the actual roof.
    for inset in _support_insets(anchor, n, footprint, width if photo_cap_z is not None else 0.):
        candidate = anchor - n * float(inset)
        if footprint is not None and not footprint.covers(Point(float(candidate[0]), float(candidate[1]))):
            continue
        if body_depth is not None and not _body_on_roof(candidate, n, width, body_depth, footprint):
            continue
        local_delta = strip[:, :2] - candidate[None, :]
        along = np.abs(local_delta @ n)
        local_lateral = np.abs(local_delta @ tangent)
        nearby = (along <= .65) & (local_lateral <= half_width)
        if not nearby.any():
            continue
        local = strip[nearby]
        offsets = local[:, :2] - candidate[None, :]
        distances = np.linalg.norm(offsets, axis=1)
        # Weighted local plane fit gives a stable surface height for a sloped
        # roof, with a nearest-vertex fallback for sparse or collinear samples.
        if len(local) >= 3 and np.linalg.matrix_rank(np.column_stack([offsets, np.ones(len(offsets))])) >= 3:
            weights = 1. / np.maximum(distances, .08)
            design = np.column_stack([offsets, np.ones(len(offsets))])
            coef, *_ = np.linalg.lstsq(design * weights[:, None], local[:, 2] * weights, rcond=None)
            surface_z = float(coef[2])
        else:
            idx = int(np.argmin(distances))
            if distances[idx] > .65:
                continue
            surface_z = float(local[idx, 2])
        error = abs(surface_z - target_z)
        record = (error, float(inset), candidate, surface_z, float(np.min(distances)))
        if photo_cap_z is not None:
            if target_z - .05 <= surface_z < photo_cap_z - .05:
                preferred.append(record)
            continue
        if surface_z <= ceiling and error <= .25:
            preferred.append(record)
        elif surface_z <= ceiling and error <= .5:
            fallback.append(record)

    # Prefer the closest support to the wall among good intersections, then the
    # smallest vertical mismatch. The bounded fallback is deliberately weaker.
    options = preferred or fallback
    if not options:
        return None
    key = (lambda r: (abs(r[1]), r[4], r[0])) if photo_cap_z is not None else (lambda r: (round(r[0], 3), r[4], r[1]))
    error, inset, candidate, surface_z, _ = min(options, key=key)
    return candidate, inset, surface_z, error


def _mesh_ray_support(anchor, n, width, target_z, footprint, vertices, faces,
                      maximum_support_z=None, photo_cap_z=None, body_depth=None):
    """Sample a triangle mesh's height directly along an inward wall ray."""
    import numpy as np

    if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] < 3:
        return None
    Point = __import__("shapely.geometry", fromlist=["Point"]).Point
    # Project each face to XY and cache its barycentric denominator. Vertical or
    # edge-on faces have zero projected area and cannot define roof height z(x,y).
    triangles = vertices[faces[:, :3], :3]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    den = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
           + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))
    valid = np.abs(den) > 1e-10
    triangles, a, b, c, den = triangles[valid], a[valid], b[valid], c[valid], den[valid]
    if not len(triangles):
        return None

    preferred, fallback = [], []
    ceiling = target_z + _MAX_SUPPORT_ABOVE_BOTTOM if maximum_support_z is None else maximum_support_z
    # Facade registration can lie up to 1.5 m inside the mapped eave.
    # Search that bounded outward interval too, but only on the actual roof.
    for inset in _support_insets(anchor, n, footprint, width if photo_cap_z is not None else 0.):
        candidate = anchor - n * float(inset)
        if footprint is not None and not footprint.covers(Point(float(candidate[0]), float(candidate[1]))):
            continue
        if body_depth is not None and not _body_on_roof(candidate, n, width, body_depth, footprint):
            continue
        x, y = float(candidate[0]), float(candidate[1])
        w0 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0])
              + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / den
        w1 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0])
              + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / den
        w2 = 1. - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        if not inside.any():
            continue
        # If the mesh has overlapping projected faces, use the upper supported
        # surface that remains at or below the observed sill.
        z_values = (w0[inside] * triangles[inside, 0, 2]
                    + w1[inside] * triangles[inside, 1, 2]
                    + w2[inside] * triangles[inside, 2, 2])
        below = (z_values[(z_values >= target_z - .05) & (z_values < photo_cap_z - .05)]
                 if photo_cap_z is not None else z_values[z_values <= ceiling])
        if not len(below):
            continue
        surface_z = float(np.max(below))
        error = abs(surface_z - target_z)
        record = (error, float(inset), candidate, surface_z)
        if photo_cap_z is not None:
            preferred.append(record)
            continue
        if error <= .25:
            preferred.append(record)
        elif error <= .5:
            fallback.append(record)
    options = preferred or fallback
    if not options:
        return None
    key = (lambda r: (abs(r[1]), r[0])) if photo_cap_z is not None else (lambda r: (round(r[0], 3), r[1]))
    error, inset, candidate, surface_z = min(options, key=key)
    return candidate, inset, surface_z, error


def _body_on_roof(front, normal, width, depth, footprint):
    if footprint is None:
        return True
    import numpy as np
    from shapely.geometry import Polygon
    tangent = np.array([-normal[1], normal[0]])
    body = Polygon([front - tangent * width / 2., front + tangent * width / 2.,
                    front + tangent * width / 2. - normal * depth,
                    front - tangent * width / 2. - normal * depth])
    return footprint.buffer(1e-6).covers(body)


def _support_insets(anchor, normal, footprint, width=0.):
    import numpy as np
    from shapely.geometry import LineString
    values = list(np.arange(-1.5, 6.0001, .1))
    if footprint is not None:
        tangent = np.array([-normal[1], normal[0]])
        for lateral in (0., -width / 2., width / 2.):
            origin = anchor + tangent * lateral
            hits = footprint.boundary.intersection(LineString([origin + normal * 1.5, origin - normal * 6.]))
            for hit in getattr(hits, 'geoms', [hits]):
                if hit.geom_type == 'Point':
                    values.append(float((origin - np.asarray(hit.coords[0])) @ normal) + 1e-6)
    return sorted(set(round(float(v), 8) for v in values))


def _has_source_front(feature):
    evidence = feature.get('frontEvidence')
    if not isinstance(evidence, dict) or evidence.get('coordinateSpace') != 'normalized-front':
        return False
    source = evidence.get('provenance') or {}
    return (source.get('status') in ('accepted-approximate', 'partial-approximate')
            and all(isinstance(source.get(k), str) and len(source[k]) == 64
                    for k in ('sourceSha256', 'metadataSha256')))


def _aperture_support_ceiling(feature, bottom_z, top_z):
    """Keep the host below the actual source aperture, not a generic 8 cm sill.

    A front's opaque plinth can meet the host roof. Its explicitly measured
    opening cannot be covered. Invalid/missing evidence retains the legacy
    clearance rather than granting arbitrary overlap.
    """
    import numpy as np
    from shapely.geometry import Polygon, box

    evidence = feature.get('frontEvidence')
    fallback = bottom_z + _MAX_SUPPORT_ABOVE_BOTTOM
    if not isinstance(evidence, dict) or evidence.get('coordinateSpace') != 'normalized-front':
        return fallback, 'legacy-sill-clearance'
    items = evidence.get('apertures', evidence.get('windows'))
    if not isinstance(items, list) or not items:
        return fallback, 'legacy-sill-clearance'
    lows = []
    width = float(np.clip(feature.get('x1M', 0) - feature.get('x0M', 0), .8, 8.))
    for item in items:
        if not isinstance(item, dict):
            return fallback, 'legacy-sill-clearance'
        try:
            if 'outline01' in item:
                points = np.asarray(item['outline01'], dtype=float)
                if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                        or not np.isfinite(points).all()
                        or (points < 0).any() or (points > 1).any()):
                    return fallback, 'legacy-sill-clearance'
                if not Polygon(points).is_valid:
                    return fallback, 'legacy-sill-clearance'
                opening = Polygon(np.c_[(points[:, 0] - .5) * width,
                                         bottom_z + points[:, 1] * (top_z - bottom_z)])
            else:
                rect = item['rect01']
                if len(rect) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in rect):
                    return fallback, 'legacy-sill-clearance'
                if not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
                    return fallback, 'legacy-sill-clearance'
                opening = box((rect[0] - .5) * width, bottom_z + rect[1] * (top_z - bottom_z),
                              (rect[2] - .5) * width, bottom_z + rect[3] * (top_z - bottom_z))
            a, b, c, d = opening.bounds
            frame = float(item.get('frameWidthM', min(.055, (c - a) * .15, (d - b) * .15)))
            if not math.isfinite(frame) or frame <= 0:
                return fallback, 'legacy-sill-clearance'
            infill = opening.buffer(-frame, join_style=2)
            if infill.is_empty or not infill.is_valid:
                return fallback, 'legacy-sill-clearance'
            lows.append(float(infill.bounds[1]))
        except (ValueError, TypeError, KeyError):
            return fallback, 'legacy-sill-clearance'
    return min(lows) - .005, 'source-aperture-clearance'


def _bay_records(profile):
    bays = profile.get("bays") or []
    if isinstance(bays, dict):
        bays = [bays]
    f = profile.get("frameU") or [None, None]
    try:
        f0, f1 = map(float, f)
    except (TypeError, ValueError):
        f0 = f1 = None
    out = []
    for bay in bays:
        if not isinstance(bay, dict):
            continue
        x0, x1 = bay.get("x0"), bay.get("x1")
        if x0 is None or x1 is None:
            rect = bay.get("rect")
            if rect and len(rect) >= 3 and f0 is not None:
                x0, x1 = f0 + float(rect[0]) * (f1 - f0), f0 + float(rect[2]) * (f1 - f0)
        if x0 is not None and x1 is not None:
            out.append((float(x0), float(x1), float(bay.get("projectionM") or 0.)))
    return out


def resolve(observations, profiles, part_id, source_id, footprint, z0, roof_samples):
    """Return ``(feature_specs, audit)`` for supported observed dormers/caps.

    Observation x coordinates are metres in the source profile's own frame. Rejected
    features are recorded in audit, so callers can distinguish missing evidence from
    a valid empty result.
    """
    import numpy as np

    features, audit = [], {"status": "resolved", "accepted": [], "rejected": []}
    for obs in observations or []:
        side = obs.get("side")
        if not side:
            continue
        joined = _profiles_for_side(profiles or [], side, part_id, source_id)
        if not joined:
            continue
        # Exact part identity is authoritative. If duplicate exact records exist,
        # retain each evidence profile but never pull a sibling part by feature ID.
        for profile in joined:
            profile_id = profile.get("id")
            reading_confidence = (profile.get("reading") or {}).get("confidence")
            n_raw = profile.get("normal")
            frame = profile.get("frameU")
            plane_v = profile.get("planeV")
            if n_raw is None or not frame or plane_v is None:
                audit["rejected"].append({"side": side, "profileId": profile_id, "reason": "incomplete-profile-frame"})
                continue
            try:
                frame0, frame1 = map(float, frame)
                plane_v = float(plane_v)
                _, n, tangent = _xy(profile, 0., plane_v)
            except (TypeError, ValueError):
                audit["rejected"].append({"side": side, "profileId": profile_id, "reason": "invalid-profile-frame"})
                continue
            for item in obs.get("features") or []:
                conf = item.get("confidence")
                if conf not in _CONFIDENT:
                    audit["rejected"].append({"side": side, "kind": item.get("kind"), "reason": "low-feature-confidence"})
                    continue
                bounds = _bounds(item)
                if bounds is None:
                    audit["rejected"].append({"side": side, "kind": item.get("kind"), "reason": "missing-or-invalid-bounds"})
                    continue
                x0, x1, bottom, top = bounds
                width = x1 - x0
                frame_width = frame1 - frame0
                if x0 < -.05 or x1 > frame_width + .05:
                    audit["rejected"].append({"side": side, "kind": item.get("kind"), "reason": "outside-profile-frame"})
                    continue

                kind = item.get("kind")
                if kind == "dormer":
                    anchor, _, _ = _xy(profile, frame0 + (x0 + x1) / 2., plane_v)
                    support_ceiling, clearance_status = _aperture_support_ceiling(
                        item, float(z0) + bottom, float(z0) + top)
                    source_front = _has_source_front(item)
                    depth = min(3., max(1., width * .9))
                    source_cap = top - (.15 if item.get('roof') == 'shed' else min(.6, width * .3))
                    if (isinstance(item.get('roofEaveM'), (int, float))
                            and isinstance(item.get('roofEaveEvidence'), dict)
                            and item['roofEaveEvidence'].get('status') == 'source-derived'
                            and bottom + .05 < item['roofEaveM'] < top):
                        source_cap = item['roofEaveM']
                    support = _sample_support(anchor, n, tangent, width, float(z0) + bottom,
                                              footprint, roof_samples, support_ceiling,
                                              float(z0) + source_cap if source_front else None,
                                              depth if source_front else None)
                    if support is None:
                        audit["rejected"].append({"side": side, "kind": kind, "bounds": [x0, x1, bottom, top],
                                                  "reason": "no-inset-roof-support"})
                        continue
                    front, inset, support_z, support_error = support
                    # Use the sample-supported inward point to set the front anchor;
                    # the center is then one half-depth farther into the roof.
                    center = front - n * (depth / 2.)
                    if source_front and footprint is not None:
                        from shapely.geometry import Polygon
                        body = Polygon([front - tangent * width / 2., front + tangent * width / 2.,
                                        front + tangent * width / 2. - n * depth,
                                        front - tangent * width / 2. - n * depth])
                        if not footprint.buffer(1e-6).covers(body):
                            audit['rejected'].append({'side': side, 'kind': kind,
                                'reason': 'source-component-body-crosses-roof-domain'})
                            continue
                    roof_kind = item.get("roof")
                    cap = .15 if roof_kind == "shed" else min(.6, width * .3)
                    eave, eave_status = top - cap, "inferred-width-prior"
                    observed_eave = item.get("roofEaveM")
                    eave_evidence = item.get("roofEaveEvidence")
                    if observed_eave is not None:
                        if (isinstance(observed_eave, (int, float)) and not isinstance(observed_eave, bool)
                                and np.isfinite(observed_eave) and bottom + .05 < observed_eave < top
                                and isinstance(eave_evidence, dict)
                                and eave_evidence.get("status") == "source-derived"):
                            eave, eave_status = float(observed_eave), "source-derived-uncertain-prior"
                        else:
                            audit["rejected"].append({"side": side, "kind": kind,
                                "reason": "invalid-or-unattributed-component-eave", "fieldOnly": True})
                    spec = {"kind": "dormer", "xy": [float(center[0]), float(center[1])],
                            "facing": [float(n[0]), float(n[1])], "widthM": width, "depthM": depth,
                            "eaveM": eave, "ridgeM": top, "roof": roof_kind,
                            "windowCount": int(item.get("windowCount") or 0), "bottomM": bottom,
                            "wallMaterial": item.get("wallColor"), "trimMaterial": item.get("trimColor"),
                            "roofMaterial": item.get("roofColor")}
                    front_evidence = item.get("frontEvidence")
                    if isinstance(front_evidence, dict):
                        spec["frontEvidence"] = front_evidence
                    if source_front:
                        spec['composeWithHost'] = True
                elif kind == "conical-cap":
                    mid = frame0 + (x0 + x1) / 2.
                    bay_matches = [b for b in _bay_records(profile)
                                   if abs((b[0] + b[1]) / 2. - mid) <= max(.5, width * .25)
                                   and abs((b[1] - b[0]) - width) <= max(.75, width * .35)]
                    if not bay_matches:
                        audit["rejected"].append({"side": side, "kind": kind, "bounds": [x0, x1, bottom, top],
                                                  "reason": "no-matching-profile-bay"})
                        continue
                    bx0, bx1, projection = min(bay_matches, key=lambda b: abs((b[0] + b[1]) / 2. - mid))
                    center, _, _ = _xy(profile, (bx0 + bx1) / 2., plane_v + projection / 2.)
                    spec = {"kind": "conical-cap", "xy": [float(center[0]), float(center[1])],
                            "facing": [float(n[0]), float(n[1])], "widthM": width,
                            "depthM": max(projection * 2., 1.), "eaveM": bottom, "ridgeM": top,
                            "roof": "cone", "windowCount": 0,
                            "wallMaterial": item.get("wallColor"), "trimMaterial": item.get("trimColor"),
                            "roofMaterial": item.get("roofColor")}
                else:
                    audit["rejected"].append({"side": side, "kind": kind, "reason": "unsupported-feature-kind"})
                    continue
                features.append(spec)
                audit["accepted"].append({"side": side, "profileId": profile_id,
                                          "featureKind": kind, "confidence": conf,
                                          "profileConfidence": reading_confidence,
                                          "placement": "sample-supported" if kind == "dormer" else "profile-bay-anchored",
                                          "insetM": round(inset, 3) if kind == "dormer" else None,
                                          "frontSurfaceZM": round(support_z, 3) if kind == "dormer" else None,
                                          "frontSurfaceErrorM": round(support_error, 3) if kind == "dormer" else None,
                                          "supportClearance": clearance_status if kind == "dormer" else None,
                                          "supportPolicy": ('source-front-body-intersection' if source_front
                                                            else 'legacy-sill-intersection') if kind == 'dormer' else None,
                                          "maximumSupportZM": round(support_ceiling, 3) if kind == "dormer" else None,
                                          "depthStatus": "inferred-width-prior" if kind == "dormer" else None,
                                          "eaveStatus": eave_status if kind == "dormer" else None,
                                          "frontGeometry": "source-described" if kind == "dormer" and isinstance(item.get("frontEvidence"), dict) else None})
    if audit["rejected"] and not features:
        audit["status"] = "unresolved"
    elif audit["rejected"]:
        audit["status"] = "partial"
    elif not features:
        audit["status"] = "no-observations"
    return features, audit
