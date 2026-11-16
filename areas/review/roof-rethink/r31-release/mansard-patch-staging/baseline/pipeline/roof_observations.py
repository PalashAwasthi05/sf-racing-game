"""Turn confident, building-scale facade roof readings into roof_forms.build specs.

This module deliberately has no place or style rules. A roof mentioned on a facade is
local evidence unless its reading explicitly says it spans the full frontage.
"""

from copy import deepcopy


_ROOF_KINDS = {"gable", "hip", "mansard", "shed", "pyramid"}
_CONFIDENCE = {"medium", "high"}


def measured_frontage(roofline):
    """Derive a boundary from reviewed source pixels, never from a freehand metric claim.

    A reader may recognize a gable while lacking reliable metric registration.
    Keep that qualitative evidence separate; missing or conflicting pixel
    measurements cannot become hard boundary heights.
    """
    from .roof_image_measurements import derive_gable_frontage
    if not isinstance(roofline, dict) or not isinstance(roofline.get('measurement'), dict):
        return None, {'status': 'unverified-source', 'reason': 'missing pixel measurement provenance'}
    measurement = deepcopy(roofline['measurement'])
    supplied = roofline.get('pointsM')
    if supplied:
        for view in measurement.get('views', []):
            view['pointsM'] = [{'xM': q[0], 'zM': q[1]} for q in supplied]
    audit = derive_gable_frontage(measurement)
    line = audit.get('acceptedLine')
    return ([[q['xM'], q['zM']] for q in line] if line else None), audit


def _same(a, b):
    return a is not None and b is not None and str(a) == str(b)


def _full_width(roof):
    # Explicit local labels win even when a broad span was accidentally measured.
    if roof.get("scope") in ("local", "facade", "dormer", "decorative") or roof.get("kind") == "dormer":
        return False
    if roof.get("fullWidth") is True or roof.get("scope") in ("building", "full-width", "fullWidth"):
        return True
    x = roof.get("x")
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            if abs(float(x[1]) - float(x[0])) >= .8:
                return True
        except (TypeError, ValueError):
            pass
    # Width ratios are useful when a reader reports a measured frontage span.
    for key in ("widthShare", "frontageShare", "coverage"):
        value = roof.get(key)
        if isinstance(value, (int, float)) and value >= .8:
            return True
    return False


def proposal(profiles, part_id, source_id):
    """Return ``(spec_or_none, audit)`` for confident global roof observations.

    Profiles join only by exact ``match.partId`` or ``match.featureId`` identity.
    A gable is retained when explicitly outlined or measured across the full width;
    conflicting global roof kinds or dimensions remain unresolved.
    """
    relevant = []
    for profile in profiles:
        match = profile.get("match") or {}
        # Part identity is authoritative when present. Feature identity is a fallback
        # for profiles without a part identifier, never a way to pull sibling parts in.
        if (_same(match.get("partId"), part_id) if match.get("partId") is not None
                else _same(match.get("featureId"), source_id)):
            relevant.append(profile)

    evidence, excluded = [], []
    for profile in relevant:
        match = profile.get("match") or {}
        identity = {"partId": match.get("partId"), "featureId": match.get("featureId")}
        reading = profile.get("reading") or {}
        confidence = reading.get("confidence")
        if confidence not in _CONFIDENCE:
            excluded.append({**identity, "reason": "low-confidence"})
            continue
        outlined_gable = bool(profile.get("gableOutline"))
        for roof in profile.get("roofs") or []:
            kind = roof.get("kind")
            if kind not in _ROOF_KINDS:
                continue
            if not _full_width(roof) and not (kind == "gable" and outlined_gable):
                excluded.append({**identity, "kind": kind, "reason": "facade-local-or-partial"})
                continue
            values = {k: deepcopy(roof[k]) for k in ("pitchDeg", "riseM", "overhangM") if isinstance(roof.get(k), (int, float))}
            height = profile.get("heightRefM")
            if isinstance(height, (int, float)) and isinstance(roof.get("z"), (int, float)):
                values["wallTopM"] = height * roof["z"]
            evidence.append({"kind": kind, **values, "confidence": confidence, **identity})

    kinds = {e["kind"] for e in evidence}
    dimensions = {key: [e[key] for e in evidence if key in e] for key in ("pitchDeg", "riseM", "overhangM", "wallTopM")}
    tolerances = {"pitchDeg": 2., "riseM": .2, "overhangM": .1, "wallTopM": .5}
    conflicts = []
    if len(kinds) > 1:
        conflicts.append("conflicting-kinds")
    for key, values in dimensions.items():
        if values and max(values) - min(values) > tolerances[key]:
            conflicts.append(f"conflicting-{key}")
    audit = {"status": "unresolved" if conflicts else ("proposed" if evidence else "no-global-roof"),
             "provenance": evidence, "excluded": excluded, "conflicts": conflicts}
    if conflicts or not evidence:
        return None, audit
    spec = {"kind": next(iter(kinds))}
    for key, vals in dimensions.items():
        if vals:
            spec[key] = vals[0]
    return spec, audit


def validate(footprint, points, spec, ground_z):
    """Check an independently observed shape against spatial roof samples.

    No dimensions are optimized against these samples. A rejected reading remains
    evidence to revisit, rather than becoming a different invented roof type.
    """
    import numpy as np
    from general_world.roof_planes import RoofPolicy, sample_cells, evaluate
    from .roof_forms import build
    wall = spec.get('wallTopM')
    if wall is None or wall < 2.5 or spec.get('riseM', 0) <= 0:
        return None, {'status': 'incomplete-dimensions'}
    patches, report = build(footprint, spec, ground_z + wall, wall)
    samples, coverage = sample_cells(footprint, points, RoofPolicy(cell_m=.75))
    if not patches or len(samples) < 30 or coverage < .6:
        return None, {'status': 'insufficient-spatial-support', 'cells': len(samples), 'coverage': coverage}
    errors = np.abs(evaluate(patches[0]['model'], samples[:, :2]) - samples[:, 2])
    median, fraction = float(np.median(errors)), float(np.mean(errors <= .3))
    accepted = median <= .3 and fraction >= .8
    return (patches if accepted else None), {'status': 'supported' if accepted else 'contradicted',
        'cells': len(samples), 'coverage': coverage, 'medianErrorM': median,
        'within30cm': fraction, 'proposal': spec, 'form': report}
