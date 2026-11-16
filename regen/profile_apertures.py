"""Stable aperture inventory for source-measured and repeated facade openings."""
from __future__ import annotations

import copy
import math
import re


def audit_profile(profile, *, source_hash_verified=False, fits=None):
    """Return a profile with explicit openings first and safe measured groups after.

    Group rows are computed by the same facade grammar used for rendering, in
    the profile's true metric frame. Only source-hashed, measured-center window
    grids qualify. Ribbon banks, fitted/unsupported banks, and non-window
    groups are skipped. Metadata marks group entries as preserve-existing-only;
    callers must prove that native receiver geometry already exists.
    """
    from pipeline.facade_grammar import openings as compile_openings, ribbon_bank

    out = copy.deepcopy(profile)
    explicit = copy.deepcopy(profile.get('openings') or [])
    out['openings'] = explicit
    explicit_count = len(explicit)
    provenance = profile.get('readingProvenance') or {}
    source = provenance.get('sourceReading') or {}
    hash_looks_valid = bool(re.fullmatch(r'[0-9a-fA-F]{64}', str(source.get('sha256', ''))))
    source_gate = (source_hash_verified and provenance.get('status') == 'ok' and
                   provenance.get('accepted') is True and
                   provenance.get('registeredCropSource') is True and hash_looks_valid)
    frame = profile.get('frameU') or []
    height = profile.get('heightRefM')
    if len(frame) == 2 and isinstance(height, (int, float)) and math.isfinite(float(height)):
        width = abs(float(frame[1]) - float(frame[0])); height = float(height)
    else:
        width = 0.; height = 0.
    fits = [] if fits is None else fits
    group_indices = {}
    skipped = []
    if source_gate and width > 0 and height > 0:
        for group_index, group in enumerate(profile.get('groups') or []):
            if not isinstance(group, dict):
                skipped.append({'groupIndex': group_index, 'reason': 'malformed-group'})
                continue
            style = group.get('style') or {}
            columns, rows = group.get('columns'), group.get('rows')
            region = group.get('region')
            centers_ok = (
                isinstance(columns, int) and not isinstance(columns, bool) and columns > 0 and
                isinstance(rows, int) and not isinstance(rows, bool) and rows > 0 and
                isinstance(group.get('xCenters'), list) and len(group['xCenters']) == columns and
                isinstance(group.get('yCenters'), list) and len(group['yCenters']) == rows and
                all(isinstance(v, (int, float)) and math.isfinite(float(v)) and 0 <= v <= 1
                    for v in group['xCenters'] + group['yCenters']) and
                all(a < b for a, b in zip(group['xCenters'], group['xCenters'][1:])) and
                all(a < b for a, b in zip(group['yCenters'], group['yCenters'][1:]))
            )
            if group.get('fitId'):
                skipped.append({'groupIndex': group_index, 'reason': 'fitted-bank-not-supported'})
                continue
            if style.get('kind') != 'window':
                skipped.append({'groupIndex': group_index, 'reason': 'group-is-not-window'})
                continue
            if style.get('ribbon'):
                skipped.append({'groupIndex': group_index, 'reason': 'ribbon-bank-not-supported'})
                continue
            if not centers_ok:
                skipped.append({'groupIndex': group_index, 'reason': 'measured-cell-centers-required'})
                continue
            if (not isinstance(region, (list, tuple)) or len(region) != 4 or
                    not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in region) or
                    region[0] < 0 or region[0] >= region[2] or region[2] > 1 or
                    region[1] >= region[3]):
                skipped.append({'groupIndex': group_index, 'reason': 'invalid-group-region'})
                continue
            if ribbon_bank(group, width, region,
                           float(group.get('spacingX', 1.0 / columns)), columns):
                skipped.append({'groupIndex': group_index, 'reason': 'ribbon-bank-not-supported'})
                continue
            temp = {**profile, 'groups': [group], 'openings': []}
            try:
                generated = compile_openings(temp, fits, width, height, height)
            except (KeyError, TypeError, ValueError, OverflowError):
                skipped.append({'groupIndex': group_index, 'reason': 'group-opening-compile-failed'})
                continue
            # The grammar can collapse a bank to a long ribbon based on aspect
            # ratio even when the raw style does not request one. Abstain there.
            if not generated or any((op.get('style') or {}).get('ribbon') or op.get('lights', 1) > 1
                                    for op in generated):
                skipped.append({'groupIndex': group_index, 'reason': 'compiled-ribbon-or-empty-bank'})
                continue
            generated = [op for op in generated if op.get('group') == 0]
            first = len(out['openings'])
            for op in generated:
                op = {**op, '_auditGroupIndex': group_index,
                      '_auditCell': [int(op['ix']), int(op['iy'])],
                      '_receiverPolicy': 'preserve-existing-only'}
                out['openings'].append(op)
            group_indices[group_index] = list(range(first, len(out['openings'])))
    elif profile.get('groups'):
        skipped.extend({'groupIndex': i, 'reason': 'raw-source-hash-not-verified'}
                       for i, _ in enumerate(profile.get('groups') or []))
    return out, {'explicitOpeningCount': explicit_count,
                 'groupOpeningCount': len(out['openings']) - explicit_count,
                 'groupOpeningIndices': group_indices,
                 'skippedGroups': skipped,
                 'groupReceiverPolicy': 'existing-receiver-only; no receiver geometry may be synthesized'}
