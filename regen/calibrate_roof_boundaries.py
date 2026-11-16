"""Calibrate source-bound roof boundary evidence without inventing missing X.

Verified 2D pixel points follow the exact wall homography. A saved Y bracket
with a bounded frame width follows the full-width projective envelope and stays
an uncertain prior. Current file hashes freeze current inputs; absent historical
hashes are explicitly not upgraded to historical source verification.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from regen.compile_roof_fronts import _sha, _path, _source, _mapping, _map_metric, _number
from regen.reading_provenance import load_registered_sides


def _safe_map(points, context, wm):
    points = np.asarray(points, float)
    if wm is not None:
        # A homography can have finite corners but a pole through its interior.
        xyz = np.c_[wm.mid[0] + wm.t[0] * (points[:, 0] - wm.L / 2) + wm.n[0] * .05,
                    wm.mid[1] + wm.t[1] * (points[:, 0] - wm.L / 2) + wm.n[1] * .05,
                    wm.z0 + points[:, 1]]
        denominator = ((xyz - wm.Cn) @ wm.M.T) @ np.r_[wm.n, 0.]
        if np.min(np.abs(denominator)) < 1e-6 or np.min(denominator) < 0 < np.max(denominator):
            raise ValueError('boundary envelope crosses the projective horizon')
    return _map_metric(points, context, wm)


def _pixel_points(points, context, wm):
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 1 or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points > context['size'])):
        raise ValueError('invalid boundary pixel points')
    raw = np.c_[(points[:, 0] - context['pad']) / context['ppm'],
                context['height'] - (points[:, 1] - context['pad']) / context['ppm']]
    return _safe_map(raw, context, wm)


def _point_measurements(evidence, side, calibration, base):
    measurements = evidence.get('pointMeasurements')
    if not isinstance(measurements, list) or not measurements:
        return None
    outputs, heights, margins = [], [], []
    for record in measurements:
        if record.get('verified') is not True:
            raise ValueError('boundary pixel points were not source-verified')
        context = _source(record.get('source'), side, base)
        wm, camera = _mapping(context, calibration, base)
        points = _pixel_points(record.get('pointsPx'), context, wm)
        uncertainty = record.get('pointUncertaintyPx')
        if not _number(uncertainty) or uncertainty < 0:
            raise ValueError('verified boundary points need explicit pixel uncertainty')
        raw = np.asarray(record['pointsPx'], float)
        errors = []
        for delta in ((-uncertainty, -uncertainty), (-uncertainty, uncertainty),
                      (uncertainty, -uncertainty), (uncertainty, uncertainty)):
            shifted = raw + delta
            metric = np.c_[(shifted[:, 0] - context['pad']) / context['ppm'],
                           context['height'] - (shifted[:, 1] - context['pad']) / context['ppm']]
            errors.append(np.abs(_safe_map(metric, context, wm)[:, 1] - points[:, 1]))
        margin = float(np.max(errors))
        heights.extend(points[:, 1].tolist())
        margins.append(margin)
        outputs.append({'pointsPx': record['pointsPx'], 'pointsM': points.tolist(),
                        'pointUncertaintyPx': uncertainty, 'heightReadingUncertaintyM': margin,
                        'sourcePath': str(context['path']), 'sourceSha256': context['source']['sha256'],
                        'metadataPath': str(context['metadata_path']), 'metadataSha256': _sha(context['metadata_path']),
                        'image': context['view']['image'], 'view': context['source']['view'], 'calibration': camera})
    return {'status': 'calibrated-verified-points-uncertain-prior', 'heightRangeM': [min(heights), max(heights)],
            'pixelUncertaintyM': max(margins), 'measurements': outputs, 'historicalSourceHashesVerified': True,
            'exactHardConstraint': False, 'requiresIndependentLidarSupport': True}


def _bracket_envelope(evidence, side, side_record, calibration, base):
    key = 'junctionPixelYRange' if 'junctionPixelYRange' in evidence else 'visibleRoofEdgePixelYRange'
    bracket = evidence.get(key)
    if not isinstance(bracket, list) or len(bracket) != 2 or not all(_number(v) for v in bracket):
        return None
    if not side_record or side_record.get('status') != 'ok':
        raise ValueError('registered side metadata is missing or ambiguous')
    piece = side_record['piece']
    frame = evidence.get('frameM') or {}
    if (not _number(frame.get('width')) or not _number(frame.get('height'))
            or not np.isclose(frame['width'], piece['lengthM'], atol=.02)
            or not np.isclose(frame['height'], piece['heightM'], atol=1e-4)):
        raise ValueError('saved boundary frame disagrees with exact side metadata')
    paths = evidence.get('sources')
    if not isinstance(paths, list) or not paths:
        raise ValueError('boundary bracket has no source paths')
    reports, all_heights, margins = [], [], []
    original_range = None
    for value in paths:
        path = _path(value, base)
        matches = [(i, v) for i, v in enumerate(piece.get('views', [])) if Path(v.get('rect', '')).name == path.name]
        if len(matches) != 1:
            raise ValueError('boundary source has no unique exact side/view match')
        index, view = matches[0]
        # Check the actual source path as well as its filename, avoiding a crop
        # from another side directory that happens to have the same basename.
        expected = Path(side_record['path']).parent / 'img' / view['rect']
        if path.resolve() != expected.resolve():
            raise ValueError('boundary source path differs from registered metadata directory')
        current_hash = _sha(path)
        saved_hash = (evidence.get('sourceHashes') or {}).get(value)
        if saved_hash is not None and saved_hash != current_hash:
            raise ValueError('historical boundary source hash mismatch')
        source = {'sourceKey': side + ':v' + str(index) + ':rect', 'side': side, 'view': index,
                  'imageType': 'rect', 'path': str(path), 'sha256': current_hash,
                  'metadataPath': str(side_record['path']), 'metadataImageId': view.get('image'),
                  'pixelsPerMetre': evidence.get('pixelsPerMetre'), 'rectPadPx': evidence.get('padPx'),
                  'inspected': True}  # interface check only; no new visual inspection claimed below
        context = _source(source, side, base)
        wm, camera = _mapping(context, calibration, base)
        low = context['height'] - (max(bracket) - context['pad']) / context['ppm']
        high = context['height'] - (min(bracket) - context['pad']) / context['ppm']
        original_range = [low, high]
        width = float(frame['width'])
        points = np.array([[0, low], [width, low], [width, high], [0, high]])
        mapped = _safe_map(points, context, wm)
        interval = [float(mapped[:, 1].min()), float(mapped[:, 1].max())]
        # No new manual-reading precision is invented. The existing bracket
        # carries its manual spread; half a pixel adds only raster quantization.
        pixel_uncertainty = evidence.get('pixelUncertaintyPx', .5)
        if not _number(pixel_uncertainty) or pixel_uncertainty < 0:
            raise ValueError('invalid bracket pixel uncertainty')
        expanded = points.copy()
        expanded[[0, 1], 1] -= pixel_uncertainty / context['ppm']
        expanded[[2, 3], 1] += pixel_uncertainty / context['ppm']
        uncertain = _safe_map(expanded, context, wm)
        margin = max(interval[0] - float(uncertain[:, 1].min()), float(uncertain[:, 1].max()) - interval[1])
        all_heights.extend(interval)
        margins.append(margin)
        reports.append({'sourcePath': str(path), 'currentSourceSha256': current_hash,
                        'historicalSourceHashVerified': saved_hash is not None,
                        'sourceInspection': 'existing bracket retained; no new visual source inspection',
                        'metadataPath': str(side_record['path']), 'metadataSha256': _sha(side_record['path']),
                        'image': view['image'], 'view': index, 'frameHeightM': context['height'],
                        'pixelsPerMetre': context['ppm'], 'padPx': context['pad'],
                        'pixelYRange': bracket, 'unknownXRangeM': [0., width],
                        'sourceEnvelopeCornersM': points.tolist(), 'mappedEnvelopeCornersM': mapped.tolist(),
                        'heightRangeM': interval, 'pixelQuantizationUncertaintyPx': pixel_uncertainty,
                        'pixelQuantizationUncertaintyM': margin, 'calibration': camera})
    return {'status': 'calibrated-full-width-envelope-uncertain-prior',
            'role': 'host-eave' if key == 'junctionPixelYRange' else 'visible-roof-edge-only',
            'originalHeightRangeM': original_range, 'heightRangeM': [min(all_heights), max(all_heights)],
            'pixelUncertaintyM': max(margins), 'views': reports, 'unknownX': True,
            'exactHardConstraint': False, 'requiresIndependentLidarSupport': True,
            'historicalSourceHashesVerified': all(v['historicalSourceHashVerified'] for v in reports),
            'limitation': 'Envelope transforms the saved Y bracket over the bounded full facade width; it does not locate a boundary point or validate camera position.'}


def calibrate_boundaries(document, calibration, sides, *, base=Path('.')):
    output = copy.deepcopy(document)
    rows = output if isinstance(output, list) else output.get('observations', output.get('roofObservations'))
    if not isinstance(rows, list):
        raise ValueError('missing observations list')
    audit = {'version': 'source-boundary-calibration-v1', 'records': [],
             'sourceContract': {'verifiedPoints': 'eaveEvidence/ridgeEvidence.pointMeasurements: [{verified:true, source:{inspected:true,path,sha256,metadataPath,side,view,metadataImageId,imageType:"rect"}, pointsPx:[[x,y],...], pointUncertaintyPx:number}]',
                                'yBrackets': 'saved Y bracket + exact source paths and bounded frame width => conservative full-width envelope only',
                                'priorOnly': 'Envelope midpoint and anchorUncertaintyM are uncertain priors requiring independent LiDAR support; confidence is never upgraded.',
                                'historicalHashes': 'Current-source hashes are recorded separately and never claimed as historical verification.',
                                'rejectedRecords': 'Rejected source-audit/gable pixels are not revived.'}}
    for row in rows:
        side = row.get('side')
        report = {'side': side}
        history = row.setdefault('hostBoundaryCalibration', {})
        for field, evidence_field, label in (('eaveM', 'eaveEvidence', 'eave'), ('hostRidgeM', 'ridgeEvidence', 'ridge')):
            old = row.get(field)
            evidence = row.get(evidence_field) or {}
            original = {'valueM': old, 'confidence': row.get(label + 'Confidence'), 'evidence': copy.deepcopy(evidence)}
            try:
                if str(evidence.get('status', '')).startswith('rejected'):
                    raise ValueError('existing source audit rejected this boundary; no rejected pixels revived')
                transformed = _point_measurements(evidence, side, calibration, Path(base))
                if transformed is None and label == 'eave':
                    transformed = _bracket_envelope(evidence, side, sides.get(side), calibration, Path(base))
                if transformed is None:
                    raise ValueError('no source-verified 2D boundary points or bounded pixel-Y bracket')
                low, high = transformed['heightRangeM']
                transformed['original'] = original
                history[label] = transformed
                updated_evidence = copy.deepcopy(evidence)
                updated_evidence['boundaryCalibration'] = transformed
                updated_evidence['status'] = transformed['status']
                if transformed.get('role') != 'visible-roof-edge-only':
                    row[field] = float((low + high) / 2)
                    existing = row.get('anchorUncertaintyM', 0.)
                    existing = float(existing) if _number(existing) and existing >= 0 else 0.
                    row['anchorUncertaintyM'] = max(existing, (high - low) / 2 + transformed['pixelUncertaintyM'])
                    updated_evidence['junctionHeightRangeM' if label == 'eave' else 'ridgeHeightRangeM'] = [low, high]
                else:
                    updated_evidence['visibleRoofEdgeHeightRangeM'] = [low, high]
                row[evidence_field] = updated_evidence
                report[label] = {'status': transformed['status'], 'originalM': old, 'activeM': row.get(field),
                                 'heightRangeM': [low, high], 'widthM': high - low}
            except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
                reason = str(error)
                history[label] = {'status': 'unresolved-calibration', 'reason': reason, 'original': original,
                                  'exactHardConstraint': False, 'confidenceUpgraded': False}
                # A ridge scalar with no attributable source points cannot remain
                # a hard roof crest. Preserve it as historical provenance only.
                if label == 'ridge' and old is not None:
                    row[field] = None
                # Unverified/raw wall-top eaves remain their previous low/unknown
                # confidence priors. Small or rejected calibration must not erase them.
                report[label] = {'status': 'unresolved-calibration', 'originalM': old, 'activeM': row.get(field), 'reason': reason}
                message = label.capitalize() + ' calibration unresolved: ' + reason + '.'
                if message not in row.setdefault('unresolved', []):
                    row['unresolved'].append(message)
        audit['records'].append(report)
    audit['counts'] = {'sides': len(rows),
                       'verifiedPointBoundaries': sum(v.get('status') == 'calibrated-verified-points-uncertain-prior' for r in audit['records'] for v in (r['eave'], r['ridge'])),
                       'bracketEnvelopes': sum(r['eave'].get('status') == 'calibrated-full-width-envelope-uncertain-prior' for r in audit['records']),
                       'unresolvedRidges': sum(r['ridge'].get('status') == 'unresolved-calibration' for r in audit['records'])}
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--sides', type=Path, required=True, help='original registered side metadata directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.observations.resolve(), args.calibration.resolve()):
        parser.error('--output must differ from source inputs')
    result, audit = calibrate_boundaries(json.loads(args.observations.read_text()), json.loads(args.calibration.read_text()),
                                       load_registered_sides(args.sides), base=args.observations.parent)
    audit['inputs'] = {key: {'path': str(path), 'sha256': _sha(path)} for key, path in
                      [('observations', args.observations), ('calibration', args.calibration)]}
    if isinstance(result, list):
        result = {'observations': result}
    result['roofBoundaryCalibration'] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit['counts']))


if __name__ == '__main__':
    main()
