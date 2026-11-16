"""Calibrate roof component bounds and independent typed boundary points.

A visible apex constrains topM even when width, base and hidden depth remain
priors. Apertures retain their metric positions when the component frame changes.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from regen.compile_roof_fronts import (_sha, _path, _source, _mapping, _map_metric,
    _pixels, _polygon, _trace, _number, BOUNDS)
from pipeline.roof_image_measurements import derive_gable_frontage


def _source_mapping(context, calibration, base):
    source_calibration = context['view'].get('gravityCalibration') or {}
    if not source_calibration.get('manifest'):
        return _mapping(context, calibration, base)
    manifest = _path(source_calibration['manifest'], base)
    original = json.loads(manifest.read_text())['images'][context['view']['image']]
    target = (calibration or {}).get('images', {}).get(context['view']['image'], original)
    if not np.allclose(original['cameraXY'], target['cameraXY'], atol=1e-8):
        raise ValueError('component source and target camera positions differ')
    if original.get('sourceSha256') != target.get('sourceSha256'):
        raise ValueError('component source and target panoramas differ')
    record = copy.deepcopy(target)
    record['referenceRotation'] = original['rotation']
    record['applied'] = not np.allclose(original['rotation'], target['rotation'], atol=1e-10)
    wm, provenance = _mapping(context, {'images': {context['view']['image']: record}}, base)
    provenance['sourceOrientationManifest'] = str(manifest)
    provenance['sourceOrientationManifestSha256'] = _sha(manifest)
    provenance['sourceAlreadyGravityCalibrated'] = True
    return wm, provenance


def _point(record, context, wm):
    role = record.get('role')
    if role not in ('apex', 'top', 'bottom', 'left', 'right', 'roof-eave'):
        raise ValueError('unsupported component point role')
    if record.get('pixelSegments') is not None:
        if role not in ('apex', 'top'):
            raise ValueError('opposing slopes can only establish an apex/top')
        measurement = {'sourcePixelsVerified': record.get('sourcePixelsVerified') is True,
            'frame': {'lengthM': context['piece']['lengthM'], 'heightRefM': context['height']},
            'views': [{'imageSizePx': context['size'], 'pixelsPerM': context['ppm'],
                       'padPx': context['pad'], 'uncertaintyPx': record.get('uncertaintyPx'),
                       'pixelSegments': record['pixelSegments']}]}
        fit = derive_gable_frontage(measurement)
        if fit['status'] != 'accepted':
            raise ValueError('component apex slope intersection rejected: ' + '; '.join(fit.get('unresolved', [])))
        view = fit['views'][0]
        old_point = np.array([[view['apexM']['xM'], view['apexM']['zM']]])
        point = _map_metric(old_point, context, wm)[0]
        uncertainty = view['uncertaintyM'][1]
        errors = []
        for dx in (-uncertainty['xM'], uncertainty['xM']):
            for dz in (-uncertainty['zM'], uncertainty['zM']):
                errors.append(np.abs(_map_metric(old_point + [dx, dz], context, wm)[0] - point))
        margin = np.max(errors, axis=0)
        detail = {'derivation': 'bounded-short-extrapolation-of-opposing-visible-slopes',
                  'pixelSegments': record['pixelSegments'], 'pixelUncertaintyPx': record['uncertaintyPx'],
                  'fit': fit}
    else:
        pixel = np.asarray(record.get('pointPx'), float)
        uncertainty = record.get('uncertaintyPx')
        if pixel.shape != (2,) or not np.isfinite(pixel).all() or np.any(pixel < 0) or np.any(pixel > context['size']):
            raise ValueError('component point lies outside its source image')
        if not _number(uncertainty) or uncertainty < 0:
            raise ValueError('component point requires explicit pixel uncertainty')
        def convert(p):
            raw = [[(p[0] - context['pad']) / context['ppm'], context['height'] - (p[1] - context['pad']) / context['ppm']]]
            return _map_metric(raw, context, wm)[0]
        point = convert(pixel)
        margin = np.max([abs(convert(pixel + [dx, dy]) - point)
                         for dx in (-uncertainty, uncertainty) for dy in (-uncertainty, uncertainty)], axis=0)
        detail = {'derivation': 'independent-visible-source-point', 'pointPx': pixel.tolist(),
                  'pixelUncertaintyPx': uncertainty}
    return {'role': role, 'pointM': point.tolist(), 'visibility': record.get('visibility'), 'readingUncertaintyM': margin.tolist(),
            'confidence': record.get('confidence', 'medium'), 'exact': False, **detail,
            'sourcePath': str(context['path']), 'sourceSha256': context['source']['sha256'],
            'metadataPath': str(context['metadata_path']), 'metadataSha256': _sha(context['metadata_path']),
            'image': context['view']['image'], 'view': context['source']['view']}


def _reframe_front(feature, old, new):
    evidence = feature.get('frontEvidence')
    if not isinstance(evidence, dict):
        return
    def remap(points):
        points = np.asarray(points, float)
        metric = points * [old[1] - old[0], old[3] - old[2]] + [old[0], old[2]]
        normalized = (metric - [new[0], new[2]]) / [new[1] - new[0], new[3] - new[2]]
        if np.any(normalized < -1e-7) or np.any(normalized > 1 + 1e-7):
            raise ValueError('component bound update would exclude known front geometry')
        return np.clip(normalized, 0, 1).tolist()
    if 'outline01' in evidence:
        evidence['outline01'] = remap(evidence['outline01'])
    for aperture in evidence.get('apertures', []):
        if 'outline01' in aperture:
            aperture['outline01'] = remap(aperture['outline01'])
        elif 'rect01' in aperture:
            x0, y0, x1, y1 = aperture['rect01']
            points = remap([[x0, y0], [x1, y1]])
            aperture['rect01'] = [*points[0], *points[1]]
    evidence.setdefault('provenance', {})['componentFrameUpdatedWithoutMovingApertures'] = True


def calibrate_components(document, proposals, calibration, *, base=Path('.')):
    output = copy.deepcopy(document)
    rows = output if isinstance(output, list) else output.get('observations', output.get('roofObservations'))
    sources = {}
    for source in proposals.get('sources', []):
        key = source['sourceKey']
        sources[key] = None if key in sources else source
    audit = {'records': [], 'sourceContract': {'componentPoints': 'Typed role apex|top|bottom|left|right with sourceKey, pointPx and uncertaintyPx; or verified opposing pixelSegments with explicit uncertaintyPx.',
        'roofEaveM': 'Independent roof-eave points constrain the visible cap-eave height; preserve visibility qualifiers, do not confuse with a concealed wall junction.',
        'bounds': 'Complete source outline/bounds constrain full extents. Missing extents remain explicitly low-confidence priors, independent of aperture availability.',
        'apertures': 'Existing metric aperture geometry is preserved while its normalized component frame changes.'}}
    for proposal in proposals.get('features', []):
        entry = {'side': proposal.get('side'), 'featureIndex': proposal.get('featureIndex')}
        try:
            matches = [r for r in rows if r.get('side') == entry['side']]
            if len(matches) != 1: raise ValueError('ambiguous/missing exact observation side')
            index = entry['featureIndex']
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(matches[0].get('features', [])):
                raise ValueError('feature index missing')
            original_feature = matches[0]['features'][index]
            feature = copy.deepcopy(original_feature)
            if feature.get('kind') != proposal.get('kind'): raise ValueError('component kind mismatch')
            guard = proposal.get('observationBoundsM') or {}
            front_provenance = (feature.get('frontEvidence') or {}).get('provenance') or {}
            source_guard = front_provenance.get('originalObservationBoundsM', {k: feature.get(k) for k in BOUNDS})
            if any(not _number(guard.get(k)) or not _number(source_guard.get(k))
                   or not np.isclose(guard[k], source_guard[k], atol=1e-6, rtol=0) for k in BOUNDS):
                raise ValueError('component original bounds guard mismatch')
            source = sources.get(proposal.get('chosenSourceKey'))
            context = _source(source, entry['side'], Path(base))
            wm, camera = _source_mapping(context, calibration, Path(base))
            old = [feature[k] for k in BOUNDS]
            # Already-compiled fronts carry their target frame. Do not apply the
            # same gravity map to their metric component bounds a second time.
            accepted_outline = (str(proposal.get('pixelEvidenceStatus', '')).startswith('accepted-approximate')
                                and proposal.get('confidence') in ('medium', 'high'))
            trace = _trace(proposal.get('silhouette')) if accepted_outline else None
            source_bounds = proposal.get('componentBoundsPx', proposal.get('boundsPx', (proposal.get('silhouette') or {}).get('boundsPx')))
            if not accepted_outline:
                source_bounds = None
            field_evidence = {}
            if trace is not None:
                points = _pixels(trace['pointsPx'], context, wm)
                x0, bottom, x1, top = _polygon(points, 'component outline').bounds
                new = [x0, x1, bottom, top]
                for field in BOUNDS: field_evidence[field] = {'basis': 'approximate-source-outline', 'confidence': proposal.get('confidence'), 'exact': False}
            elif source_bounds is not None:
                if isinstance(source_bounds, dict): source_bounds = source_bounds.get('boundsPx')
                if (not isinstance(source_bounds, list) or len(source_bounds) != 4
                        or not all(_number(v) for v in source_bounds)
                        or not (source_bounds[0] < source_bounds[2] and source_bounds[1] < source_bounds[3])):
                    raise ValueError('invalid complete source component bounds')
                a, b, c, d = source_bounds
                mapped = _pixels([[a, b], [c, b], [c, d], [a, d]], context, wm)
                lo, hi = mapped.min(axis=0), mapped.max(axis=0)
                new = [lo[0], hi[0], lo[1], hi[1]]
                for field in BOUNDS: field_evidence[field] = {'basis': 'approximate-source-bounds', 'confidence': proposal.get('confidence'), 'exact': False}
            elif front_provenance:
                new = list(old)
                for field in BOUNDS: field_evidence[field] = {'basis': 'already-mapped-original-prior', 'confidence': 'low', 'exact': False}
            else:
                x0, x1, bottom, top = [guard[k] for k in BOUNDS]
                mapped = _map_metric([[x0, bottom], [x1, bottom], [x1, top], [x0, top]], context, wm)
                lo, hi = mapped.min(axis=0), mapped.max(axis=0)
                new = [lo[0], hi[0], lo[1], hi[1]]
                for field in BOUNDS: field_evidence[field] = {'basis': 'calibrated-original-prior', 'confidence': 'low', 'exact': False}
            point_records = list(proposal.get('componentPoints') or [])
            apex = (proposal.get('silhouette') or {}).get('apexEstimatePx')
            if isinstance(apex, dict) and isinstance(apex.get('point'), list):
                uncertainty = apex.get('uncertaintyPx')
                if isinstance(uncertainty, list): uncertainty = max(uncertainty)
                point_records.append({'role': 'apex', 'pointPx': apex['point'], 'uncertaintyPx': uncertainty,
                                      'confidence': proposal.get('confidence', 'medium')})
            measured = []
            for record in point_records:
                point_context = _source(sources.get(record.get('sourceKey', proposal.get('chosenSourceKey'))), entry['side'], Path(base))
                point_wm, point_camera = _source_mapping(point_context, calibration, Path(base))
                result = _point(record, point_context, point_wm)
                result['calibration'] = point_camera
                measured.append(result)
            independent_fields = {}
            for field, roles, axis in [('roofEaveM', ('roof-eave',), 1), ('topM', ('apex', 'top'), 1), ('bottomM', ('bottom',), 1),
                                        ('x0M', ('left',), 0), ('x1M', ('right',), 0)]:
                values = [p for p in measured if p['role'] in roles]
                if not values: continue
                lower = min(p['pointM'][axis] - p['readingUncertaintyM'][axis] for p in values)
                upper = max(p['pointM'][axis] + p['readingUncertaintyM'][axis] for p in values)
                overlap_lo = max(p['pointM'][axis] - p['readingUncertaintyM'][axis] for p in values)
                overlap_hi = min(p['pointM'][axis] + p['readingUncertaintyM'][axis] for p in values)
                if len(values) > 1 and overlap_lo > overlap_hi:
                    raise ValueError('independent source point intervals disagree for ' + field)
                if field in BOUNDS:
                    new[BOUNDS.index(field)] = (lower + upper) / 2
                else:
                    independent_fields[field] = (lower + upper) / 2
                confidence = ('high' if all(p['confidence'] == 'high' for p in values) else
                              'medium' if all(p['confidence'] in ('medium', 'high') for p in values) else 'low')
                field_evidence[field] = {'basis': 'independent-source-point', 'confidence': confidence, 'exact': False,
                                         'intervalM': [lower, upper], 'readingUncertaintyM': (upper - lower) / 2}
            if not np.isfinite(new).all() or new[0] >= new[1] or new[2] >= new[3] - .05:
                raise ValueError('component calibration produces inverted or flattened bounds')
            if 'roofEaveM' in independent_fields and not new[2] < independent_fields['roofEaveM'] <= new[3]:
                raise ValueError('independent roof eave lies outside component vertical extent')
            _reframe_front(feature, old, new)
            feature.update(independent_fields)
            if 'roofEaveM' in independent_fields:
                feature['roofEaveEvidence'] = {'status': 'source-derived', 'confidence': field_evidence['roofEaveM']['confidence'], 'exact': False,
                    'readingUncertaintyM': field_evidence['roofEaveM']['readingUncertaintyM'],
                    'heightRangeM': field_evidence['roofEaveM']['intervalM'],
                    'sourceFields': [p for p in measured if p['role'] == 'roof-eave'],
                    'qualification': 'Approximate visible cap-eave edge prior; wall junction/depth may be concealed. Camera position remains unvalidated.'}
            feature.update({k: float(v) for k, v in zip(BOUNDS, new)})
            feature['boundsEvidence'] = {'status': ('calibrated-priors-only-unverified' if all('prior' in item['basis'] for item in field_evidence.values()) else 'source-constrained-with-explicit-priors'),
                'originalObservationBoundsM': guard, 'fields': field_evidence, 'componentPoints': measured,
                'sourcePath': str(context['path']), 'sourceSha256': context['source']['sha256'], 'calibration': camera,
                'hiddenDepthMeasured': False, 'overallConfidencePreserved': True,
                'unresolved': [field + ' remains an unverified prior awaiting a source trace.' for field, item in field_evidence.items() if item['confidence'] == 'low'],
                'roofEaveVisibility': [p.get('visibility') for p in point_records if p.get('role') == 'roof-eave']}
            matches[0]['features'][index] = feature
            entry.update(status='calibrated', boundsM={k: feature[k] for k in BOUNDS}, fields=field_evidence,
                         independentPoints=len(measured), confidence=feature.get('confidence'))
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            entry.update(status='unresolved', reason=str(error))
        audit['records'].append(entry)
    audit['counts'] = {'features': len(audit['records']), 'calibrated': sum(r['status'] == 'calibrated' for r in audit['records']),
                       'unresolved': sum(r['status'] == 'unresolved' for r in audit['records'])}
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observations', 'proposals', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.observations.resolve(), args.proposals.resolve(), args.calibration.resolve()):
        parser.error('output must differ from source inputs')
    result, audit = calibrate_components(json.loads(args.observations.read_text()), json.loads(args.proposals.read_text()),
                                       json.loads(args.calibration.read_text()), base=args.proposals.parent)
    audit['inputs'] = {k: {'path': str(p), 'sha256': _sha(p)} for k, p in
                      [('observations', args.observations), ('proposals', args.proposals), ('calibration', args.calibration)]}
    if isinstance(result, list): result = {'observations': result}
    result['roofComponentCalibration'] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit['counts']))


if __name__ == '__main__': main()
