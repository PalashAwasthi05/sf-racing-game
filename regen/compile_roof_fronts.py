"""Compile reviewed source pixels into independent roof-front aperture constraints.

Run from the project root with PYTHONPATH=. Source images and side metadata are
verified before conversion; cached metric proposals are deliberately never used.
Gravity correction does not validate camera position or upgrade confidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from regen.remap_readings import WallMap

BOUNDS = ('x0M', 'x1M', 'bottomM', 'topM')
CONFIDENT = ('medium', 'high')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _path(value, base):
    p = Path(value)
    if p.is_absolute() or p.exists():
        return p
    return base / p


def _source(source, side, base):
    if not isinstance(source, dict) or source.get('inspected') is not True:
        raise ValueError('source was not inspected')
    if source.get('side') != side or source.get('imageType') != 'rect':
        raise ValueError('source side/type mismatch; metric conversion requires the chosen rectified source')
    path = _path(source['path'], base)
    if not source.get('sha256') or _sha(path) != source['sha256']:
        raise ValueError('source image hash mismatch')
    metadata_path = _path(source['metadataPath'], base)
    metadata = json.loads(metadata_path.read_text())
    records = metadata.get('pieces', []) if isinstance(metadata, dict) else metadata
    matches = [p for p in records if isinstance(p, dict) and p.get('id') == side]
    if len(matches) != 1:
        raise ValueError('source metadata has missing or ambiguous exact side identity')
    piece = matches[0]
    index = source.get('view')
    views = piece.get('views', [])
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(views):
        raise ValueError('chosen source view missing from metadata')
    view = views[index]
    if view.get('image') != source.get('metadataImageId') or not view.get('image'):
        raise ValueError('chosen source image identity mismatch')
    if Path(view.get('rect', '')).name != path.name:
        raise ValueError('chosen rectified filename does not match metadata view')
    if source.get('metadataCaptureDate') is not None and source['metadataCaptureDate'] != view.get('captureDate'):
        raise ValueError('source capture date mismatch')
    if source.get('registrationLabel') is not None and source['registrationLabel'] != view.get('registered'):
        raise ValueError('source registration state mismatch')
    ppm, pad, height, length = view.get('ppm'), view.get('rectPad'), piece.get('heightM'), piece.get('lengthM')
    if not all(_number(v) for v in (ppm, pad, height, length)) or min(ppm, height, length) <= 0 or pad < 0:
        raise ValueError('source metadata has invalid metric frame')
    frame = np.asarray(piece.get('frameU'), float)
    if frame.shape != (2,) or not np.isfinite(frame).all() or not np.isclose(frame[1] - frame[0], length, atol=.02):
        raise ValueError('source metadata frameU/length mismatch')
    for field, actual in (('pixelsPerMetre', ppm), ('rectPadPx', pad)):
        if field in source and (not _number(source[field]) or not np.isclose(source[field], actual, atol=1e-6)):
            raise ValueError('source declared frame differs from metadata ' + field)
    with Image.open(path) as image:
        size = image.size
    if source.get('sizePx') is not None and list(size) != source['sizePx']:
        raise ValueError('source image dimensions changed')
    if abs(size[0] - (length * ppm + 2 * pad)) > 2 or abs(size[1] - (height * ppm + 2 * pad)) > 2:
        raise ValueError('source image size does not match registered metric frame')
    return dict(source=source, path=path, metadata_path=metadata_path, piece=piece, view=view,
                ppm=float(ppm), pad=float(pad), height=float(height), size=size)


def _mapping(context, calibration, base):
    image = context['view']['image']
    record = (calibration or {}).get('images', {}).get(image)
    provenance = {'status': 'not-applied', 'image': image, 'positionValidated': False,
                  'confidenceUpgraded': False, 'unseenGeometryRecovered': False}
    if not record or record.get('applied') is not True:
        provenance['reason'] = 'no accepted calibration supplied for this image'
        return None, provenance
    view, piece = context['view'], context['piece']
    if 'registration' in record and view.get('registered', False) != record['registration']:
        raise ValueError('calibration and source use different camera placement modes')
    if not _number(view.get('cameraGroundZ')):
        raise ValueError('source view has no finite camera ground elevation')
    old = np.asarray(record.get('referenceRotation', record.get('oldR')), float)
    new = np.asarray(record.get('rotation', record.get('newR')), float)
    xy = np.asarray(record.get('cameraXY'), float)
    if old.shape != (3, 3) or new.shape != (3, 3) or xy.shape != (2,) or not np.isfinite([*old.flat, *new.flat, *xy]).all():
        raise ValueError('calibration has malformed camera geometry')
    for rotation in (old, new):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
            raise ValueError('calibration matrix is not a proper rotation')
    for field in ('normal', 'planeV', 'groundZ'):
        value = np.asarray(piece.get(field), float)
        if not np.isfinite(value).all():
            raise ValueError('source wall frame is incomplete for calibration')
    if np.asarray(piece['normal']).shape != (2,) or np.linalg.norm(piece['normal']) < 1e-8:
        raise ValueError('source wall normal is invalid')
    # Panorama hash belongs to the panorama, never to the rectified source hash.
    if record.get('sourcePath') and record.get('sourceSha256'):
        if _sha(_path(record['sourcePath'], base)) != record['sourceSha256']:
            raise ValueError('calibration panorama hash mismatch')
    from frontage_crops_v15 import EYE_HEIGHT_M
    center = [*xy, float(view['cameraGroundZ']) + EYE_HEIGHT_M]
    wm = WallMap(piece, old, center, new, center)
    provenance.update(status='applied', method='image-gravity-wall-homography',
                      cameraXY=xy.tolist(), cameraZ=center[2], eyeHeightM=EYE_HEIGHT_M,
                      referenceRotation=old.tolist(), rotation=new.tolist(),
                      correctionDeg=record.get('correctionDeg'), sourceSha256=record.get('sourceSha256'))
    return wm, provenance


def _map_metric(points, context, wm):
    points = np.asarray(points, float)
    if wm is not None:
        x, z = wm(points[:, 0], points[:, 1])
        points = np.c_[x, z]
    if (not np.isfinite(points).all() or np.max(abs(points[:, 0])) > 5 * context['piece']['lengthM']
            or np.max(abs(points[:, 1])) > 5 * context['height']):
        raise ValueError('source projection leaves a bounded wall frame')
    return points


def _pixels(points, context, wm):
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points > np.asarray(context['size']))):
        raise ValueError('source pixel polygon is malformed or outside its image')
    raw = np.c_[(points[:, 0] - context['pad']) / context['ppm'],
                context['height'] - (points[:, 1] - context['pad']) / context['ppm']]
    return _map_metric(raw, context, wm)


def _polygon(points, label):
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 1e-8:
        raise ValueError(label + ' is not a valid nonempty polygon')
    return polygon


def _trace(record):
    if not isinstance(record, dict):
        return None
    trace = record.get('outlineApproxPx')
    if isinstance(trace, dict) and isinstance(trace.get('pointsPx'), list):
        return trace
    if isinstance(record.get('pointsPx'), list) and not record.get('outlineIsBoundingEnvelopeOnly'):
        return record
    return None


def _uncertainty(trace, points, context, wm):
    uncertainty = trace.get('pointUncertaintyPx', trace.get('edgeUncertaintyPx'))
    result = {'readingUncertaintyPx': uncertainty if _number(uncertainty) and uncertainty >= 0 else None,
              'registrationUncertaintyM': None,
              'qualification': 'Approximate source trace; reading uncertainty excludes camera position and registration error.'}
    if result['readingUncertaintyPx'] is not None:
        # Propagate image reading uncertainty through the same projective mapping.
        delta = float(uncertainty)
        samples = []
        raw = np.asarray(points, float)
        nominal = _pixels(raw, context, wm)
        for offset in ((-delta, -delta), (-delta, delta), (delta, -delta), (delta, delta)):
            shifted = raw + offset
            metric = np.c_[(shifted[:, 0] - context['pad']) / context['ppm'],
                           context['height'] - (shifted[:, 1] - context['pad']) / context['ppm']]
            samples.append(np.abs(_map_metric(metric, context, wm) - nominal))
        result['maximumReadingUncertaintyM'] = np.max(samples, axis=(0, 1)).tolist()
    return result


def _opening_kind(opening):
    semantics = opening.get('runtimeSemantics') or {}
    typed = semantics.get('openingType')
    kind = opening.get('kind')
    if typed == 'louver' or kind in ('louver', 'arched-louver-vent'):
        return 'louver'
    if typed == 'closed' or kind in ('closed', 'closed-panel'):
        return 'closed'
    if typed in ('window', 'glazing') or kind in ('window', 'arched-glazed-window', 'rectangular-window',
                                                'rectangular-glazing-aperture', 'rectangular-window-aperture'):
        return 'window'
    return None


def _compile_feature(feature, proposal, context, wm, calibration_provenance):
    for key in BOUNDS:
        if (not _number(feature.get(key)) or not _number((proposal.get('observationBoundsM') or {}).get(key))
                or not np.isclose(feature[key], proposal['observationBoundsM'][key], atol=1e-6, rtol=0)):
            raise ValueError('original observation bounds guard mismatch')
    if feature.get('kind') != proposal.get('kind') or feature.get('kind') != 'dormer':
        raise ValueError('feature identity/kind mismatch or unsupported non-dormer')
    if feature.get('confidence') not in CONFIDENT or proposal.get('confidence') not in CONFIDENT:
        raise ValueError('feature confidence is low or unknown')
    if not str(proposal.get('pixelEvidenceStatus', '')).startswith('accepted-approximate'):
        raise ValueError('pixel evidence is partial, occluded or unaccepted')
    silhouette_trace = _trace(proposal.get('silhouette'))
    silhouette = None
    if silhouette_trace is not None:
        silhouette_points = _pixels(silhouette_trace['pointsPx'], context, wm)
        silhouette = _polygon(silhouette_points, 'source silhouette')
        x0, bottom, x1, top = silhouette.bounds
    else:
        x0, x1, bottom, top = [feature[k] for k in BOUNDS]
        corners = _map_metric([[x0, bottom], [x1, bottom], [x1, top], [x0, top]], context, wm)
        x0, bottom = corners.min(axis=0)
        x1, top = corners.max(axis=0)
    if x1 - x0 < .05 or top - bottom < .05:
        raise ValueError('mapped feature bounds are degenerate')
    def normalize(points):
        normalized = (np.asarray(points) - [x0, bottom]) / [x1 - x0, top - bottom]
        if np.any(normalized < -1e-7) or np.any(normalized > 1 + 1e-7):
            raise ValueError('source aperture falls outside mapped feature bounds')
        return np.clip(normalized, 0, 1).tolist()
    apertures, skipped = [], []
    for opening_index, opening in enumerate(proposal.get('openings') or []):
        if not isinstance(opening, dict) or opening.get('status') in ('occluded', 'partial-only', 'unknown', 'unresolved'):
            skipped.append({'openingIndex': opening_index, 'reason': 'opening is occluded/partial/unknown'})
            continue
        if opening.get('sourcePath') and _path(opening['sourcePath'], context['metadata_path'].parent).resolve() != context['path'].resolve():
            raise ValueError('aperture source path differs from chosen source')
        if opening.get('sourceSha256') and opening['sourceSha256'] != context['source']['sha256']:
            raise ValueError('aperture source hash differs from chosen source')
        if opening.get('sourceKey') and opening['sourceKey'] != context['source'].get('sourceKey'):
            raise ValueError('aperture source identity differs from chosen source')
        kind = _opening_kind(opening)
        semantics = opening.get('runtimeSemantics') or {}
        if (kind == 'window' and (opening.get('glazed') is False or semantics.get('glass') is False)
                or kind in ('louver', 'closed') and (opening.get('glazed') is True or semantics.get('glass') is True)):
            raise ValueError('aperture type contradicts explicit glazing semantics')
        trace = _trace(opening)
        if trace is None:
            semantics = opening.get('runtimeSemantics') or {}
            rectangular = opening.get('kind') in ('rectangular-window', 'rectangular-glazing-aperture', 'rectangular-window-aperture') or semantics.get('shape') == 'rectangular'
            bounds = opening.get('boundsPx')
            if (kind != 'window' or not rectangular or opening.get('boundsAreEnvelopeOnly')
                    or not isinstance(bounds, list) or len(bounds) != 4 or not all(_number(v) for v in bounds)
                    or not (bounds[0] < bounds[2] and bounds[1] < bounds[3])):
                skipped.append({'openingIndex': opening_index, 'reason': 'no genuine aperture polygon or accepted rectangular window bounds'})
                continue
            a, b, c, e = bounds
            trace = {'pointsPx': [[a, b], [c, b], [c, e], [a, e]],
                     'pointUncertaintyPx': opening.get('edgeUncertaintyPx')}
        if kind is None:
            skipped.append({'openingIndex': opening_index, 'reason': 'opening kind is unknown'})
            continue
        points = _pixels(trace['pointsPx'], context, wm)
        polygon = _polygon(points, 'aperture')
        if silhouette is not None and not silhouette.buffer(1e-7).covers(polygon):
            raise ValueError('aperture lies outside source silhouette')
        aperture = {'kind': kind, 'outline01': normalize(points),
                    'provenance': {'sourceOpeningIndex': opening_index, 'pointsPx': trace['pointsPx'],
                                   'pointsM': points.tolist(), 'uncertainty': _uncertainty(trace, trace['pointsPx'], context, wm)}}
        # Pitch/count is passed only when explicitly present and valid; missing
        # source evidence never becomes a guessed number of louver blades.
        for field in ('bladeCount', 'bladePitchM', 'frameWidthM'):
            value = opening.get(field)
            if value is None:
                continue
            if not _number(value) or value <= 0:
                raise ValueError('invalid explicit ' + field)
            if field == 'bladeCount' and (not isinstance(value, int) or value > 1024):
                raise ValueError('invalid explicit bladeCount')
            if field.startswith('blade') and kind != 'louver':
                raise ValueError('blade evidence belongs to a louver aperture')
            aperture[field] = value
        if 'bladeCount' in aperture and 'bladePitchM' in aperture:
            raise ValueError('ambiguous simultaneous blade count and pitch')
        apertures.append(aperture)
    if not apertures:
        raise ValueError('no accepted visible typed apertures')
    result = copy.deepcopy(feature)
    result.update(x0M=float(x0), x1M=float(x1), bottomM=float(bottom), topM=float(top))
    provenance = {'status': 'accepted-approximate' if not skipped else 'partial-approximate',
                  'sourceKey': context['source'].get('sourceKey'), 'sourcePath': str(context['path']),
                  'sourceSha256': context['source']['sha256'], 'metadataPath': str(context['metadata_path']),
                  'metadataSha256': _sha(context['metadata_path']), 'side': proposal['side'],
                  'image': context['view']['image'], 'view': context['source']['view'],
                  'originalObservationBoundsM': proposal['observationBoundsM'],
                  'pixelConversion': {'heightM': context['height'], 'padPx': context['pad'], 'ppm': context['ppm'],
                                      'formula': 'x=(pixelX-pad)/ppm; z=heightM-(pixelY-pad)/ppm'},
                  'calibration': calibration_provenance, 'confidenceUpgraded': False,
                  'skippedOpenings': skipped, 'hiddenDepthMeasured': False}
    evidence = {'coordinateSpace': 'normalized-front', 'apertures': apertures, 'provenance': provenance}
    if silhouette is not None:
        evidence['outline01'] = normalize(silhouette_points)
        provenance['silhouette'] = {'pointsPx': silhouette_trace['pointsPx'], 'pointsM': silhouette_points.tolist(),
                                   'uncertainty': _uncertainty(silhouette_trace, silhouette_trace['pointsPx'], context, wm)}
    else:
        provenance['silhouette'] = {'status': 'generated-front-retained', 'reason': 'no complete source silhouette trace'}
    result['frontEvidence'] = evidence
    result['windowCount'] = sum(a['kind'] == 'window' for a in apertures)
    return result


def compile_fronts(observations, proposals, calibration=None, *, base=Path('.')):
    """Return a copied observation document and explicit per-proposal audit."""
    output = copy.deepcopy(observations)
    rows = output if isinstance(output, list) else output.get('observations', output.get('roofObservations'))
    if not isinstance(rows, list):
        raise ValueError('observations must be a list or contain observations/roofObservations')
    sources = {}
    for source in proposals.get('sources', []):
        key = source.get('sourceKey')
        if key in sources:
            sources[key] = None  # ambiguity is unresolved, never filename-order selection
        else:
            sources[key] = source
    audit = {'method': 'verified-source-pixels-to-roof-front-apertures', 'accepted': [], 'unresolved': [],
             'limitations': ['Camera position, heading and facade depth remain uncertain.',
                            'Approximate source traces retain their confidence; no unseen depth is recovered.']}
    identities = Counter((p.get('side'), p.get('featureIndex')) for p in proposals.get('features', [])
                         if isinstance(p, dict) and isinstance(p.get('side'), str)
                         and isinstance(p.get('featureIndex'), int))
    for proposal in proposals.get('features', []):
        entry = {'side': proposal.get('side'), 'featureIndex': proposal.get('featureIndex')}
        try:
            key = (entry['side'], entry['featureIndex'])
            if identities.get(key, 0) > 1:
                raise ValueError('duplicate proposals leave the feature ambiguous')
            matches = [r for r in rows if r.get('side') == entry['side']]
            if len(matches) != 1:
                raise ValueError('missing or ambiguous observation side')
            features = matches[0].get('features', [])
            index = entry['featureIndex']
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(features):
                raise ValueError('observation feature index missing')
            # Gate unsupported evidence before opening files or applying maps.
            if not str(proposal.get('pixelEvidenceStatus', '')).startswith('accepted-approximate'):
                raise ValueError('pixel evidence is partial, occluded or unaccepted')
            context = _source(sources.get(proposal.get('chosenSourceKey')), entry['side'], Path(base))
            wm, calibration_provenance = _mapping(context, calibration, Path(base))
            replacement = _compile_feature(features[index], proposal, context, wm, calibration_provenance)
            features[index] = replacement
            audit['accepted'].append({**entry, 'apertures': len(replacement['frontEvidence']['apertures']),
                                      'status': replacement['frontEvidence']['provenance']['status']})
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            audit['unresolved'].append({**entry, 'reason': str(error)})
    audit['counts'] = {'proposals': len(proposals.get('features', [])), 'accepted': len(audit['accepted']),
                       'unresolved': len(audit['unresolved'])}
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.observations.resolve(), args.proposals.resolve(),
                                *([args.calibration.resolve()] if args.calibration else [])}:
        parser.error('--output must be a new path distinct from source inputs')
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    observations, audit = compile_fronts(json.loads(args.observations.read_text()),
        json.loads(args.proposals.read_text()), calibration, base=args.proposals.parent)
    audit['inputs'] = {name: {'path': str(path), 'sha256': _sha(path)} for name, path in
                      [('observations', args.observations), ('proposals', args.proposals)] +
                      ([('calibration', args.calibration)] if args.calibration else [])}
    if isinstance(observations, list):
        document = {'observations': observations}
    else:
        document = observations
    document['frontEvidenceCompilation'] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit['counts']))


if __name__ == '__main__':
    main()
