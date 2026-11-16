"""Verify reviewed stereo roof picks and compile bounded source roof parts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.roof_source_parts import derive


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(name, base):
    path = Path(name)
    return path if path.is_absolute() or path.exists() else Path(base) / path


def verify_source(source, side, base):
    source = copy.deepcopy(source)
    if source.get('side') != side or source.get('sourceAlreadyGravityCalibrated') is not True:
        raise ValueError('source side or gravity-corrected crop state mismatch')
    for prefix in ('source', 'metadata', 'calibration'):
        path = _path(source[prefix + 'Path'], base)
        if _sha(path) != source.get(prefix + 'Sha256'):
            raise ValueError(prefix + ' hash mismatch')
    metadata = json.loads(_path(source['metadataPath'], base).read_text())
    records = metadata.get('pieces', []) if isinstance(metadata, dict) else metadata
    pieces = [p for p in records if p.get('id') == side]
    if len(pieces) != 1:
        raise ValueError('source metadata side is missing or ambiguous')
    piece = pieces[0]
    index = source.get('view')
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(piece.get('views', [])):
        raise ValueError('chosen source view missing')
    view = piece['views'][index]
    if source.get('image') != view.get('image') or Path(view.get('rect', '')).name != Path(source['sourcePath']).name:
        raise ValueError('chosen source image/frame identity mismatch')
    for saved, actual in (('captureDate', 'captureDate'), ('registration', 'registered')):
        if source.get(saved) != view.get(actual):
            raise ValueError('source ' + saved + ' mismatch')
    frame = source['frame']
    for key in ('normal', 'frameU', 'planeV', 'lengthM', 'heightM', 'groundZ'):
        if not np.allclose(frame.get(key), piece.get(key), atol=1e-8, rtol=0):
            raise ValueError('source metadata frame mismatch: ' + key)
        frame[key] = copy.deepcopy(piece[key])
    for saved, actual in (('pixelsPerMetre', 'ppm'), ('padPx', 'rectPad')):
        if source.get(saved) != view.get(actual):
            raise ValueError('source crop scale/padding mismatch')
    calibration_path = _path(source['calibrationPath'], base)
    calibration = json.loads(calibration_path.read_text())
    record = calibration.get('images', {}).get(str(view['image']))
    if not record or record.get('applied') is not True or not (view.get('gravityCalibration') or {}).get('applied'):
        raise ValueError('source does not have an accepted gravity calibration')
    declared_manifest = _path(view['gravityCalibration']['manifest'], base)
    if _sha(declared_manifest) != source['calibrationSha256']:
        raise ValueError('crop references a different calibration manifest')
    rotation = np.asarray(record.get('rotation'), float)
    if rotation.shape != (3, 3) or not np.allclose(rotation, source.get('calibrationRotation'), atol=1e-10, rtol=0):
        raise ValueError('source crop rotation differs from manifest')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError('invalid calibration rotation')
    if record.get('registration') != view.get('registered'):
        raise ValueError('camera placement mode mismatch')
    # This constant is the real image sampler's eye height, not a proposal value.
    from frontage_crops_v15 import EYE_HEIGHT_M
    camera = [*record['cameraXY'], float(view['cameraGroundZ']) + EYE_HEIGHT_M]
    if not np.allclose(camera, source.get('cameraXYZ'), atol=1e-8, rtol=0) or source.get('eyeHeightM') != EYE_HEIGHT_M:
        raise ValueError('source camera differs from actual metadata/calibration')
    if record.get('sourcePath') and record.get('sourceSha256'):
        if _sha(_path(record['sourcePath'], base)) != record['sourceSha256']:
            raise ValueError('calibration panorama source hash mismatch')
    with Image.open(_path(source['sourcePath'], base)) as image:
        size = image.size
    ppm, pad = float(view['ppm']), float(view['rectPad'])
    expected = np.array([piece['lengthM'], piece['heightM']]) * ppm + 2 * pad
    if np.max(abs(expected - size)) > 2:
        raise ValueError('source crop dimensions differ from actual metric frame')
    # A corrected crop must be joined to the exact rotation which produced it.
    source.update(cameraXYZ=camera, sizePx=list(size), sampleConvention='pixel-center',
                  cameraXYValidated=False, calibrationRotation=rotation.tolist())
    return source


def compile_parts(observations, evidence, base=Path('.')):
    result = copy.deepcopy(observations)
    rows = result if isinstance(result, list) else result.get('observations', [])
    if not isinstance(rows, list):
        raise ValueError('observations must contain a list')
    sources = evidence.get('sources', [])
    by_key = {}
    for source in sources:
        key = source.get('sourceKey')
        if not key or key in by_key:
            raise ValueError('ambiguous evidence source key')
        by_key[key] = source
    audit = []
    for case in evidence.get('cases', []):
        side = case.get('side')
        record = {'side': side, 'status': 'unresolved', 'confidenceUpgraded': False}
        matches = [row for row in rows if row.get('side') == side]
        try:
            if len(matches) != 1:
                raise ValueError('observation side is missing or ambiguous')
            if case.get('status') != 'conditional-multiview-planar-patch-proposal':
                raise ValueError(case.get('status', 'source roof face unresolved'))
            if case.get('sourcePixelsVisuallyInspected') is not True:
                raise ValueError('paired source picks were not visually inspected')
            if case.get('confidence') not in ('medium', 'high'):
                raise ValueError('source face confidence is insufficient')
            keys = {m.get('sourceKey') for c in case['corners'] for m in c['matches']}
            verified = [verify_source(by_key[k], side, base) for k in sorted(keys)]
            proposed = case['part']
            part = {'kind': 'bounded-source-plane', 'confidence': case['confidence'],
                    'cornerOrder': copy.deepcopy(proposed['cornerOrder']),
                    'lowerEdgeCornerNames': copy.deepcopy(proposed['lowerEdgeCornerNames']),
                    'sources': verified,
                    'corners': [{key: copy.deepcopy(c[key]) for key in ('name', 'role', 'matches')} for c in case['corners']],
                    'proof': {'status': 'source-verified', 'sourcePixelsVisuallyInspected': True,
                              'cameraXYValidated': False, 'confidenceUpgraded': False,
                              'cameraUncertaintyIncluded': False,
                              'qualification': 'Conditional on registered camera positions; pixel bounds exclude camera, height and gravity estimation error.'},
                    'lowerJoin': copy.deepcopy(case.get('lowerJoin', {})),
                    'remainingUnknowns': copy.deepcopy(case.get('remainingUnknowns', []))}
            derived = derive(part)
            # This cache is review material only. Runtime derives geometry again.
            part['derivationAudit'] = derived['audit']
            matches[0].setdefault('roofParts', []).append(part)
            record.update(status='compiled-conditional-source-plane', **{'derivation': derived['audit']})
        except (ValueError, TypeError, KeyError, IndexError, OSError, np.linalg.LinAlgError) as error:
            record['reason'] = str(error)
            if len(matches) == 1:
                matches[0].setdefault('roofPartUnresolved', []).append({'reason': str(error), 'sourceCaseStatus': case.get('status')})
        audit.append(record)
    if isinstance(result, list):
        result = {'observations': result}
    result['roofPartCompilation'] = {'schemaVersion': 'bounded-source-roof-parts-v1', 'cases': audit,
                                     'confidenceUpgraded': False, 'cameraXYValidated': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    observations = json.loads(Path(args.observations).read_text())
    evidence = json.loads(Path(args.evidence).read_text())
    result = compile_parts(observations, evidence, Path('.'))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cases': [{key: c[key] for key in ('side', 'status', 'reason') if key in c}
                                for c in result['roofPartCompilation']['cases']]}))


if __name__ == '__main__':
    main()
