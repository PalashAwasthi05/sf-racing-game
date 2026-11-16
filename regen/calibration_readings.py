"""Apply image-supported camera orientation to an existing measured reading.

The exact wall homography keeps source geometry reusable. It does not recover
unseen storeys, validate provider positions, or upgrade reading confidence.
"""
import copy
import numpy as np
from regen.remap_readings import WallMap, remap


def calibrate_reading(reading, piece, source_view, calibration, eye_height):
    record = calibration.get('images', {}).get(source_view.get('image'))
    if record is None or not record.get('applied'):
        return copy.deepcopy(reading)
    if source_view.get('registered', False) != record.get('registration', False):
        raise ValueError('calibration and source crop use different camera placement modes')
    if source_view.get('cameraGroundZ') is None:
        raise ValueError('source crop has no camera height')
    old = np.asarray(record['referenceRotation'], float)
    new = np.asarray(record['rotation'], float)
    xy = np.asarray(record['cameraXY'], float)
    if old.shape != (3, 3) or new.shape != (3, 3) or xy.shape != (2,) or not np.isfinite([*old.flat, *new.flat, *xy]).all():
        raise ValueError('nonfinite or malformed camera calibration')
    for rotation in (old, new):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
            raise ValueError('camera calibration is not a proper rotation')
    center = [*xy, float(source_view['cameraGroundZ']) + eye_height]
    mapping = WallMap(piece, old, center, new, center)
    # A correction that sends the observed facade through the horizon is not
    # usable as a finite metric reading.
    x, z = mapping([0., piece['lengthM'], 0., piece['lengthM']],
                   [0., 0., piece['heightM'], piece['heightM']])
    if not np.isfinite([*x, *z]).all() or max(np.abs(x)) > 5 * piece['lengthM'] or max(np.abs(z)) > 5 * piece['heightM']:
        raise ValueError('camera correction projects the facade outside a bounded wall frame')
    result = remap(reading, mapping)
    result['cameraCalibration'] = {
        'status': 'applied', 'method': 'image-line-gravity-wall-homography',
        'image': source_view['image'], 'sourceSha256': record.get('sourceSha256'),
        'correctionDeg': record['correctionDeg'], 'positionValidated': False,
        'unseenGeometryRecovered': False,
    }
    return result
