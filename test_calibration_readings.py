import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from regen.calibration_readings import calibrate_reading
from regen.remap_readings import WallMap


def fixture():
    piece = {'normal': [0., -1.], 'frameU': [0., 10.], 'planeV': 0.,
             'lengthM': 10., 'heightM': 15., 'groundZ': 0.}
    proper = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    stale = Rotation.from_euler('x', -18, degrees=True).as_matrix() @ proper
    view = {'image': 'synthetic', 'registered': 'bias', 'cameraGroundZ': 0.}
    manifest = {'images': {'synthetic': {'applied': True, 'registration': 'bias',
        'referenceRotation': stale.tolist(), 'rotation': proper.tolist(),
        'cameraXY': [5., -12.], 'correctionDeg': 18., 'sourceSha256': 'synthetic-test'}}}
    return piece, view, manifest


def test_wrong_camera_tilt_is_recovered_by_image_ray_projection_not_height_scaling():
    piece, view, manifest = fixture()
    record = manifest['images']['synthetic']
    inverse = WallMap(piece, record['rotation'], [5., -12., 2.],
                      record['referenceRotation'], [5., -12., 2.])
    # Derive a bad-frame measurement from a known true roof corner.
    _, wrong_top = inverse(5., 12.)
    source = {'wallTopM': float(wrong_top), 'storeys': [0., 2., 4.], 'confidence': 'low'}
    original = copy.deepcopy(source)
    fixed = calibrate_reading(source, piece, view, manifest, 2.)
    assert fixed['wallTopM'] == pytest.approx(12.)
    assert fixed['confidence'] == 'low'
    assert fixed['cameraCalibration']['positionValidated'] is False
    assert source == original
    assert not np.isclose(fixed['storeys'][0], 0.)  # changes offset and scale projectively


def test_calibration_rejects_wrong_source_state_and_nonrotation():
    piece, view, manifest = fixture()
    with pytest.raises(ValueError, match='placement modes'):
        calibrate_reading({}, piece, {**view, 'registered': 'skyline'}, manifest, 2.)
    manifest['images']['synthetic']['rotation'][0][0] = 5.
    with pytest.raises(ValueError, match='proper rotation'):
        calibrate_reading({}, piece, view, manifest, 2.)


def test_no_accepted_calibration_keeps_source_measurements():
    piece, view, manifest = fixture()
    manifest['images']['synthetic']['applied'] = False
    reading = {'wallTopM': 5., 'confidence': 'low'}
    assert calibrate_reading(reading, piece, view, manifest, 2.) == reading
