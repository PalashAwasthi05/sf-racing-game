"""Calibrated crop measurements must not receive a second camera/height warp."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


WORK = Path(__file__).resolve().parent


def fixture(tmp_path, calibrated=True):
    area = tmp_path / 'area'
    nominal = area / 'regen/sides'
    corrected = tmp_path / 'corrected'
    readings = tmp_path / 'readings'
    for folder in (nominal, corrected / 'img', readings):
        folder.mkdir(parents=True)
    piece = dict(id='example-side', building='example', part='example',
                 normal=[1., 0.], frameU=[0., 8.], planeV=0.,
                 lengthM=8., heightM=12., groundZ=0.,
                 views=[dict(image='old', rect='old.jpg', persp='old-p.jpg')])
    (nominal / 'sides.json').write_text(json.dumps({'pieces': [piece]}))
    corrected_piece = dict(piece, views=[dict(image='new', rect='new.jpg', persp='new-p.jpg',
                                             **({'gravityCalibration': {'applied': True}} if calibrated else {}))])
    (corrected / 'sides.json').write_text(json.dumps({'pieces': [corrected_piece]}))
    for name in ('new.jpg', 'new-p.jpg'):
        (corrected / 'img' / name).touch()
    reading = dict(id=piece['id'], registeredCrops=True, calibratedCrops=True,
                   thisBuilding=True, confidence='medium', use=0,
                   wallTopM=8., storeys=[0., 2., 4., 6., 8.],
                   materials=[dict(x0=0., x1=8., z0=0., z1=8., family='render', hex='#aabbcc')],
                   banks=[], openings=[], bays=[])
    (readings / 'example-side.json').write_text(json.dumps(reading))
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'profiles': [], 'method': 'fixture'}))
    calibration = tmp_path / 'calibration.json'
    calibration.write_text('{}')  # Already measured on corrected crops: never read this as a second warp.
    output = tmp_path / 'result.json'
    command = [sys.executable, str(WORK / 'areas/ring/profiles.py'), str(area), str(base),
               str(readings), str(output), '--nominal', '--calibrated-sides', str(corrected),
               '--camera-calibration', str(calibration)]
    return command, output


def test_corrected_crop_uses_corrected_camera_and_preserves_measured_height(tmp_path):
    command, output = fixture(tmp_path)
    result = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    profile, = json.loads(output.read_text())['profiles']
    assert profile['sourceIds'] == ['mapillary:new']
    assert profile['readingProvenance']['calibratedCropSource'] is True
    assert 'cameraCalibration' not in profile['readingProvenance']
    assert 'heights stretched' not in result.stdout
    assert profile['reading']['wallTopM'] == pytest.approx(8.)


def test_corrected_reading_refuses_uncorrected_metadata(tmp_path):
    command, _ = fixture(tmp_path, calibrated=False)
    result = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'requires its corrected crop metadata' in result.stderr
