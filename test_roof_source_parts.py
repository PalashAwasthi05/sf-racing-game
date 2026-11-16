"""Bounded photo-derived roofs preserve source attribution and native ownership."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from shapely.geometry import Polygon, Point, box

from pipeline.roof_source_parts import derive, resolve, compose
from regen.compile_roof_parts import compile_parts


def _part():
    frame = {'normal': [0, -1], 'planeV': 0., 'frameU': [0., 10.],
             'lengthM': 10., 'heightM': 15., 'groundZ': 0.}
    cameras = [[-4., -15., 2.], [12., -15., 2.]]
    sources = [{'sourceKey': str(i), 'image': str(i), 'side': 'front', 'view': i,
                'frame': frame, 'cameraXYZ': camera, 'pixelsPerMetre': 100., 'padPx': 0.,
                'sourceSha256': 'a' * 64, 'metadataSha256': 'b' * 64, 'calibrationSha256': 'c' * 64,
                'sampleConvention': 'pixel-center', 'sourceAlreadyGravityCalibrated': True,
                'sizePx': [1000, 1500], 'cameraXYValidated': False}
               for i, camera in enumerate(cameras)]
    vertices = [[2., -.3, 6.], [8., -.3, 6.], [8., 1., 8.], [2., 1., 8.]]
    names = ['a', 'b', 'c', 'd']
    corners = []
    for name, point in zip(names, vertices):
        matches = []
        for i, camera in enumerate(map(np.array, cameras)):
            point = np.asarray(point)
            t = (-.05 - camera[1]) / (point[1] - camera[1])
            wall = camera + (point - camera) * t
            pixel = [wall[0] * 100 - .5, (15 - wall[2]) * 100 - .5]
            matches.append({'sourceKey': str(i), 'pointPx': pixel, 'uncertaintyPx': .2})
        corners.append({'name': name, 'role': 'observed roof edge intersection', 'matches': matches})
    return {'kind': 'bounded-source-plane', 'confidence': 'medium', 'sources': sources,
            'corners': corners, 'cornerOrder': names, 'lowerEdgeCornerNames': ['a', 'b'],
            'proof': {'status': 'source-verified', 'sourcePixelsVisuallyInspected': True}}, np.asarray(vertices)


def _profile():
    return {'id': 'regen-front', 'match': {'partId': 'owned'}, 'normal': [0, -1],
            'frameU': [0, 10], 'planeV': 0.}


def _shell():
    top = trimesh.Trimesh(vertices=[[0, 0, 7], [10, 0, 7], [10, 12, 7], [0, 12, 7]],
                          faces=[[0, 1, 2], [0, 2, 3]], process=False)
    return {'surface': top, 'skirt': None, 'audit': {}, 'boundaryChains': [['retained']]}


def _height(mesh, xy):
    found = []
    for triangle in mesh.triangles:
        p = Polygon(triangle[:, :2])
        if p.area > 1e-9 and p.buffer(1e-8).covers(Point(*xy)):
            weights = np.linalg.solve(np.vstack([triangle[:, :2].T, np.ones(3)]), [*xy, 1])
            found.append(float(weights @ triangle[:, 2]))
    return max(found) if found else None


def test_stereo_reconstruction_ignores_poisoned_cached_pitch_and_vertices():
    part, expected = _part()
    part['conditionalVerticesXYZ'] = [[900, 900, 900]] * 4
    part['planeZCoefficients'] = [0, 0, 900]
    result = derive(part)
    np.testing.assert_allclose(result['vertices'], expected, atol=1e-10)
    assert result['audit']['maximumPlanarityResidualM'] < 1e-10
    assert result['audit']['cameraXYValidated'] is False
    assert result['audit']['cameraUncertaintyIncluded'] is False
    assert all(max(r['pixelOnlyUncertaintyXYZM']) > 0 for r in result['audit']['corners'])


def test_source_plane_adds_only_bounded_overhang_and_preserves_lower_rear():
    part, vertices = _part()
    items, audit = resolve([{'side': 'front', 'roofParts': [part]}], [_profile()], 'owned', 'source', box(0, 0, 10, 12))
    assert len(items) == 1
    shell = compose(_shell(), items)
    assert shell['audit']['boundedSourceRoofParts'][0]['status'] == 'applied-conditional-source-plane'
    assert _height(shell['surface'], [5, .9]) == pytest.approx(6 + 2 * 1.2 / 1.3)
    assert _height(shell['surface'], [5, -.2]) == pytest.approx(6 + 2 * .1 / 1.3)
    assert _height(shell['surface'], [5, -.4]) is None
    assert _height(shell['surface'], [5, 10]) == pytest.approx(7)
    assert shell['boundaryChains'] == [['retained']]
    # Source face plus provisional returns is a closed, positive wedge.
    wedge = trimesh.util.concatenate([shell['surface'].submesh([np.arange(len(shell['surface'].faces)-2, len(shell['surface'].faces))], append=True), shell['skirt']])
    wedge.merge_vertices()
    assert wedge.is_watertight and wedge.volume > 0


def test_existing_higher_roof_survives_without_inventing_lower_rear_pitch():
    part, _ = _part()
    item = derive(part)
    shell = _shell()
    shell['surface'].vertices[:, 2] = 9
    shell = compose(shell, [item])
    assert _height(shell['surface'], [5, .8]) == 9
    assert _height(shell['surface'], [5, 8]) == 9
    assert shell['audit']['boundedSourceRoofParts'][0]['nativeRoofRemovedM2'] == 0


def test_courtyard_neighbor_and_profile_frame_conflicts_are_local_unresolved():
    part, _ = _part()
    footprint = Polygon([(0, 0), (10, 0), (10, 12), (0, 12)], holes=[[(4, .1), (6, .1), (6, .8), (4, .8)]])
    items, audit = resolve([{'side': 'front', 'roofParts': [part]}], [_profile()], 'owned', 'source', footprint)
    assert not items and 'courtyard' in audit[0]['reason']
    items, _ = resolve([{'side': 'front', 'roofParts': [part]}], [_profile()], 'owned', 'source', box(0, 0, 10, 12))
    shell = compose(_shell(), items, [box(3, -.4, 7, 0)])
    assert shell['audit']['boundedSourceRoofParts'][0]['status'] == 'unresolved'
    assert shell['surface'].area == 120
    profile = _profile(); profile['planeV'] = 1
    items, audit = resolve([{'side': 'front', 'roofParts': [part]}], [profile], 'owned', 'source', box(0, 0, 10, 12))
    assert not items and 'frames differ' in audit[0]['reason']


def test_bad_ray_match_does_not_fall_back_to_generic_roof():
    part, _ = _part()
    part['corners'][0]['matches'][1]['pointPx'][1] += 60
    with pytest.raises(ValueError, match='rays disagree'):
        derive(part)
    items, audit = resolve([{'side': 'front', 'roofParts': [part]}], [_profile()], 'owned', 'source', box(0, 0, 10, 12))
    assert not items and audit[0]['status'] == 'unresolved'


def test_short_baseline_cannot_gain_acceptance_from_wider_uncertainty_cones():
    part, vertices = _part()
    part['sources'][1]['cameraXYZ'] = [-3.9, -15., 2.]
    for corner, point in zip(part['corners'], vertices):
        camera = np.asarray(part['sources'][1]['cameraXYZ'])
        wall = camera + (point - camera) * ((-.05 - camera[1]) / (point[1] - camera[1]))
        corner['matches'][1]['pointPx'] = [wall[0] * 100 - .5, (15 - wall[2]) * 100 - .5]
    with pytest.raises(ValueError, match='weak stereo triangulation'):
        derive(part)


def _disk_evidence(tmp_path):
    part, _ = _part()
    def write(path, value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / 'calibration.json'
    views = [{'image': str(i), 'rect': f'v{i}.jpg', 'registered': 'skyline', 'captureDate': '2022-01-01',
              'ppm': 100., 'rectPad': 0., 'cameraGroundZ': -.4,
              'gravityCalibration': {'manifest': str(manifest), 'applied': True}} for i in range(2)]
    frame = part['sources'][0]['frame']
    metadata = tmp_path / 'sides.json'
    metadata_hash = write(metadata, [{'id': 'front', **frame, 'views': views}])
    calibration_hash = write(manifest, {'images': {str(i): {'applied': True, 'rotation': np.eye(3).tolist(),
                                            'cameraXY': s['cameraXYZ'][:2], 'registration': 'skyline'}
                                            for i, s in enumerate(part['sources'])}})
    for i, source in enumerate(part['sources']):
        path = tmp_path / f'v{i}.jpg'; Image.new('RGB', (1000, 1500)).save(path)
        source.update(sourcePath=str(path), sourceSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      metadataPath=str(metadata), metadataSha256=metadata_hash,
                      calibrationPath=str(manifest), calibrationSha256=calibration_hash,
                      calibrationRotation=np.eye(3).tolist(), captureDate='2022-01-01', registration='skyline', eyeHeightM=2.4)
    evidence = {'sources': part['sources'], 'cases': [{'side': 'front',
                'status': 'conditional-multiview-planar-patch-proposal', 'confidence': 'medium',
                'sourcePixelsVisuallyInspected': True, 'corners': part['corners'],
                'part': {k: part[k] for k in ('cornerOrder', 'lowerEdgeCornerNames')}}]}
    return evidence


def test_adapter_verifies_actual_crop_metadata_rotation_and_camera(tmp_path):
    evidence = _disk_evidence(tmp_path)
    result = compile_parts([{'side': 'front', 'confidence': 'low', 'features': [{'kept': True}]}], evidence)
    row = result['observations'][0]
    assert row['confidence'] == 'low' and row['features'] == [{'kept': True}]
    assert len(row['roofParts']) == 1
    assert result['roofPartCompilation']['cases'][0]['status'] == 'compiled-conditional-source-plane'
    for field, value, reason in [('sourceSha256', 'f' * 64, 'hash mismatch'),
                                 ('image', 'other-image', 'identity mismatch'),
                                 ('cameraXYZ', [1, 1, 1], 'camera differs')]:
        stale = copy.deepcopy(evidence); stale['sources'][0][field] = value
        result = compile_parts([{'side': 'front'}], stale)
        assert not result['observations'][0].get('roofParts')
        assert reason in result['roofPartCompilation']['cases'][0]['reason']


def test_real_reviewed_source_roof_face_derives_and_composes_without_reusing_cached_xyz():
    path = Path('areas/review/roof-rethink/observations-v8-source-roof-parts.json')
    if not path.exists():
        pytest.skip('external reviewed source fixture not installed')
    observations = json.loads(path.read_text())['observations']
    row = next(r for r in observations if r.get('roofParts'))
    part = copy.deepcopy(row['roofParts'][0]); part['derivationAudit']['verticesXYZ'] = [[1, 2, 300]] * 4
    item = derive(part)
    assert item['audit']['pitchDeg'] == pytest.approx(65.5178848433, abs=.001)
    assert item['audit']['maximumPlanarityResidualM'] < .02
    assert max(c['raySeparationM'] for c in item['audit']['corners']) < .12
    polygon = item['polygon']
    shell = {'surface': trimesh.creation.box(extents=[20, 20, .01]), 'skirt': None, 'audit': {}}
    shell['surface'].apply_translation([*polygon.centroid.coords[0], -38.])
    before = _height(shell['surface'], [polygon.centroid.x + 5, polygon.centroid.y + 5])
    result = compose(shell, [item])
    assert result['audit']['boundedSourceRoofParts'][0]['status'] == 'applied-conditional-source-plane'
    assert result['audit']['boundedSourceRoofParts'][0]['measuredFaceAreaM2'] > 10
    assert _height(result['surface'], [polygon.centroid.x + 5, polygon.centroid.y + 5]) == before
