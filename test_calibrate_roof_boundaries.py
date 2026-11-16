import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from regen.calibrate_roof_boundaries import calibrate_boundaries
from regen.remap_readings import WallMap


class RoofBoundaryCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'img').mkdir()
        self.image = self.root / 'img' / 'synthetic_rect.png'
        Image.new('RGB', (120, 120), 'white').save(self.image)
        self.piece = {'id': 'synthetic-side', 'normal': [0., -1.], 'frameU': [0., 10.],
                      'lengthM': 10., 'heightM': 10., 'groundZ': 0., 'planeV': 0.,
                      'views': [{'image': 'synthetic-image', 'rect': self.image.name, 'ppm': 10.,
                                 'rectPad': 10, 'cameraGroundZ': 0., 'registered': 'bias'}]}
        self.metadata = self.root / 'sides.json'
        self.metadata.write_text(json.dumps({'pieces': [self.piece]}))
        self.sides = {'synthetic-side': {'status': 'ok', 'piece': self.piece, 'path': self.metadata}}
        self.row = {'side': 'synthetic-side', 'eaveM': 6., 'eaveConfidence': 'high',
                    'hostRidgeM': 8., 'ridgeConfidence': 'medium', 'anchorUncertaintyM': .1,
                    'unresolved': ['Existing uncertainty must survive.'],
                    'eaveEvidence': {'sources': [str(self.image)], 'frameM': {'width': 10., 'height': 10.},
                                     'pixelsPerMetre': 10., 'padPx': 10, 'junctionPixelYRange': [48, 52]}}
        old = np.eye(3)
        new = Rotation.from_euler('xyz', [2., 0., 1.], degrees=True).as_matrix()
        self.calibration = {'images': {'synthetic-image': {'applied': True, 'registration': 'bias',
                            'referenceRotation': old.tolist(), 'rotation': new.tolist(),
                            'cameraXY': [5., -12.], 'correctionDeg': 2.2}}}

    def test_unknown_x_uses_full_width_envelope_and_retains_confidence(self):
        original = copy.deepcopy(self.row)
        output, audit = calibrate_boundaries([self.row], self.calibration, self.sides)
        result = output[0]
        boundary = result['hostBoundaryCalibration']['eave']
        self.assertEqual(audit['counts']['bracketEnvelopes'], 1)
        self.assertTrue(boundary['unknownX'])
        self.assertFalse(boundary['exactHardConstraint'])
        self.assertFalse(boundary['historicalSourceHashesVerified'])
        self.assertEqual(result['eaveConfidence'], 'high')
        self.assertEqual(result['ridgeConfidence'], 'medium')
        self.assertIn('Existing uncertainty must survive.', result['unresolved'])
        low, high = boundary['heightRangeM']
        self.assertAlmostEqual(result['eaveM'], (low + high) / 2)
        self.assertGreaterEqual(result['anchorUncertaintyM'], (high - low) / 2 + boundary['pixelUncertaintyM'])
        record = self.calibration['images']['synthetic-image']
        from frontage_crops_v15 import EYE_HEIGHT_M
        wm = WallMap(self.piece, record['referenceRotation'], [5, -12, EYE_HEIGHT_M],
                     record['rotation'], [5, -12, EYE_HEIGHT_M])
        x, z = np.meshgrid(np.linspace(0, 10, 51), np.linspace(5.8, 6.2, 11))
        _, mapped = wm(x, z)
        self.assertLessEqual(low - 1e-8, mapped.min())
        self.assertGreaterEqual(high + 1e-8, mapped.max())
        self.assertEqual(self.row, original)

    def test_missing_ridge_points_quarantines_scalar_under_history(self):
        output, _ = calibrate_boundaries([self.row], self.calibration, self.sides)
        self.assertIsNone(output[0]['hostRidgeM'])
        self.assertEqual(output[0]['hostBoundaryCalibration']['ridge']['original']['valueM'], 8.)
        self.assertEqual(output[0]['ridgeConfidence'], 'medium')

    def test_rejected_pixels_are_not_revived_and_unchanged_raw_eave_survives(self):
        self.row['eaveEvidence']['status'] = 'rejected-after-source-audit'
        output, audit = calibrate_boundaries([self.row], self.calibration, self.sides)
        self.assertEqual(output[0]['eaveM'], 6.)
        self.assertEqual(audit['counts']['bracketEnvelopes'], 0)
        self.assertIn('rejected', output[0]['hostBoundaryCalibration']['eave']['reason'])
        del self.row['eaveEvidence']
        output, _ = calibrate_boundaries([self.row], {'images': {}}, self.sides)
        self.assertEqual(output[0]['eaveM'], 6.)

    def test_visible_edge_interval_never_becomes_hidden_eave(self):
        self.row['eaveM'] = None
        evidence = self.row['eaveEvidence']
        evidence['visibleRoofEdgePixelYRange'] = evidence.pop('junctionPixelYRange')
        output, _ = calibrate_boundaries([self.row], self.calibration, self.sides)
        self.assertIsNone(output[0]['eaveM'])
        self.assertIn('visibleRoofEdgeHeightRangeM', output[0]['eaveEvidence'])

    def test_historical_hash_and_frame_mismatch_leave_eave_unchanged(self):
        for mistake in ('hash', 'frame'):
            row = copy.deepcopy(self.row)
            if mistake == 'hash': row['eaveEvidence']['sourceHashes'] = {str(self.image): 'stale'}
            else: row['eaveEvidence']['frameM']['height'] = 11.
            output, audit = calibrate_boundaries([row], self.calibration, self.sides)
            self.assertEqual(output[0]['eaveM'], 6.)
            self.assertEqual(audit['counts']['bracketEnvelopes'], 0)

    def test_verified_2d_points_are_mapped_directly_with_exact_source_hash(self):
        source = {'side': 'synthetic-side', 'view': 0, 'imageType': 'rect', 'path': str(self.image),
                  'sha256': hashlib.sha256(self.image.read_bytes()).hexdigest(),
                  'metadataPath': str(self.metadata), 'metadataImageId': 'synthetic-image', 'inspected': True}
        self.row['eaveEvidence'] = {'pointMeasurements': [{'verified': True, 'source': source,
                                      'pointsPx': [[30, 50], [90, 50]], 'pointUncertaintyPx': 1.}]}
        output, audit = calibrate_boundaries([self.row], self.calibration, self.sides)
        self.assertEqual(audit['counts']['verifiedPointBoundaries'], 1)
        result = output[0]['hostBoundaryCalibration']['eave']
        self.assertTrue(result['historicalSourceHashesVerified'])
        self.assertEqual(len(result['measurements'][0]['pointsM']), 2)
        self.assertEqual(result['measurements'][0]['pointsPx'], [[30, 50], [90, 50]])
        source['sha256'] = 'stale'
        output, audit = calibrate_boundaries([self.row], self.calibration, self.sides)
        self.assertEqual(audit['counts']['verifiedPointBoundaries'], 0)
        self.assertEqual(output[0]['eaveM'], 6.)


if __name__ == '__main__': unittest.main()
