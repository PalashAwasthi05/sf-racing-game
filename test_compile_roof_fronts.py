import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from regen.compile_roof_fronts import compile_fronts
from regen.remap_readings import WallMap


class RoofFrontCompilerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = self.root / 'chosen_rect.png'
        Image.new('RGB', (120, 120), 'white').save(self.image)
        self.piece = {'id': 'synthetic-side', 'normal': [0., -1.], 'frameU': [0., 10.],
                      'lengthM': 10., 'heightM': 10., 'groundZ': 0., 'planeV': 0.,
                      'views': [{'image': 'synthetic-image', 'rect': self.image.name, 'ppm': 10.,
                                 'rectPad': 10, 'cameraGroundZ': 0., 'registered': 'bias'}]}
        self.metadata = self.root / 'sides.json'
        self.metadata.write_text(json.dumps({'pieces': [self.piece]}))
        self.feature = {'kind': 'dormer', 'x0M': 2., 'x1M': 6., 'bottomM': 4., 'topM': 8.,
                        'roof': 'gable', 'windowCount': 1, 'confidence': 'medium'}
        self.observations = [{'side': 'synthetic-side', 'features': [self.feature]}]
        self.source = {'sourceKey': 'chosen', 'side': 'synthetic-side', 'view': 0, 'imageType': 'rect',
                       'path': str(self.image), 'sha256': hashlib.sha256(self.image.read_bytes()).hexdigest(),
                       'metadataPath': str(self.metadata), 'metadataImageId': 'synthetic-image',
                       'sizePx': [120, 120], 'pixelsPerMetre': 10., 'rectPadPx': 10, 'inspected': True}
        self.opening = {'kind': 'arched-glazed-window', 'boundsAreEnvelopeOnly': True,
                        'outlineApproxPx': {'pointsPx': [[40, 60], [60, 60], [60, 45], [50, 35], [40, 45]],
                                            'pointUncertaintyPx': 2},
                        'registeredFrameOutline': {'pointsM': [[999, 999]]}}
        self.proposal = {'side': 'synthetic-side', 'featureIndex': 0, 'kind': 'dormer',
                         'chosenSourceKey': 'chosen', 'confidence': 'medium',
                         'pixelEvidenceStatus': 'accepted-approximate',
                         'observationBoundsM': {k: self.feature[k] for k in ('x0M', 'x1M', 'bottomM', 'topM')},
                         'silhouette': {'outlineApproxPx': {'pointsPx': [[30, 70], [70, 70], [70, 30], [30, 30]],
                                                           'pointUncertaintyPx': 2}},
                         'openings': [self.opening]}
        self.proposals = {'sources': [self.source], 'features': [self.proposal]}

    def compile(self, calibration=None):
        return compile_fronts(self.observations, self.proposals, calibration, base=self.root)

    def test_pixels_use_verified_metadata_frame_and_ignore_cached_metric_data(self):
        before = copy.deepcopy(self.observations)
        output, audit = self.compile()
        self.assertEqual(audit['counts']['accepted'], 1)
        result = output[0]['features'][0]
        aperture = result['frontEvidence']['apertures'][0]
        np.testing.assert_allclose(aperture['outline01'], [[.25, .25], [.75, .25], [.75, .625], [.5, .875], [.25, .625]])
        np.testing.assert_allclose(aperture['provenance']['pointsM'][0], [3, 5])
        np.testing.assert_allclose(aperture['provenance']['uncertainty']['maximumReadingUncertaintyM'], [.2, .2])
        self.assertEqual(result['confidence'], 'medium')
        self.assertFalse(result['frontEvidence']['provenance']['confidenceUpgraded'])
        self.assertEqual(self.observations, before)

    def test_missing_silhouette_retains_bounds_and_uses_aperture_only_schema(self):
        self.proposal['silhouette'] = {'croppedApex': True, 'completeOutlinePx': None}
        output, audit = self.compile()
        result = output[0]['features'][0]
        self.assertEqual(audit['counts']['accepted'], 1)
        self.assertNotIn('outline01', result['frontEvidence'])
        for key in ('x0M', 'x1M', 'bottomM', 'topM'):
            self.assertEqual(result[key], self.feature[key])

    def test_louver_polygon_preserves_nonglass_semantics_without_blade_guess(self):
        self.opening.update(kind='arched-louver-vent', glazed=False,
                            runtimeSemantics={'shape': 'arched', 'openingType': 'louver', 'windowCount': 0})
        output, audit = self.compile()
        result = output[0]['features'][0]
        self.assertEqual(audit['counts']['accepted'], 1)
        self.assertEqual(result['windowCount'], 0)
        self.assertEqual(result['frontEvidence']['apertures'][0]['kind'], 'louver')
        self.assertNotIn('bladeCount', result['frontEvidence']['apertures'][0])
        self.assertNotIn('bladePitchM', result['frontEvidence']['apertures'][0])

    def test_stale_hash_and_original_bounds_guard_leave_feature_unresolved(self):
        for mutate in ('hash', 'bounds', 'image', 'frame', 'inspection'):
            with self.subTest(mutate=mutate):
                proposals = copy.deepcopy(self.proposals)
                if mutate == 'hash': proposals['sources'][0]['sha256'] = 'stale'
                if mutate == 'bounds': proposals['features'][0]['observationBoundsM']['topM'] = 9.
                if mutate == 'image': proposals['sources'][0]['metadataImageId'] = 'other-image'
                if mutate == 'frame': proposals['sources'][0]['rectPadPx'] = 0
                if mutate == 'inspection': proposals['sources'][0]['inspected'] = False
                output, audit = compile_fronts(self.observations, proposals, base=self.root)
                self.assertEqual(audit['counts']['accepted'], 0)
                self.assertEqual(audit['counts']['unresolved'], 1)
                self.assertEqual(output, self.observations)

    def test_occluded_or_envelope_only_bounds_never_become_rectangular_glass(self):
        for status, kind, envelope in [('occluded', 'rectangular-glazing-aperture', False),
                                        (None, 'arched-glazed-window', True),
                                        ('partial-only', 'rectangular-glazing-aperture', False)]:
            with self.subTest(status=status, kind=kind):
                proposal = copy.deepcopy(self.proposals)
                proposal['features'][0]['openings'] = [{'kind': kind, 'status': status,
                    'boundsPx': [40, 35, 60, 60], 'edgeUncertaintyPx': 2, 'boundsAreEnvelopeOnly': envelope}]
                output, audit = compile_fronts(self.observations, proposal, base=self.root)
                self.assertEqual(audit['counts']['accepted'], 0)
                self.assertEqual(output, self.observations)

    def test_clear_rectangular_bounds_are_accepted_without_inventing_arch_corners(self):
        self.proposal['openings'] = [{'kind': 'rectangular-glazing-aperture', 'boundsPx': [40, 35, 60, 60],
                                     'edgeUncertaintyPx': 2}]
        output, audit = self.compile()
        self.assertEqual(audit['counts']['accepted'], 1)
        self.assertEqual(len(output[0]['features'][0]['frontEvidence']['apertures'][0]['outline01']), 4)

    def test_wall_map_applies_to_pixels_and_existing_bounds_without_upgrading_confidence(self):
        self.proposal['silhouette'] = {}
        old = np.eye(3)
        new = Rotation.from_euler('x', 3, degrees=True).as_matrix()
        record = {'applied': True, 'registration': 'bias', 'referenceRotation': old.tolist(),
                  'rotation': new.tolist(), 'cameraXY': [5., -12.], 'correctionDeg': 3.}
        output, audit = self.compile({'images': {'synthetic-image': record}})
        self.assertEqual(audit['counts']['accepted'], 1)
        from frontage_crops_v15 import EYE_HEIGHT_M
        wm = WallMap(self.piece, old, [5., -12., EYE_HEIGHT_M], new, [5., -12., EYE_HEIGHT_M])
        xs, zs = wm([2, 6, 6, 2], [4, 4, 8, 8])
        result = output[0]['features'][0]
        np.testing.assert_allclose([result['x0M'], result['x1M'], result['bottomM'], result['topM']],
                                   [min(xs), max(xs), min(zs), max(zs)])
        mx, mz = wm(3., 5.)
        np.testing.assert_allclose(result['frontEvidence']['apertures'][0]['provenance']['pointsM'][0], [mx, mz])
        self.assertEqual(result['confidence'], self.feature['confidence'])
        self.assertFalse(result['frontEvidence']['provenance']['calibration']['positionValidated'])

    def test_duplicate_proposals_and_mixed_source_apertures_remain_unresolved(self):
        for mistake in ('duplicate', 'aperture-source'):
            with self.subTest(mistake=mistake):
                proposals = copy.deepcopy(self.proposals)
                if mistake == 'duplicate':
                    proposals['features'].append(copy.deepcopy(proposals['features'][0]))
                else:
                    proposals['features'][0]['openings'][0]['sourceSha256'] = 'different-view'
                output, audit = compile_fronts(self.observations, proposals, base=self.root)
                self.assertEqual(audit['counts']['accepted'], 0)
                self.assertEqual(output, self.observations)

    def test_low_confidence_and_partial_feature_are_unresolved(self):
        for field, value in [('confidence', 'low'), ('pixelEvidenceStatus', 'partial-occluded')]:
            proposal = copy.deepcopy(self.proposals)
            proposal['features'][0][field] = value
            output, audit = compile_fronts(self.observations, proposal, base=self.root)
            self.assertEqual(audit['counts']['accepted'], 0)
            self.assertEqual(output, self.observations)


if __name__ == '__main__': unittest.main()
