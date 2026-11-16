import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from regen.calibrate_roof_components import calibrate_components


class RoofComponentCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        image = root / 'rect.png'
        Image.new('RGB', (120, 120), 'white').save(image)
        metadata = root / 'sides.json'
        piece = {'id': 'side', 'normal': [0, -1], 'frameU': [0, 10], 'lengthM': 10., 'heightM': 10.,
                 'groundZ': 0., 'planeV': 0., 'views': [{'image': 'image', 'rect': image.name, 'ppm': 10.,
                 'rectPad': 10, 'cameraGroundZ': 0., 'registered': 'bias'}]}
        metadata.write_text(json.dumps({'pieces': [piece]}))
        source = {'sourceKey': 'source', 'side': 'side', 'view': 0, 'imageType': 'rect', 'path': str(image),
                  'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'metadataPath': str(metadata),
                  'metadataImageId': 'image', 'inspected': True}
        self.feature = {'kind': 'conical-cap', 'x0M': 2., 'x1M': 6., 'bottomM': 3., 'topM': 8., 'confidence': 'high'}
        self.rows = [{'side': 'side', 'features': [self.feature]}]
        self.proposal = {'side': 'side', 'featureIndex': 0, 'kind': 'conical-cap', 'chosenSourceKey': 'source',
                         'confidence': 'medium', 'pixelEvidenceStatus': 'partial-visible-silhouette',
                         'observationBoundsM': {k: self.feature[k] for k in ('x0M', 'x1M', 'bottomM', 'topM')},
                         'silhouette': {'apexEstimatePx': {'point': [50, 40], 'uncertaintyPx': [2, 2]}}}
        self.proposals = {'sources': [source], 'features': [self.proposal]}

    def test_independent_apex_updates_cap_top_without_apertures_or_closed_outline(self):
        output, audit = calibrate_components(self.rows, self.proposals, {'images': {}})
        feature = output[0]['features'][0]
        self.assertEqual(audit['counts']['calibrated'], 1)
        self.assertAlmostEqual(feature['topM'], 7.)
        self.assertEqual(feature['bottomM'], 3.)
        self.assertEqual(feature['confidence'], 'high')
        self.assertEqual(feature['boundsEvidence']['fields']['bottomM']['confidence'], 'low')
        self.assertEqual(feature['boundsEvidence']['fields']['topM']['confidence'], 'medium')
        self.assertNotIn('frontEvidence', feature)

    def test_roof_eave_and_apex_keep_existing_aperture_metric_coordinates(self):
        self.feature['kind'] = self.proposal['kind'] = 'dormer'
        self.feature['frontEvidence'] = {'coordinateSpace': 'normalized-front',
            'apertures': [{'kind': 'window', 'outline01': [[.25, .1], [.75, .1], [.75, .5], [.25, .5]]}],
            'provenance': {'originalObservationBoundsM': self.proposal['observationBoundsM']}}
        self.proposal['componentPoints'] = [{'role': 'roof-eave', 'pointPx': [30, 50], 'uncertaintyPx': 1}]
        before = copy.deepcopy(self.feature['frontEvidence']['apertures'][0]['outline01'])
        output, audit = calibrate_components(self.rows, self.proposals, {'images': {}})
        feature = output[0]['features'][0]
        self.assertAlmostEqual(feature['roofEaveM'], 6.)
        self.assertEqual(feature['roofEaveEvidence']['status'], 'source-derived')
        self.assertGreater(feature['roofEaveEvidence']['readingUncertaintyM'], 0)
        old_metric = np.array(before) * [4, 5] + [2, 3]
        new_metric = np.array(feature['frontEvidence']['apertures'][0]['outline01']) * [4, 4] + [2, 3]
        np.testing.assert_allclose(old_metric, new_metric)

    def test_unknown_eave_keeps_fallback_and_apex_cannot_flatten_component(self):
        output, _ = calibrate_components(self.rows, self.proposals, {'images': {}})
        self.assertNotIn('roofEaveM', output[0]['features'][0])
        self.proposal['silhouette']['apexEstimatePx']['point'] = [50, 100]
        output, audit = calibrate_components(self.rows, self.proposals, {'images': {}})
        self.assertEqual(audit['counts']['unresolved'], 1)
        self.assertEqual(output, self.rows)

    def test_complete_bounds_without_apertures_are_source_constrained(self):
        self.proposal['pixelEvidenceStatus'] = 'accepted-approximate'
        self.proposal['silhouette'] = {}
        self.proposal['componentBoundsPx'] = [20, 30, 80, 80]
        output, audit = calibrate_components(self.rows, self.proposals, {'images': {}})
        feature = output[0]['features'][0]
        self.assertEqual(audit['counts']['calibrated'], 1)
        np.testing.assert_allclose([feature[k] for k in ('x0M', 'x1M', 'bottomM', 'topM')], [1, 7, 3, 8])
        self.assertEqual(feature['boundsEvidence']['fields']['x0M']['basis'], 'approximate-source-bounds')

    def test_short_visible_slopes_derive_apex_but_long_extrapolation_is_rejected(self):
        self.proposal['silhouette'] = {}
        self.proposal['componentPoints'] = [{'role': 'apex', 'sourcePixelsVerified': True, 'uncertaintyPx': 1,
            'pixelSegments': [[[30, 60], [45, 40]], [[55, 40], [70, 60]]]}]
        output, audit = calibrate_components(self.rows, self.proposals, {'images': {}})
        self.assertEqual(audit['counts']['calibrated'], 1)
        self.assertGreater(output[0]['features'][0]['topM'], 7.)
        self.proposal['componentPoints'][0]['pixelSegments'] = [[[10, 70], [25, 50]], [[95, 50], [110, 70]]]
        output, audit = calibrate_components(self.rows, self.proposals, {'images': {}})
        self.assertEqual(audit['counts']['unresolved'], 1)
        self.assertEqual(output, self.rows)


if __name__ == '__main__': unittest.main()
