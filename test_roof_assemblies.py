import unittest
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box

from pipeline.roof_assemblies import resolve


class RoofAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "id": "regen-source-a",
            "match": {"partId": "part-a", "featureId": "source-a"},
            "normal": [1., 0.],
            "frameU": [100., 110.],
            "planeV": 20.,
            "reading": {"confidence": "high"},
            "bays": [{"rect": [.35, .1, .65, .9], "projectionM": .8}],
        }
        self.observation = {
            "side": "source-a",
            "features": [{"kind": "dormer", "x0M": 4., "x1M": 6.,
                          "bottomM": 3., "topM": 4.5, "roof": "gable",
                          "windowCount": 1, "wallColor": "#ddd", "trimColor": "#fff",
                          "roofColor": "#555", "confidence": "high"}],
        }
        # The source wall's normal is +x, so the roof lies in the -x direction.
        self.footprint = box(10., 95., 21., 115.)
        self.samples = np.array([[18.5, 105., 5.], [18.5, 105.5, 5.], [10., 100., 6.]])

    def test_source_frame_transform_and_supported_dormer_spec(self):
        specs, audit = resolve([self.observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., self.samples)
        self.assertEqual(audit["status"], "resolved")
        self.assertEqual(len(specs), 1)
        # local midpoint 5 m is translated by frameU[0] to u=105; inward axis is -x.
        self.assertAlmostEqual(specs[0]["xy"][0], 17.6)
        self.assertAlmostEqual(specs[0]["xy"][1], 105.)
        self.assertEqual(specs[0]["facing"], [1., 0.])
        self.assertEqual(specs[0]["widthM"], 2.)
        self.assertAlmostEqual(specs[0]["depthM"], 1.8)
        self.assertEqual(specs[0]["windowCount"], 1)
        self.assertAlmostEqual(audit["accepted"][0]["insetM"], 1.5)
        self.assertAlmostEqual(audit["accepted"][0]["frontSurfaceZM"], 5.)

    def test_source_front_evidence_is_preserved_and_depth_stays_a_prior(self):
        front = {"coordinateSpace": "normalized-front", "outline01": [[0, 0], [1, 0], [.5, 1]],
                 "windows": []}
        observation = {**self.observation, "features": [{**self.observation["features"][0],
                                                           "frontEvidence": front}]}
        specs, audit = resolve([observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., self.samples)
        self.assertEqual(specs[0]["frontEvidence"], front)
        self.assertEqual(audit["accepted"][0]["frontGeometry"], "source-described")
        self.assertEqual(audit["accepted"][0]["depthStatus"], "inferred-width-prior")

    def test_source_component_eave_preserves_observed_roof_rise(self):
        observed = {**self.observation, "features": [{**self.observation["features"][0],
            "roofEaveM": 3.4, "roofEaveEvidence": {"status": "source-derived", "uncertaintyM": .12}}]}
        specs, audit = resolve([observed], [self.profile], "part-a", "source-a",
                               self.footprint, 2., self.samples)
        self.assertAlmostEqual(specs[0]["eaveM"], 3.4)
        self.assertAlmostEqual(specs[0]["ridgeM"], 4.5)
        self.assertEqual(audit['accepted'][0]['eaveStatus'], 'source-derived-uncertain-prior')

    def test_invalid_or_unattributed_component_eave_keeps_explicit_prior_status(self):
        for eave, evidence in ((4.6, {'status': 'source-derived'}), (3.4, None),
                               (float('nan'), {'status': 'source-derived'})):
            with self.subTest(eave=eave, evidence=evidence):
                observed = {**self.observation, "features": [{**self.observation["features"][0],
                    "roofEaveM": eave, "roofEaveEvidence": evidence}]}
                specs, audit = resolve([observed], [self.profile], "part-a", "source-a",
                                       self.footprint, 2., self.samples)
                self.assertAlmostEqual(specs[0]['eaveM'], 3.9)
                self.assertEqual(audit['accepted'][0]['eaveStatus'], 'inferred-width-prior')
                self.assertEqual(audit['rejected'][0]['reason'], 'invalid-or-unattributed-component-eave')

    def test_dormer_front_intersects_fitted_slope_and_keeps_lateral_center(self):
        # The fitted shell falls from z=7 at the wall plane to z=5.5 at 5 m
        # inward. The observed dormer sill is at z0 + bottomM = 5.5.
        rows = []
        for inset in np.arange(0., 5.01, .25):
            for lateral in np.arange(-.25, .251, .25):
                rows.append([20. - inset, 105. + lateral,
                             7. - .3 * inset + .1 * lateral])
        observation = {**self.observation, "features": [{**self.observation["features"][0],
                                                           "bottomM": 3.5}]}
        specs, audit = resolve([observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., np.asarray(rows))
        self.assertEqual(audit["status"], "resolved")
        accepted = audit["accepted"][0]
        self.assertAlmostEqual(accepted["frontSurfaceZM"], 5.5, delta=.2)
        # The dormer center is half its inferred depth behind the intersected
        # front, but remains on the original source-frame x midpoint ray.
        self.assertAlmostEqual(specs[0]["xy"][1], 105.)
        self.assertAlmostEqual(specs[0]["xy"][0], 20. - accepted["insetM"] - specs[0]["depthM"] / 2.)
        self.assertAlmostEqual(accepted["insetM"], 5., delta=.2)

    def test_sparse_large_triangles_support_barycentric_front_height(self):
        # Only four vertices cover the roof plane. No vertex lies near the
        # measured sill intersection at (15, 105), so this must query triangles.
        vertices = np.array([
            [14., 95., 4.2], [20., 95., 6.],
            [20., 115., 8.], [14., 115., 6.2],
        ])
        surface = SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]))
        observation = {**self.observation, "features": [{**self.observation["features"][0],
                                                           "bottomM": 3.5}]}
        specs, audit = resolve([observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., surface)
        self.assertEqual(audit["status"], "resolved")
        accepted = audit["accepted"][0]
        self.assertAlmostEqual(accepted["frontSurfaceZM"], 5.5, delta=.01)
        self.assertAlmostEqual(accepted["insetM"], 5., delta=.1)
        self.assertAlmostEqual(specs[0]["xy"][1], 105.)

    def test_registered_wall_inside_eave_can_find_outward_sill_intersection(self):
        # Frame is 1 m inside the mapped eave. The observed sill lies outward
        # of that frame, still within the roof, not on the high inward slope.
        surface = SimpleNamespace(vertices=np.array([[18., 95., 8.], [21., 95., 5.],
                                                      [21., 115., 5.], [18., 115., 8.]]),
                                  faces=np.array([[0, 1, 2], [0, 2, 3]]))
        specs, audit = resolve([self.observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., surface)
        self.assertEqual(len(specs), 1)
        self.assertAlmostEqual(audit["accepted"][0]["insetM"], -1., delta=.1)
        self.assertAlmostEqual(audit["accepted"][0]["frontSurfaceZM"], 5., delta=.1)

    def test_exact_part_identity_excludes_sibling_with_same_source(self):
        sibling = {**self.profile, "id": "regen-source-a-sibling",
                   "match": {"partId": "part-b", "featureId": "source-a"}}
        # The matching profile side is explicit here, as independent readings may
        # share a source feature while belonging to different building parts.
        sibling["id"] = "regen-source-a"
        specs, _ = resolve([self.observation], [sibling], "part-a", "source-a",
                           self.footprint, 2., self.samples)
        self.assertEqual(specs, [])

    def test_exact_part_wins_over_unmatched_fallback(self):
        fallback = {**self.profile, "id": "regen-source-a",
                    "match": {"featureId": "source-a"}, "planeV": 200.}
        specs, _ = resolve([self.observation], [fallback, self.profile], "part-a", "source-a",
                           self.footprint, 2., self.samples)
        self.assertEqual(len(specs), 1)
        self.assertAlmostEqual(specs[0]["xy"][1], 105.)

    def test_missing_spatial_support_is_audited_and_rejected(self):
        specs, audit = resolve([self.observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., np.array([[18.5, 105., 12.]]))
        self.assertEqual(specs, [])
        self.assertEqual(audit["status"], "unresolved")
        self.assertEqual(audit["rejected"][0]["reason"], "no-inset-roof-support")

    def test_low_confidence_feature_is_not_emitted(self):
        observation = {**self.observation, "features": [{**self.observation["features"][0],
                                                          "confidence": "low"}]}
        specs, audit = resolve([observation], [self.profile], "part-a", "source-a",
                               self.footprint, 2., self.samples)
        self.assertEqual(specs, [])
        self.assertEqual(audit["rejected"][0]["reason"], "low-feature-confidence")


if __name__ == "__main__":
    unittest.main()
