import unittest

import numpy as np
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

from pipeline.roof_features import build, _project_front_wall, _front_evidence_meshes


class RoofFeatureComponentTests(unittest.TestCase):
    def setUp(self):
        self.part = box(0, 0, 20, 20)

    def dormer(self, **extra):
        return {"kind": "dormer", "xy": [10, 10], "facing": [1, 0], "widthM": 3,
                "depthM": 3, "eaveM": 8, "ridgeM": 9, "roof": "gable", **extra}

    def test_observed_window_count_produces_separate_glass_panels(self):
        for count in range(5):
            with self.subTest(count=count):
                parts = build([self.dormer(windowCount=count, bottomM=4)], self.part, 0, 7.7, [])
                self.assertEqual(sum(role == "glass" for _, role in parts), count)
                if count == 0:
                    self.assertFalse(any(role == "trim" for _, role in parts))  # no invented vent blades

    def test_legacy_window_defaults_and_false_flag_are_preserved(self):
        self.assertEqual(sum(role == "glass" for _, role in build([self.dormer()], self.part, 0, 5, [])), 1)
        no_window = build([self.dormer(window=False)], self.part, 0, 5, [])
        self.assertFalse(any(role in ("glass", "trim") for _, role in no_window))

    def test_bottom_anchor_fixes_sill_and_prevents_slope_sliding(self):
        f = self.dormer(windowCount=1, bottomM=4)
        parts = build([f], self.part, 0, 7.7, [])
        glass = next(mesh for mesh, role in parts if role == "glass")
        self.assertAlmostEqual(float(glass.bounds[0, 2]), 4.08, places=5)
        # With a 4 m legacy search, this low clearance would otherwise move the dormer far down the roof.
        walls = [mesh for mesh, role in parts if role == "wall"]
        self.assertTrue(walls)
        self.assertLess(max(float(mesh.bounds[1, 0]) for mesh in walls), 12.)

    def test_material_roles_are_optional_and_legacy_roles_remain_defaults(self):
        defaults = build([self.dormer(windowCount=1, bottomM=4)], self.part, 0, 5, [])
        self.assertTrue({"roof", "wall", "trim", "glass"}.issubset({role for _, role in defaults}))
        custom = build([self.dormer(windowCount=1, bottomM=4, wallMaterial="stucco_blue",
                                    trimMaterial="cream_trim", roofMaterial="dark_slate")],
                       self.part, 0, 5, [])
        roles = {role for _, role in custom}
        self.assertTrue({"stucco_blue", "cream_trim", "dark_slate", "glass"}.issubset(roles))

    def test_source_front_outline_and_each_window_keep_measured_proportions(self):
        f = self.dormer(windowCount=1, bottomM=4, frontEvidence={
            "coordinateSpace": "normalized-front",
            "outline01": [[0, 0], [1, 0], [1, .62], [.52, 1], [0, .62]],
            "windows": [
                {"rect01": [.12, .16, .43, .50], "verticalMullions01": [.5]},
                {"rect01": [.58, .13, .84, .70], "horizontalMullions01": [.5]},
            ],
        })
        parts = build([f], self.part, 0, 7.7, [])
        glass = sorted((m for m, role in parts if role == "glass"), key=lambda m: m.bounds[0, 1])
        self.assertEqual(len(glass), 2)
        heights = sorted(float(m.bounds[1, 2] - m.bounds[0, 2]) for m in glass)
        self.assertGreater(heights[1] - heights[0], .8)
        self.assertGreaterEqual(sum(role == "trim" for _, role in parts), 4)
        # The measured front has a peaked outline, rather than a rectangular cap.
        wall_meshes = [m for m, role in parts if role == "wall"]
        self.assertTrue(any(np.any(np.isclose(m.vertices[:, 2], 9.)) for m in wall_meshes))

    def test_explicit_empty_front_glazing_suppresses_generic_window_count(self):
        f = self.dormer(windowCount=3, bottomM=4, frontEvidence={
            "coordinateSpace": "normalized-front",
            "outline01": [[0, 0], [1, 0], [1, .6], [.5, 1], [0, .6]],
            "windows": [],
        })
        parts = build([f], self.part, 0, 7.7, [])
        self.assertFalse(any(role == "glass" for _, role in parts))
        self.assertFalse(any(role == "trim" for _, role in parts))

    def test_invalid_explicit_evidence_is_unresolved_without_generic_glass(self):
        cases = [
            {"coordinateSpace": "normalized-front", "outline01": [[0, 0], [1, 0], [.5, 1]]},
            {"coordinateSpace": "normalized-front", "apertures": [{"kind": "window", "outline01":
                [[.2, .2], [.8, .8], [.2, .8], [.8, .2]]}]},
            {"coordinateSpace": "normalized-front", "apertures": [{"kind": "unknown", "rect01": [.2, .2, .8, .6]}]},
            {"coordinateSpace": "normalized-front", "apertures": [{"kind": "window", "rect01": [.2, .2, .8, 1]}]},
            {"coordinateSpace": "normalized-front", "apertures": [{"kind": "louver", "rect01": [.2, .2, .8, .6], "bladeCount": 0}]},
            {"coordinateSpace": "normalized-front", "apertures": [{"kind": "louver", "rect01": [.2, .2, .8, .6], "bladePitchM": float('nan')}]},
            {"coordinateSpace": "normalized-front", "apertures": [{"kind": "window", "rect01": [.2, .2, .8, .6], "frameWidthM": 5}]},
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                parts = build([self.dormer(windowCount=3, bottomM=4, frontEvidence=evidence)], self.part, 0, 7.7, [])
                self.assertTrue(parts)
                self.assertFalse(any(role in ('glass', 'trim') for _, role in parts))
                self.assertTrue(all(m.metadata['frontEvidenceStatus'] == 'unresolved' for m, _ in parts))
                self.assertTrue(all(m.metadata.get('frontEvidenceError') for m, _ in parts))

    def aperture(self, kind='window', **extra):
        return {"kind": kind, "outline01": [[.3, .2], [.7, .2], [.7, .55], [.65, .7],
                                           [.5, .8], [.35, .7], [.3, .55]], **extra}

    def elevation(self, mesh):
        # Facing +X, horizontal front coordinate is -world Y.
        pieces = []
        for triangle in mesh.triangles:
            poly = Polygon(np.c_[-triangle[:, 1] + 10, triangle[:, 2]])
            if poly.is_valid and poly.area > 1e-8:
                pieces.append(poly)
        return unary_union(pieces)

    def test_arched_aperture_cuts_opaque_front_and_clips_glass_and_mullions(self):
        aperture = self.aperture(verticalMullions01=[.05, .5, .95], horizontalMullions01=[.95])
        parts = build([self.dormer(windowCount=3, bottomM=4, depthM=4, frontEvidence={
            'coordinateSpace': 'normalized-front', 'apertures': [aperture]})], self.part, 0, 7.7, [])
        glass = [m for m, role in parts if role == 'glass']
        self.assertEqual(len(glass), 1)
        glazing = self.elevation(glass[0])
        aperture_metric = Polygon([[(x - .5) * 3, 4 + y * 5] for x, y in aperture['outline01']])
        self.assertLess(glazing.symmetric_difference(aperture_metric.buffer(-.055, join_style=2)).area, 1e-8)
        self.assertFalse(glazing.covers(Point(-.58, 7.85)))  # arch bounding-box corner stays empty
        for mesh, _ in parts:
            if mesh.metadata.get('frontEvidenceComponent') == 'mullion':
                self.assertLess(self.elevation(mesh).difference(glazing).area, 1e-8)
        opaque_front = []
        for mesh, role in parts:
            if role != 'wall':
                continue
            for triangle in mesh.triangles:
                if np.all(np.abs(triangle[:, 0] - 12.) < .1):
                    poly = Polygon(np.c_[-triangle[:, 1] + 10, triangle[:, 2]])
                    if poly.is_valid and poly.area > 1e-8:
                        opaque_front.append(poly)
        self.assertFalse(unary_union(opaque_front).covers(Point(0, 6)))

    def test_aperture_only_preserves_projected_generated_front_including_overhang(self):
        baseline = build([self.dormer(window=False, bottomM=4, depthM=4)], self.part, 0, 7.7, [])
        front = unary_union([_project_front_wall(m, np.array([10., 10.]), np.array([1., 0.]), 2.)
                             for m, role in baseline if role == 'wall'])
        self.assertGreater(front.bounds[2] - front.bounds[0], 3.)  # actual eave overhang, not a fabricated width box
        parts = build([self.dormer(windowCount=3, bottomM=4, depthM=4, frontEvidence={
            'coordinateSpace': 'normalized-front', 'apertures': []})], self.part, 0, 7.7, [])
        panels = [m for m, _ in parts if m.metadata.get('frontEvidenceComponent') == 'front-wall']
        self.assertEqual(len(panels), 1)
        self.assertLess(self.elevation(panels[0]).symmetric_difference(front).area, 1e-8)
        self.assertFalse(any(role in ('glass', 'trim') for _, role in parts))

    def test_louver_and_closed_apertures_are_nonglass_without_invented_blades(self):
        for kind in ('louver', 'closed'):
            with self.subTest(kind=kind):
                parts = build([self.dormer(windowCount=3, bottomM=4, depthM=4, frontEvidence={
                    'coordinateSpace': 'normalized-front', 'apertures': [self.aperture(kind)]})], self.part, 0, 7.7, [])
                self.assertFalse(any(role == 'glass' for _, role in parts))
                components = [m.metadata.get('frontEvidenceComponent') for m, _ in parts]
                self.assertEqual(components.count('nonglass-insert'), 1)
                self.assertEqual(components.count('aperture-frame'), 1)
                self.assertNotIn('louver-blade', components)

    def test_louver_blades_require_and_follow_explicit_count_or_pitch(self):
        for evidence, expected in [({'bladeCount': 7}, 7), ({'bladePitchM': .5}, 6)]:
            with self.subTest(evidence=evidence):
                parts = build([self.dormer(bottomM=4, depthM=4, frontEvidence={
                    'coordinateSpace': 'normalized-front', 'apertures': [self.aperture('louver', **evidence)]})],
                    self.part, 0, 7.7, [])
                self.assertFalse(any(role == 'glass' for _, role in parts))
                blades = [m for m, _ in parts if m.metadata.get('frontEvidenceComponent') == 'louver-blade']
                self.assertEqual(len(blades), expected)
                insert = next(m for m, _ in parts if m.metadata.get('frontEvidenceComponent') == 'nonglass-insert')
                for blade in blades:
                    self.assertLess(self.elevation(blade).difference(self.elevation(insert)).area, 1e-8)

    def test_square_dormer_gable_faces_explicit_direction_after_rotation(self):
        for angle in (0, 37, 123):
            with self.subTest(angle=angle):
                theta = np.radians(angle)
                face = np.array([np.cos(theta), np.sin(theta)])
                tangent = np.array([face[1], -face[0]])
                parts = build([self.dormer(facing=face.tolist(), window=False, bottomM=4)],
                              self.part, 0, 7.7, [])
                wall = next(m for m, role in parts if role == 'wall')
                front = _project_front_wall(wall, np.array([10., 10.]), face, 1.5)
                side = _project_front_wall(wall, np.array([10., 10.]), tangent, 1.65)
                self.assertIsNotNone(front)
                self.assertIsNotNone(side)
                self.assertAlmostEqual(front.bounds[3], 9., places=5)
                self.assertTrue(front.buffer(1e-7).covers(Point(0, 9)))
                self.assertLess(side.bounds[3], 8.01)
                # An explicit photographed outline remains authoritative over the generated front.
                evidence = {'coordinateSpace': 'normalized-front',
                            'outline01': [[0, 0], [1, 0], [1, .8], [.65, 1], [.35, 1], [0, .8]],
                            'apertures': []}
                source_parts = build([self.dormer(facing=face.tolist(), bottomM=4, frontEvidence=evidence)],
                                     self.part, 0, 7.7, [])
                panel = next(m for m, _ in source_parts if m.metadata.get('frontEvidenceComponent') == 'front-wall')
                transverse = (panel.vertices[:, :2] - [10, 10]) @ tangent
                top = transverse[np.isclose(panel.vertices[:, 2], 9)]
                self.assertAlmostEqual(float(top.max() - top.min()), .9, places=5)

    def test_frame_ring_inset_is_equal_in_metric_coordinates(self):
        parts = _front_evidence_meshes({'coordinateSpace': 'normalized-front',
            'outline01': [[0, 0], [1, 0], [1, 1], [0, 1]],
            'apertures': [{'kind': 'window', 'rect01': [.1, .1, .9, .9], 'frameWidthM': .07}]},
            np.array([0., 0.]), np.array([1., 0.]), 4., 0., 2., 'wall', 'trim')
        glass = next(m for m, role in parts if role == 'glass')
        self.assertAlmostEqual(np.ptp(glass.vertices[:, 1]), 3.2 - .14)
        self.assertAlmostEqual(np.ptp(glass.vertices[:, 2]), 1.6 - .14)

    def test_conical_cap_is_elliptical_closed_and_has_consistent_normals(self):
        f = {"kind": "conical-cap", "xy": [0, 10], "widthM": 4, "depthM": 2,
             "eaveM": 6, "ridgeM": 9}
        parts = build([f], box(0, 0, 20, 20), 2, 5, [])  # cap center is on the part boundary
        self.assertEqual([role for _, role in parts], ["roof", "trim"])
        roof = parts[0][0]
        self.assertEqual(len(roof.vertices), 34)  # 32-point oval, eave center and apex
        self.assertTrue(np.isfinite(roof.vertices).all())
        self.assertTrue(np.isfinite(roof.face_normals).all())
        self.assertTrue(roof.is_winding_consistent)
        self.assertTrue(roof.is_volume)
        self.assertAlmostEqual(float(roof.bounds[1, 2]), 11.)
        radial = np.sqrt(((roof.vertices[:32, 0] - 0) / 2) ** 2 + ((roof.vertices[:32, 1] - 10) / 1) ** 2)
        np.testing.assert_allclose(radial, np.ones(32), atol=1e-6)

    def test_conical_cap_may_overhang_part_with_center_outside(self):
        f = {"kind": "conical-cap", "xy": [-.5, 10], "widthM": 4, "depthM": 2,
             "eaveM": 6, "ridgeM": 9}
        parts = build([f], self.part, 0, 5, [])
        self.assertEqual(len(parts), 2)

    def test_conical_cap_width_tracks_front_tangent_and_depth_tracks_facing(self):
        f = {"kind": "conical-cap", "xy": [10, 10], "facing": [1, 0], "widthM": 4, "depthM": 2,
             "eaveM": 6, "ridgeM": 9}
        roof, trim = [mesh for mesh, _ in build([f], self.part, 0, 5, [])]
        self.assertAlmostEqual(float(roof.bounds[1, 0] - roof.bounds[0, 0]), 2., places=5)
        self.assertAlmostEqual(float(roof.bounds[1, 1] - roof.bounds[0, 1]), 4., places=5)
        np.testing.assert_allclose(trim.bounds[:, :2], roof.bounds[:, :2], atol=1e-6)

    def test_conical_cap_skips_when_ellipse_does_not_intersect_part(self):
        f = {"kind": "conical-cap", "xy": [-3, 10], "widthM": 4, "depthM": 2,
             "eaveM": 6, "ridgeM": 9}
        self.assertEqual(build([f], self.part, 0, 5, []), [])


if __name__ == "__main__":
    unittest.main()
