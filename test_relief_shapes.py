import unittest

from shapely.geometry import box

from pipeline import facade_detail


class ReliefShapeTests(unittest.TestCase):
    def context(self, clip=None):
        prisms, beams = [], []
        def prism(poly, depth, thickness, material):
            prisms.append((poly, depth, thickness, material))
        def beam(a, b, thickness, material):
            beams.append((a, b, thickness, material))
        ctx = {
            'block': lambda *args, **kwargs: None,
            'prism': prism,
            'beam': beam,
            'clip': clip if clip is not None else box(0., 0., 10., 10.),
            'low': 0., 'high': 10., 'W': 10., 'Hs': 10., 'H': 10.,
            'mat': 'stone', 'wall_mat': lambda x, z: 'stone',
        }
        return ctx, prisms, beams

    def test_roundel_uses_circular_outline_and_rosette_instead_of_a_solid_rectangle(self):
        ctx, prisms, _ = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.2, .3, .8, .7], 'shape': 'roundel', 'material': 'stone'}]}, [])
        self.assertGreaterEqual(len(prisms), 2)
        ring = prisms[0][0]
        self.assertLess(ring.area, box(2., 3., 8., 7.).area * .8)
        self.assertAlmostEqual(ring.bounds[0], 3.16, places=2)
        self.assertGreater(prisms[1][0].area, 0.)  # center rosette

    def test_roundel_repeat_count_creates_cell_sized_roundels(self):
        ctx, prisms, _ = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [0., .4, .975, .5], 'shape': 'roundel', 'repeatCount': 16}]}, [])
        rings = [poly for poly, *_ in prisms[::2]]
        self.assertEqual(len(rings), 16)
        for ring in rings:
            x0, z0, x1, z1 = ring.bounds
            self.assertLessEqual(x1 - x0, z1 - z0 + 1e-8)
        centers = [(poly.bounds[0] + poly.bounds[2]) / 2 for poly in rings]
        self.assertTrue(all(b > a for a, b in zip(centers, centers[1:])))

    def test_ring_count_adds_concentric_nested_bands(self):
        ctx, prisms, _ = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.1, .2, .9, .8], 'shape': 'roundel',
                                               'repeatCount': 2, 'ringCount': 3}]}, [])
        self.assertEqual(len(prisms), 8)  # 3 rings plus center disk for each medallion
        self.assertGreater(prisms[0][0].area, prisms[1][0].area)
        self.assertGreater(prisms[1][0].area, prisms[2][0].area)

    def test_cartouche_cardinal_flourishes_remain_inside_authored_rectangle(self):
        ctx, prisms, _ = self.context()
        rect = box(2., 3., 8., 7.)
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.2, .3, .8, .7], 'shape': 'cartouche'}]}, [])
        self.assertGreaterEqual(len(prisms), 6)  # border, center, and four leaves
        self.assertTrue(all(rect.covers(poly) for poly, *_ in prisms))

    def test_typed_circle_outline_is_independent_of_legacy_cartouche_and_flourishes(self):
        ctx, prisms, _ = self.context()
        authored = box(2., 3., 8., 7.)
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.2, .3, .8, .7], 'shape': 'cartouche',
                                               'outerOutline': 'circle', 'motif': 'unknown',
                                               'cardinalFlourishes': True}]}, [])
        ring = prisms[0][0]
        self.assertAlmostEqual(ring.bounds[2] - ring.bounds[0], ring.bounds[3] - ring.bounds[1], places=6)
        self.assertGreaterEqual(len(prisms), 5)  # frame plus four explicitly requested flourishes
        self.assertTrue(all(authored.covers(poly) for poly, *_ in prisms))

    def test_typed_unknown_motif_does_not_invent_center(self):
        ctx, prisms, _ = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.3, .3, .7, .7], 'shape': 'roundel',
                                               'outerOutline': 'circle', 'motif': 'unknown',
                                               'cardinalFlourishes': False}]}, [])
        self.assertEqual(len(prisms), 1)  # only the observed outline ring

    def test_typed_radial_motif_has_a_center_and_radial_ridges(self):
        ctx, prisms, _ = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.3, .3, .7, .7], 'shape': 'roundel',
                                               'outerOutline': 'circle', 'motif': 'radial',
                                               'cardinalFlourishes': False}]}, [])
        self.assertEqual(len(prisms), 10)  # ring, center disk, and eight narrow radial ridges
        self.assertTrue(all(poly.area > 0 for poly, *_ in prisms))

    def test_garland_arcs_repeat_inside_x_and_z_bounds(self):
        ctx, _, beams = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.2, .3, .8, .6], 'shape': 'garland',
                                               'xRange': [.2, .8], 'repeatCount': 3, 'curveDepth': .5}]}, [])
        self.assertEqual(len(beams), 24)
        points = [p for a, b, *_ in beams for p in (a, b)]
        self.assertTrue(all(2. - 1e-8 <= p[0] <= 8. + 1e-8 for p in points))
        self.assertTrue(all(3. - 1e-8 <= p[1] <= 6. + 1e-8 for p in points))
        self.assertAlmostEqual(min(p[1] for p in points), 4.5)

    def test_garland_repeat_phase_is_stable_when_wall_span_is_clipped(self):
        profile = {'reliefs': [{'rect': [0., .3, 1., .6], 'shape': 'garland', 'xRange': [0., 1.],
                                'repeatCount': 4, 'curveDepth': .5}]}
        full_ctx, _, full = self.context(box(0., 0., 20., 10.))
        facade_detail.wall(full_ctx, profile, [])
        left_ctx, _, left = self.context(box(0., 0., 10., 10.))
        right_ctx, _, right = self.context(box(10., 0., 20., 10.))
        facade_detail.wall(left_ctx, profile, [])
        facade_detail.wall(right_ctx, profile, [])
        def endpoints(items):
            return sorted((tuple(round(v, 6) for v in a), tuple(round(v, 6) for v in b)) for a, b, *_ in items)
        self.assertEqual(endpoints(full), endpoints(left + right))

    def test_garland_beam_follows_angled_band_support(self):
        from pipeline.facade_bands import band_footprint, support_path, support_depth_at

        path = support_path(0., 10., 1., [{"x0": 3., "x1": 7., "z0": 0., "z1": 2.,
                                           "p": 1., "shape": "angled", "sideAngleDeg": 30.}])
        footprint = band_footprint(path, .1, .2)
        ctx, _, _ = self.context()
        beams = []
        ctx['reliefSupports'] = [{'rect': box(0., 0., 10., 2.), 'frontDepth': .1,
                                  'planFootprint': footprint}]
        ctx['beam'] = lambda a, b, width, material: beams.append((a, b, width, material))
        facade_detail._relief_beam(ctx, (2., 1., 0.), (8., 1., 0.), .04, 'stone', .05)

        self.assertGreater(len(beams), 2)  # footprint vertices split the sloped/angled support
        for a, b, *_ in beams:
            self.assertAlmostEqual(a[2], support_depth_at(footprint, a[0]) + .05, places=5)
            self.assertAlmostEqual(b[2], support_depth_at(footprint, b[0]) + .05, places=5)
        self.assertGreater(max(end[2] for beam in beams for end in beam[:2]),
                           min(end[2] for beam in beams for end in beam[:2]))

    def test_garland_beam_does_not_bridge_an_abrupt_support_step(self):
        ctx, _, _ = self.context()
        beams = []
        ctx['reliefSupports'] = [
            {'rect': box(0., 0., 5., 2.), 'frontDepth': .3, 'planFootprint': box(0., .25, 5., .35)},
            {'rect': box(5., 0., 10., 2.), 'frontDepth': .8, 'planFootprint': box(5., .75, 10., .85)},
        ]
        ctx['beam'] = lambda a, b, width, material: beams.append((a, b, width, material))
        facade_detail._relief_beam(ctx, (2., 1., 0.), (8., 1., 0.), .04, 'stone', .02)

        self.assertEqual(len(beams), 2)
        self.assertTrue(all(a[0] < 5. + 1e-8 and b[0] <= 5. + 1e-8 for a, b, *_ in beams[:1]))
        self.assertTrue(all(a[0] >= 5. - 1e-8 and b[0] > 5. for a, b, *_ in beams[1:]))
        self.assertAlmostEqual(beams[0][0][2], .37)
        self.assertAlmostEqual(beams[1][0][2], .87)

    def test_flat_garland_beam_keeps_legacy_depth(self):
        ctx, _, beams = self.context()
        facade_detail._relief_beam(ctx, (1., 1., 0.), (4., 2., 0.), .04, 'stone', .07)
        self.assertEqual(len(beams), 1)
        self.assertAlmostEqual(beams[0][0][2], .07)
        self.assertAlmostEqual(beams[0][1][2], .07)

    def test_unspecified_shape_keeps_legacy_rectangular_panel(self):
        ctx, prisms, _ = self.context()
        facade_detail.wall(ctx, {'reliefs': [{'rect': [.2, .3, .8, .7]}]}, [])
        self.assertEqual(prisms[0][0].bounds, (2., 3., 8., 7.))

    def test_relief_mesh_projects_beyond_the_actual_band_face(self):
        """Relief meshes overlap their backing band by a small amount and remain visibly proud of its front face."""
        import numpy as np
        import trimesh
        from pipeline.facade_grammar import compile_wall

        emitted = []
        def cube(center, size, key, material, angle=0.):
            mesh = trimesh.creation.box(extents=size)
            mesh.apply_translation(center)
            emitted.append((key, material, mesh))
        def put(mesh, key, material): emitted.append((key, material, mesh.copy()))
        ctx = {'polys': lambda g: [g], 'cube': cube, 'put': put,
               'cylinder': lambda *args, **kwargs: None}
        frontage = box(0., 0., 10., 8.)
        profile = {
            'id': 'relief-clearance', 'normal': [0., -1.], 'sourceIds': ['test'], 'material': 'stucco',
            'bands': [{'height': .5, 'thicknessM': .4, 'depthM': .3, 'projectionM': .1, 'material': 'stone'}],
            'reliefs': [{'rect': [.2, .475, .8, .525], 'shape': 'panel', 'projectionM': .04, 'material': 'ivory'}],
        }
        result = compile_wall(ctx, {'g': frontage, 'id': 'relief-clearance'}, np.array([0., 0.]),
                              np.array([10., 0.]), 0., 8., profile, [])
        self.assertIsNotNone(result)
        band_meshes = [mesh for key, material, mesh in emitted if material == 'stone']
        relief_meshes = [mesh for key, material, mesh in emitted if material == 'ivory']
        self.assertTrue(band_meshes and relief_meshes)
        # Frontage faces south (-Y). The actual band box is centered at depth .10 with .3 m thickness,
        # so its outer face is at depth .25. The relief's front should sit .04 m beyond that backing.
        band_front = max(-float(mesh.vertices[:, 1].min()) for mesh in band_meshes)
        relief_front = max(-float(mesh.vertices[:, 1].min()) for mesh in relief_meshes)
        self.assertAlmostEqual(band_front, .25, places=5)
        self.assertGreaterEqual(relief_front, band_front + .039)


if __name__ == '__main__':
    unittest.main()
