import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from winding_corrections import (build_support_edges, _support_match,
    _triangle_overlap_area, _filter_overlapping, _closed_component_mask,
    _correction_map, TriangleBVH, _candidate_visible, build_candidate)


class WindingCorrectionGeometryTests(unittest.TestCase):
    def setUp(self):
        square = [[0, 0], [4, 0], [4, 4], [0, 4]]
        # The exposed west side of a CCW x/z square points west.
        self.support = build_support_edges(square, [([0, 4], [0, 0])])
        self.west = next(edge for edge in self.support if edge['outward'][0] < -0.9)

    def test_detects_only_the_inward_front_side_on_an_exterior_support(self):
        outward = np.array([[0, 0, 0], [0, 0, 2], [0, 2, 2]], dtype=float)
        inward = outward[[0, 2, 1]]
        outward_match = _support_match(outward, np.array([-1., 0, 0]), self.support)
        inward_match = _support_match(inward, np.array([1., 0, 0]), self.support)
        self.assertIsNotNone(outward_match)
        self.assertIsNotNone(inward_match)
        self.assertGreater(outward_match[1], .9)
        self.assertLess(inward_match[1], -.9)

    def test_concavity_spans_are_not_accepted_as_actual_boundary(self):
        support = build_support_edges([[0, 0], [4, 0], [4, 4], [0, 4]],
                                      [([0, 4], [0, 3]), ([0, 1], [0, 0])])
        face = np.array([[0, 0, 1.2], [0, 0, 2.8], [0, 2, 2.8]], dtype=float)
        self.assertIsNone(_support_match(face, np.array([1., 0, 0]), support))

    def test_overlap_detects_different_triangulations_and_blocks_both(self):
        # One large inward triangle overlaps the union of two smaller outward
        # triangles. This is the common case where index triples differ.
        a = {'edge_id': ('owner', 0), 'plane_offset': 0., 'side': -1.,
             'polygon': np.array([[0, 0], [2, 0], [0, 2]], dtype=float),
             'bbox': (0, 0, 2, 2), 'eligible': True}
        b = {'edge_id': ('owner', 0), 'plane_offset': 0., 'side': 1.,
             'polygon': np.array([[0, 0], [2, 0], [2, 2]], dtype=float),
             'bbox': (0, 0, 2, 2)}
        c = {'edge_id': ('owner', 0), 'plane_offset': 0., 'side': 1.,
             'polygon': np.array([[0, 0], [2, 2], [0, 2]], dtype=float),
             'bbox': (0, 0, 2, 2)}
        self.assertAlmostEqual(_triangle_overlap_area(a['polygon'], b['polygon']), 1.)
        self.assertEqual(_filter_overlapping([a, b, c]), {0})

    def test_small_coplanar_outward_fragment_blocks_a_large_wrong_face(self):
        wrong = {'edge_id': ('owner', 1), 'plane_offset': 0., 'side': -1.,
                 'polygon': np.array([[0, 0], [2, 0], [0, 2]], dtype=float),
                 'bbox': (0, 0, 2, 2), 'eligible': True}
        blocker = {'edge_id': ('owner', 1), 'plane_offset': .001, 'side': 1.,
                   'polygon': np.array([[.2, .2], [.4, .2], [.2, .4]], dtype=float),
                   'bbox': (.2, .2, .4, .4), 'eligible': False}
        self.assertEqual(_filter_overlapping([wrong, blocker]), {0})

    def test_offset_outward_face_blocks_the_back_of_a_thin_wall_panel(self):
        wrong = {'edge_id': ('owner', 1), 'plane_offset': 0., 'side': -1.,
                 'polygon': np.array([[0, 0], [2, 0], [0, 2]], dtype=float),
                 'bbox': (0, 0, 2, 2), 'eligible': True}
        front = {'edge_id': ('owner', 1), 'plane_offset': .08, 'side': 1.,
                 'polygon': np.array([[0, 0], [2, 0], [0, 2]], dtype=float),
                 'bbox': (0, 0, 2, 2), 'eligible': False}
        self.assertEqual(_filter_overlapping([wrong, front]), {0})

    def test_unwelded_uv_seam_cuboid_is_a_closed_shell(self):
        # A thin box whose twelve triangles do not share vertex indices.
        v = [np.array(p, dtype=float) for p in [
            [0, 0, 0], [.08, 0, 0], [.08, 2, 0], [0, 2, 0],
            [0, 0, 4], [.08, 0, 4], [.08, 2, 4], [0, 2, 4]]]
        faces = [
            (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
            (0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2)]
        triangles = np.array([[v[i], v[j], v[k]] for i, j, k in faces])
        closed = _closed_component_mask(triangles)
        self.assertTrue(closed.all())
        candidate = {'mesh': 4, 'primitive': 0, 'triangle': 0, 'side': -1., 'eligible': True}
        corrections = _correction_map([candidate], set(), {(4, 0, 0)})
        self.assertEqual(dict(corrections), {})

    def test_genuinely_open_reversed_wall_remains_eligible(self):
        wall = np.array([[0, 0, 0], [0, 2, 0], [0, 2, 2]], dtype=float)
        match = _support_match(wall[[0, 2, 1]], np.array([1., 0, 0]), self.support)
        self.assertIsNotNone(match)
        candidate = {'mesh': 7, 'primitive': 0, 'triangle': 3,
                     'side': match[1], 'eligible': True}
        corrections = _correction_map([candidate], set(), set())
        self.assertEqual(corrections[(7, 0)], {3})

    def test_open_reversed_wall_has_exterior_line_of_sight(self):
        points = np.array([[0, 0, 0], [0, 2, 0], [0, 2, 2]], dtype=float)
        candidate = {'owner': 'open', 'mesh': 2, 'primitive': 0, 'triangle': 0,
                     'points': points, 'outward': np.array([-1., 0.])}
        bvh = TriangleBVH([points], [False], [(2, 0, 0)], owner='open')
        self.assertTrue(_candidate_visible(candidate, [bvh]))

    def test_same_owner_outward_panel_obstructs_the_back_surface(self):
        back = np.array([[0, 0, 0], [0, 2, 0], [0, 2, 2]], dtype=float)
        candidate = {'owner': 'panel', 'mesh': 2, 'primitive': 0, 'triangle': 0,
                     'points': back, 'outward': np.array([-1., 0.])}
        front = np.array([[[-.08, 0, 0], [-.08, 0, 2], [-.08, 2, 2]],
                          [[-.08, 0, 0], [-.08, 2, 2], [-.08, 2, 0]]], dtype=float)
        bvh = TriangleBVH(front, [False, False], [(3, 0, 0), (3, 0, 1)], owner='panel')
        self.assertFalse(_candidate_visible(candidate, [bvh]))

    def test_empty_candidate_never_rewrites_or_adds_metadata_to_active_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            public = Path(td)
            compare = public/'compare'; compare.mkdir()
            active = compare/'neighborhood-stream.json'
            original = {'version': 1, 'scope': 'neighborhood-stream',
                        'sourceWorldHash': 'source-hash', 'cells': []}
            active.write_text(json.dumps(original, sort_keys=True))
            before = active.read_bytes()
            empty = {'version': 1, 'scope': 'neighborhood-winding-corrections',
                     'sourceWorldHash': 'source-hash',
                     'cells': [{'id': 'grid-0-0', 'contentHash': 'cell-hash', 'records': []}]}
            with patch('winding_corrections.generate', return_value=(empty, {'cells': 1})):
                result = build_candidate(public)
            self.assertIsNone(result['windingCorrections'])
            self.assertIsNone(result['candidateManifest'])
            self.assertEqual(active.read_bytes(), before)
            self.assertEqual([p.name for p in compare.iterdir()], ['neighborhood-stream.json'])


if __name__ == '__main__':
    unittest.main()
