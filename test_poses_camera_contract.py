import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from areas.ring.poses import matching_perspective, pose_source_views, resolve_camera_state, side_camera_geometry, vfov_for_image
from frontage_crops_v15 import view_basis
from regen.register_cameras import _rz, _tilt


class PoseCameraContractTests(unittest.TestCase):
    def test_registered_mode_uses_replacement_camera_from_registered_side(self):
        nominal = {
            'id': 'side-a', 'lengthM': 8.0, 'heightM': 6.0, 'frameU': [0.0, 8.0],
            'normal': [1.0, 0.0], 'planeV': 0.0, 'groundZ': 1.0,
            'views': [{'image': 'camera-a', 'rect': 'nominal-r.jpg', 'persp': 'nominal-p.jpg'}],
        }
        registered_piece = {
            **nominal, 'groundZ': 1.2,
            'views': [{'image': 'camera-b', 'registered': 'skyline', 'rect': 'registered-r.jpg',
                       'persp': 'registered-p.jpg'}],
        }
        selected = pose_source_views(nominal, {'side-a': {'status': 'ok', 'path': Path('sides-reg.json'), 'piece': registered_piece}}, True)

        self.assertEqual(selected['status'], 'ok')
        self.assertEqual(selected['piece']['groundZ'], 1.2)
        self.assertEqual(selected['views'][0]['image'], 'camera-b')
        self.assertEqual(selected['views'][0]['persp'], 'registered-p.jpg')
        self.assertEqual(selected['views'][0]['_sourceViewIndex'], 0)

    def test_registered_mode_reports_missing_or_ambiguous_side_metadata(self):
        nominal = {
            'id': 'side-a', 'lengthM': 8.0, 'heightM': 6.0, 'frameU': [0.0, 8.0],
            'normal': [1.0, 0.0], 'planeV': 0.0,
            'views': [{'image': 'camera-a', 'rect': 'nominal-r.jpg', 'persp': 'nominal-p.jpg'}],
        }
        missing = pose_source_views(nominal, {}, True)
        ambiguous = pose_source_views(nominal, {'side-a': {'status': 'ambiguous', 'candidates': ['a.json', 'b.json']}}, True)
        bad_metadata = pose_source_views(nominal, {'side-a': {'status': 'ok', 'path': Path('sides-reg.json'), 'piece': {**nominal, 'views': [{'image': 'camera-b'}]}}}, True)

        self.assertEqual(missing['status'], 'unresolved')
        self.assertEqual(missing['reason'], 'registered side record missing')
        self.assertEqual(ambiguous['status'], 'unresolved')
        self.assertEqual(ambiguous['reason'], 'registered side record ambiguous')
        self.assertEqual(bad_metadata['status'], 'unresolved')
        self.assertIn('rect, persp filename', bad_metadata['reason'])

    def test_registered_mode_keeps_partial_view_failures_explicit(self):
        nominal = {
            'id': 'side-a', 'lengthM': 8.0, 'heightM': 6.0, 'frameU': [0.0, 8.0],
            'normal': [1.0, 0.0], 'planeV': 0.0,
            'views': [],
        }
        registered_piece = {
            **nominal,
            'views': [
                {'image': 'camera-b', 'registered': 'skyline', 'rect': 'r.jpg', 'persp': 'p.jpg'},
                {'image': 'camera-c', 'registered': 'skyline', 'rect': 'r2.jpg'},
            ],
        }
        selected = pose_source_views(nominal, {'side-a': {'status': 'ok', 'path': Path('sides-reg.json'), 'piece': registered_piece}}, True)

        self.assertEqual(selected['status'], 'ok')
        self.assertEqual([v['image'] for v in selected['views']], ['camera-b'])
        self.assertEqual(selected['unresolvedViews'], [{'view': 1, 'reason': 'registered crop missing persp filename'}])

    def test_registered_camera_marker_must_match_accepted_camera_state(self):
        camera = {'xy': [10.0, 20.0]}
        skyline = {'image': 'camera-b', 'registered': 'skyline'}
        missing_skyline = resolve_camera_state(skyline, camera, {}, None, 1.7)
        self.assertEqual(missing_skyline['status'], 'unresolved')
        self.assertIn('accepted camera registration is missing', missing_skyline['reason'])

        accepted = {'camera-b': {'xy': [11.0, 21.0], 'cameraHeightM': 2.0}}
        bias_marker = resolve_camera_state({'image': 'camera-b', 'registered': 'bias'}, camera, accepted, np.array([1.0, 1.0]), 1.7)
        nominal_marker = resolve_camera_state({'image': 'camera-b', 'registered': False}, camera, accepted, None, 1.7)
        self.assertEqual(bias_marker['status'], 'unresolved')
        self.assertIn('accepted skyline registration', bias_marker['reason'])
        self.assertEqual(nominal_marker['status'], 'unresolved')
        self.assertIn('accepted skyline registration', nominal_marker['reason'])

        bias = resolve_camera_state({'image': 'camera-c', 'registered': 'bias'}, {'xy': [5.0, 7.0]}, {}, np.array([1.0, -2.0]), 1.7)
        np.testing.assert_allclose(bias['xy'], [6.0, 5.0])

    def test_registered_pose_uses_same_side_camera_registered_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            area = Path(tmp)
            nominal = area / 'regen/sides/img/side__v0_persp.jpg'
            registered = area / 'regen/sides-reg/img/side__v1_persp.jpg'
            nominal.parent.mkdir(parents=True)
            registered.parent.mkdir(parents=True)
            nominal.touch()
            registered.touch()
            source = {'persp': nominal.name}
            match = {'persp': registered.name, 'registered': 'skyline'}
            result, view = matching_perspective(
                area, 'building__s0p0', 'camera-1', source,
                {('building__s0p0', 'camera-1'): match}, True,
            )
            self.assertEqual(result, registered)
            self.assertIs(view, match)

    def test_registered_pose_fails_instead_of_using_nominal_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            area = Path(tmp)
            nominal = area / 'regen/sides/img/side__v0_persp.jpg'
            nominal.parent.mkdir(parents=True)
            nominal.touch()
            with self.assertRaisesRegex(RuntimeError, 'no registered perspective metadata'):
                matching_perspective(area, 'building__s0p0', 'camera-1',
                                     {'persp': nominal.name}, {}, True)

    def test_explicit_nominal_placement_keeps_crop_from_registered_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            area = Path(tmp)
            registered = area / 'regen/sides-reg/img/side__v0_persp.jpg'
            registered.parent.mkdir(parents=True)
            registered.touch()
            view = {'image': 'camera-a', 'registered': False, 'persp': registered.name}
            state = resolve_camera_state(view, {'xy': [1., 2.]}, {}, None, 1.7)
            self.assertEqual(state['status'], 'ok')
            actual, _ = matching_perspective(area, 'side', 'camera-a', {},
                                             {('side', 'camera-a'): view}, True)
            self.assertEqual(actual, registered)

    def test_registered_pose_fails_if_registered_crop_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = {'persp': 'side__v1_persp.jpg', 'registered': 'skyline'}
            with self.assertRaisesRegex(FileNotFoundError, 'pose image does not exist'):
                matching_perspective(tmp, 'building__s0p0', 'camera-1', {},
                                     {('building__s0p0', 'camera-1'): match}, True)

    def test_vertical_fov_tracks_actual_image_aspect(self):
        vfov = vfov_for_image(70.0, 1000, 913)
        expected = math.degrees(2 * math.atan(math.tan(math.radians(35.0)) * 0.913))
        self.assertAlmostEqual(vfov, expected)
        self.assertLess(vfov, 70.0)

    def test_registered_crop_uses_world_forward_separately_from_pano_rotation(self):
        # sides.py passes vp['forward'] unchanged to render_perspective; the corrected R
        # only changes how world rays map back into panorama coordinates.
        forward = np.array([0.25, 0.92, 0.30], dtype=float)
        forward /= np.linalg.norm(forward)
        base_rotation = np.eye(3)
        corrected_rotation = base_rotation @ _rz(-np.radians(4.0)) @ _tilt(0.01, -0.02)
        _, _, crop_center_world_ray = view_basis(forward)
        nominal_pano_ray = forward @ base_rotation.T
        registered_pano_ray = forward @ corrected_rotation.T
        np.testing.assert_allclose(crop_center_world_ray, forward)
        self.assertFalse(np.allclose(nominal_pano_ray, registered_pano_ray))

    def test_registered_side_metadata_supplies_exact_wall_frame(self):
        registered_side = {
            'normal': [0.0, 1.0], 'planeV': 12.0, 'frameU': [3.0, 9.0],
            'lengthM': 6.0, 'heightM': 8.0, 'groundZ': 1.25,
        }
        geom = side_camera_geometry(registered_side)
        np.testing.assert_allclose(geom['midpoint'], [-6.0, 12.0])
        self.assertEqual(geom['lengthM'], 6.0)
        self.assertEqual(geom['heightM'], 8.0)


if __name__ == '__main__':
    unittest.main()
