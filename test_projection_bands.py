"""Synthetic-only regression coverage for bounded projection grids."""
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import map_coordinates

import frontage_crops_v15 as fc
import auto_read_frontages_v24 as ar


def eager_rectify(wall, cam, z0, pano, base=0.):
    m = np.asarray(wall['midpoint'], float)
    n = np.asarray(wall['normal'], float)
    n /= np.linalg.norm(n)
    t = np.array([-n[1], n[0]])
    L, H = float(wall['lengthM']), float(wall['heightM'])
    W, Hp = max(8, int(L*ar.PPM)), max(8, int(H*ar.PPM))
    xs = (np.arange(W)+.5)/ar.PPM-L/2
    zs = H-(np.arange(Hp)+.5)/ar.PPM
    P = np.empty((Hp, W, 3))
    P[..., 0] = (m[0]+t[0]*xs)[None, :]+n[0]*.05
    P[..., 1] = (m[1]+t[1]*xs)[None, :]+n[1]*.05
    P[..., 2] = (z0+float(base)+zs)[:, None]
    C = np.array([cam['xy'][0], cam['xy'][1],
                  (cam['groundZ'] if cam.get('groundZ') is not None else z0)
                  +fc.EYE_HEIGHT_M+float(cam.get('heightOffsetM', 0))])
    u, v = fc.world_dirs_to_pano_uv(P-C, cam['R'])
    ph, pw = pano.shape[:2]
    img = np.stack([map_coordinates(pano[..., k], [v*(ph-1), u*(pw-1)],
                                   order=1, mode='wrap', output=np.float32)
                    for k in range(3)], -1)
    return img/255., H, L


def eager_perspective(pano, R, forward, hfov_deg, vfov_deg, width, max_height, mirror_u=False):
    right, down, f = fc.view_basis(forward)
    fx = (width/2.)/math.tan(math.radians(hfov_deg)/2.)
    height = int(round(2.*fx*math.tan(math.radians(vfov_deg)/2.)))
    if height > max_height:
        height = max_height
        vfov_deg = math.degrees(2.*math.atan((height/2.)/fx))
    xs = (np.arange(width)+.5-width/2.)/fx
    ys = (np.arange(height)+.5-height/2.)/fx
    dirs = right[None, None, :]*xs[None, :, None]+down[None, None, :]*ys[:, None, None]+f[None, None, :]
    u, v = fc.world_dirs_to_pano_uv(dirs, R, mirror_u)
    ph, pw = pano.shape[:2]
    px = (u*pw-.5) % pw
    py = np.clip(v*ph-.5, 0, ph-1)
    out = np.empty((height, width, 3), dtype=np.uint8)
    coords = np.stack([py, px])
    for c in range(3):
        out[..., c] = map_coordinates(pano[..., c], coords, order=1, mode='grid-wrap').astype(np.uint8)
    return out, vfov_deg


class ProjectionBandsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pano = np.random.default_rng(42).integers(0, 256, (47, 93, 3), dtype=np.uint8)

    def assert_banded(self, render, expected):
        shapes = []
        original = fc.world_dirs_to_pano_uv
        def record(dirs, *args):
            shapes.append(dirs.shape)
            return original(dirs, *args)
        with patch.object(fc, 'world_dirs_to_pano_uv', side_effect=record):
            actual = render()
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[0].dtype, expected[0].dtype)
        self.assertEqual(actual[1:], expected[1:])
        self.assertTrue(shapes)
        self.assertLessEqual(max(s[0] for s in shapes), 32)
        self.assertEqual(sum(s[0] for s in shapes), actual[0].shape[0])

    def test_rectify_offsets_and_partial_bands(self):
        for height, ground, offset, base in [(0.4, None, 0., 0.), (5.7, 8., 1.8, 4.), (17.1, None, -1., 2.)]:
            wall = dict(midpoint=[1., -4.], normal=[.3, .9], lengthM=24.3, heightM=height)
            cam = dict(xy=[1., 3.], groundZ=ground, heightOffsetM=offset,
                       R=fc.Rotation.from_rotvec([.3, -.8, .7]).as_matrix())
            args = wall, cam, 3., self.pano, base
            self.assert_banded(lambda: ar.rectify(*args), eager_rectify(*args))

    def test_perspective_seam_mirror_height_cap_and_large_crop(self):
        # -z crosses the equirectangular seam; non-multiple heights cover the final band.
        for width, cap, forward, mirror in [(17, 17, [0., 0., -1.], False),
                                           (131, 99, [.2, 1., -.4], True),
                                           (769, 401, [0., 0., -1.], True)]:
            args = self.pano, np.eye(3), np.array(forward), 100., 90., width, cap, mirror
            self.assert_banded(lambda: fc.render_perspective(*args), eager_perspective(*args))


if __name__ == '__main__':
    unittest.main()
