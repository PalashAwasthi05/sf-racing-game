import unittest

import numpy as np

from package_drive_stream import weld_support


class SupportWeldTest(unittest.TestCase):
    def test_curb_folds_and_paint_uvs_keep_distinct_corners(self):
        # Two level triangles with a shared edge, then a vertical curb face.
        p = np.array([(0,0,0),(1,0,0),(0,0,1), (1,0,0),(1,0,1),(0,0,1),
                      (0,0,0),(0,1,0),(1,0,0)], dtype='<f4')
        uv = p[:, [0,2]].copy()
        uv[5] += 1
        source = {'POSITION': p, 'TEXCOORD_0': uv}
        first, index = weld_support(source, np.arange(len(p)))
        self.assertEqual(index[1], index[3])
        self.assertNotEqual(index[2], index[5])
        self.assertNotEqual(index[0], index[6])
        for values in source.values():
            np.testing.assert_array_equal(values[first][index], values)
