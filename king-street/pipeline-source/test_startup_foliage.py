from copy import deepcopy
import unittest
import numpy as np
from startup_foliage import compact_foliage


class StartupFoliageTest(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [.5, 0, .5]], dtype='<f4')
        self.fan = np.array([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype='<u2')
        self.model = {
            'nodes': [{'name': 'vegetation___leaf1', 'mesh': 0}],
            'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],
            'accessors': [
                {'type': 'VEC3', 'componentType': 5126, 'count': 5, 'bufferView': 0},
                {'type': 'SCALAR', 'componentType': 5123, 'count': 12, 'bufferView': 1}],
            'bufferViews': [
                {'buffer': 0, 'byteLength': self.positions.nbytes},
                {'buffer': 0, 'byteLength': self.fan.nbytes, 'byteOffset': self.positions.nbytes}]}

    def test_verified_fan_keeps_its_exact_centre(self):
        output = compact_foliage(self.model, self.positions.tobytes() + self.fan.tobytes())
        primitive = self.model['meshes'][0]['primitives'][0]
        view = self.model['accessors'][primitive['attributes']['POSITION']]['bufferView']
        self.assertEqual(output[view], self.positions[-1].tobytes())
        self.assertTrue(self.model['nodes'][0]['extras']['foliageFanCentres'])

    def test_nonfan_and_nonvegetation_keep_their_original_geometry(self):
        original = deepcopy(self.model)
        self.fan[5] = 0
        self.assertEqual(compact_foliage(self.model, self.positions.tobytes() + self.fan.tobytes()), {})
        self.assertEqual(self.model, original)
        self.fan[5] = 4
        self.model['nodes'][0]['extras'] = {'layer': 'buildings'}
        original = deepcopy(self.model)
        self.assertEqual(compact_foliage(self.model, self.positions.tobytes() + self.fan.tobytes()), {})
        self.assertEqual(self.model, original)
