import copy
import io
import unittest

import numpy as np
from PIL import Image
from startup_tints import compact_tints


def fixture(pattern=False):
    binary = bytearray()
    model = {'bufferViews': [], 'images': [], 'textures': [], 'materials': [],
             'nodes': [], 'meshes': [], 'accessors': [{'count': 3, 'type': 'VEC3'}]}
    for i, rgb in enumerate([(160, 160, 160), (120, 128, 136)]):
        image = Image.new('RGB', (16, 16), rgb)
        if pattern:
            for y in range(16):
                for x in range(8): image.putpixel((x,y), (20, 20, 20))
        out = io.BytesIO(); image.save(out, format='PNG'); data = out.getvalue()
        model['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':len(data)})
        binary.extend(data)
        model['images'].append({'bufferView':i,'mimeType':'image/png'})
        model['textures'].append({'source':i})
        model['materials'].append({'name':f'stucco#{i}', 'pbrMetallicRoughness':{'baseColorTexture':{'index':i}}})
        model['meshes'].append({'primitives':[{'attributes':{'POSITION':0},'material':i}]})
        model['nodes'].append({'mesh':i,'name':f'buildings__{i}___stucco#{i}'})
    return model, bytes(binary)


class StartupTints(unittest.TestCase):
    def test_bakes_linear_tint_without_changing_archival_pixels_or_position(self):
        model, binary = fixture()
        original = bytes(binary)
        added = compact_tints(model, binary)
        self.assertEqual(binary, original)
        self.assertEqual(len(added), 2)
        primitives = [m['primitives'][0] for m in model['meshes']]
        self.assertEqual([p['material'] for p in primitives], [0,0])
        colors = []
        for primitive in primitives:
            self.assertEqual(primitive['attributes']['POSITION'], 0)
            accessor = model['accessors'][primitive['attributes']['COLOR_0']]
            colors.append(np.frombuffer(added[accessor['bufferView']], dtype='<f4').reshape(3,3))
        np.testing.assert_array_equal(colors[0], np.ones((3,3)))
        linear = lambda x: ((x / 255 + .055) / 1.055) ** 2.4
        expected = np.array([linear(x)/linear(160) for x in [120,128,136]], dtype='<f4')
        np.testing.assert_allclose(colors[1], np.tile(expected,(3,1)), rtol=1e-7)

    def test_patterns_observed_keys_existing_colors_and_different_policy_keep_original(self):
        for kind in ['pattern', 'observed', 'color', 'rank', 'normal', 'outside']:
            with self.subTest(kind=kind):
                model, binary = fixture(pattern=kind == 'pattern')
                if kind == 'observed': model['nodes'][1]['name'] += '~obs1~fam=plaster'
                if kind == 'color': model['meshes'][1]['primitives'][0]['attributes']['COLOR_0'] = 0
                if kind == 'rank': model['nodes'][1]['name'] = 'facades__x___stucco'
                if kind == 'normal': model['materials'][1]['normalTexture'] = {'index':1}
                if kind == 'outside': model['nodes'][1]['name'] = 'terrain__x___stucco'
                before = copy.deepcopy(model)
                self.assertEqual(compact_tints(model, binary), {})
                self.assertEqual(model, before)

    def test_thin_reveals_inherit_the_same_linear_tint_as_their_dominant_wall(self):
        model, binary = fixture()
        position = np.array([[0,0,0],[.1,0,0],[0,.1,0]],dtype='<f4').tobytes()
        view_id = len(model['bufferViews'])
        model['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':len(position)})
        binary += position
        model['accessors'][0].update({'componentType':5126,'bufferView':view_id})
        model['materials'].append({'name':'recess'})
        model['meshes'].append({'primitives':[{'attributes':{'POSITION':0},'material':2}]})
        model['nodes'].append({'mesh':2,'name':'facades__1___recess'})
        added = compact_tints(model,binary)
        wall = model['meshes'][1]['primitives'][0]['attributes']['COLOR_0']
        reveal = model['meshes'][2]['primitives'][0]['attributes']['COLOR_0']
        self.assertEqual(wall,reveal)
        self.assertEqual(model['meshes'][2]['primitives'][0]['material'], 0)
        self.assertTrue(model['meshes'][2]['extras']['startupWallReveal'])
        self.assertTrue(added)
