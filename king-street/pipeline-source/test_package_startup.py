import unittest

from package_startup import deduplicate_images


class ImageDeduplication(unittest.TestCase):
    def test_bytes_metadata_and_color_space_must_match(self):
        binary = memoryview(b'abcabcabcabcxyz')
        model = {
            'bufferViews': [{'buffer': 0, 'byteOffset': i * 3, 'byteLength': 3} for i in range(5)],
            'images': [{'bufferView': i, 'mimeType': 'image/png'} for i in range(5)],
            'textures': [{'source': i} for i in range(5)],
            'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': i}}} for i in (0, 1, 3, 4)]
                         + [{'normalTexture': {'index': 2}}],
        }
        model['images'][3]['extras'] = {'credit': 'separate asset'}
        images = deduplicate_images(model, binary)
        self.assertEqual([image['bufferView'] for image in images], [0, 2, 3, 4])
        self.assertEqual([t['source'] for t in model['textures']], [0, 0, 1, 2, 3])


class ResourcePruning(unittest.TestCase):
    def test_live_references_are_remapped_and_metadata_kept(self):
        from package_startup import prune_material_resources
        model = {
            'meshes': [{'primitives': [{'material':2,'attributes':{}}, {'attributes':{}}]}],
            'materials': [{}, {}, {'normalTexture':{'index':2}, 'pbrMetallicRoughness':{'baseColorTexture':{'index':4}},
                                  'extras':{'notATexture':{'index':99}}}],
            'textures': [{'source':0},{'source':1},{'source':4},{'source':3},{'source':2}],
            'images': [{'name':str(i)} for i in range(5)],
        }
        prune_material_resources(model)
        self.assertEqual(model['meshes'][0]['primitives'][0]['material'], 0)
        self.assertEqual(model['materials'][0]['normalTexture']['index'], 0)
        self.assertEqual(model['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index'], 1)
        self.assertEqual([t['source'] for t in model['textures']], [1,0])
        self.assertEqual([i['name'] for i in model['images']], ['2','4'])
        self.assertEqual(model['materials'][0]['extras']['notATexture']['index'], 99)
