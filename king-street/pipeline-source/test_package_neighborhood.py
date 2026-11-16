import io
import copy
from PIL import Image
import unittest

import numpy as np

from package_neighborhood import NeighborhoodPlan, ProxySimplifier, tile_of
from package_drive_stream import compile_part
from package_route_block import Source, Writer, building_id
from test_package_route_block import fixture, make_glb


def source_fixture():
    raw, _ = fixture()
    source = Source(io.BytesIO(raw))
    model = source.model
    model['materials'][0] = {'name': 'source-color', 'pbrMetallicRoughness': {'baseColorFactor': [.2, .4, .6, 1]}}
    # The same feature spans several tiles across its facade and roof.
    # Give it an actual envelope for the startup exterior path.
    model['nodes'][1]['name'] = 'buildings__A___paint'
    model['nodes'][2]['translation'] = [0, 0, -1]
    return Source(io.BytesIO(make_glb(model, raw[source.offset:])))


class NeighborhoodTests(unittest.TestCase):
    def test_support_keeps_exact_original_material_names_pbr_and_maps(self):
        source = source_fixture()
        original = {'name': 'asphalt', 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1],
            'baseColorTexture': {'index': 0}, 'metallicRoughnessTexture': {'index': 0},
            'roughnessFactor': .62, 'metallicFactor': 0}, 'normalTexture': {'index': 0, 'scale': 1.2},
            'doubleSided': False, 'extras': {'authored': True}}
        source.model['materials'].append(copy.deepcopy(original))
        for node in source.model['nodes']:
            if node.get('name', '').startswith('roads__'):
                source.model['meshes'][node['mesh']]['primitives'][0]['material'] = len(source.model['materials'])-1
        plan = NeighborhoodPlan(source, 55)
        plan.prepare_base(source)
        packed = io.BytesIO()
        compile_part(source, io.BytesIO(), packed, plan, None)
        output = Source(io.BytesIO(packed.getvalue()))
        material = next(m for m in output.model['materials'] if m.get('name') == 'asphalt')
        self.assertEqual(material, original)
        self.assertGreaterEqual(len(output.model['images']), 1)

    def test_grid_floor_and_full_feature_ownership(self):
        np.testing.assert_array_equal(tile_of([[-.01, 0], [55, 54.99]], 55), [[-1, 0], [1, 0]])
        source = source_fixture()
        plan = NeighborhoodPlan(source, 55)
        owner = plan.owners['A']
        self.assertEqual(plan.cells[owner]['tileEN'], [0, 0, 55, 55])
        packed = io.BytesIO()
        stats = compile_part(source, io.BytesIO(), packed, plan, owner)
        output = Source(io.BytesIO(packed.getvalue()))
        names = [n['name'] for n in output.model['nodes']]
        self.assertIn('buildings__A___paint', names)
        self.assertIn('roofs__A___roof', names)
        # Real mesh bounds, not the selection tile: the companion roof extends east.
        self.assertEqual(stats['boundsEN'][2], 101)
        self.assertEqual(stats['buildingIds'], ['A'])

    def test_source_foliage_is_owned_once_across_grid(self):
        source = source_fixture()
        plan = NeighborhoodPlan(source, 25)
        counts = 0
        for cell in range(len(plan.cells)):
            for node, matrix in plan.nodes:
                mesh = source.model['meshes'][node['mesh']]
                if not node.get('extras', {}).get('foliageFanCentres'):
                    continue
                if not plan.keep_node(node, mesh, building_id(node, mesh), cell):
                    continue
                primitive = mesh['primitives'][0]
                positions = source.array(primitive['attributes']['POSITION'])
                indices = source.array(primitive['indices']).reshape(-1)
                used, _ = plan.select(positions, matrix, indices, True, 4, cell)
                counts += len(used)
        self.assertEqual(counts, 3)

    def test_base_retains_source_roof_geometry_without_invented_walls(self):
        source = source_fixture()
        plan = NeighborhoodPlan(source, 55)
        plan.prepare_base(source)
        packed = io.BytesIO()
        compile_part(source, io.BytesIO(), packed, plan, None)
        output = Source(io.BytesIO(packed.getvalue()))
        proxies = [n for n in output.model['nodes'] if n.get('extras', {}).get('class') == 'distant_exterior'
                   and n['extras'].get('layer') != 'vegetation']
        self.assertEqual(len(proxies), 1)
        self.assertEqual({n['extras']['streamCell'] for n in proxies}, {plan.cells[plan.owners['A']]['id']})
        self.assertTrue(any(n.get('extras', {}).get('class') == 'collision_hull' for n in output.model['nodes']))
        for node in proxies:
            primitive = output.model['meshes'][node['mesh']]['primitives'][0]
            material = output.model['materials'][primitive['material']]
            self.assertNotIn('baseColorTexture', material['pbrMetallicRoughness'])
            self.assertEqual(material['pbrMetallicRoughness']['baseColorFactor'], [1, 1, 1, 1])
            colors = output.array(primitive['attributes']['COLOR_0'])
            np.testing.assert_allclose(colors[:, :3] / 255, np.tile([.2, .4, .6], (len(colors), 1)), atol=1/255)

    def test_source_facade_atlas_keeps_windows_silhouette_and_owned_resources(self):
        binary = io.BytesIO(); writer = Writer(binary)
        shapes = [
            ('foundations', [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 0], [4, 0, 3], [0, 0, 3]], 0),
            ('buildings', [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 0, 0], [4, 4, 0], [0, 4, 0]], 0),
            ('facades', [[1, 1, -.025], [2, 1, -.025], [2, 3, -.025], [1, 1, -.025], [2, 3, -.025], [1, 3, -.025]], 1)]
        nodes, meshes = [], []
        for layer, points, material in shapes:
            aid = writer.accessor(np.array(points, dtype='<f4'), {'type': 'VEC3', 'componentType': 5126, 'min': [], 'max': []})
            extra = {'layer': layer, 'featureId': 'A'}
            nodes.append({'name': layer+'__A___paint', 'mesh': len(meshes), 'extras': extra})
            meshes.append({'primitives': [{'attributes': {'POSITION': aid}, 'material': material}], 'extras': extra})
        model = {'asset': {'version': '2.0'}, 'nodes': nodes, 'meshes': meshes,
                 'scenes': [{'nodes': list(range(len(nodes)))}], 'scene': 0,
                 'materials': [{'name': 'paint', 'pbrMetallicRoughness': {'baseColorFactor': [.7, .5, .2, 1]}},
                               {'name': 'glass', 'pbrMetallicRoughness': {'baseColorFactor': [.02, .1, .8, 1]}}],
                 'accessors': writer.accessors, 'bufferViews': writer.views, 'buffers': [{'byteLength': binary.tell()}]}
        source = Source(io.BytesIO(make_glb(model, binary.getvalue())))
        plan = NeighborhoodPlan(source, 55); plan.prepare_base(source)
        packed = io.BytesIO(); compile_part(source, io.BytesIO(), packed, plan, None)
        output = Source(io.BytesIO(packed.getvalue()))
        self.assertEqual(plan.atlas_stats['buildings'], 1)
        self.assertLessEqual(plan.atlas_stats['texels'], 2*2048**2)
        image = output.model['images'][0]
        self.assertNotIn('neighborhoodWriterView', image.get('extras', {}))
        start, length = output.view_range(image['bufferView'])
        output.stream.seek(output.offset+start)
        pixels = np.asarray(Image.open(io.BytesIO(output.stream.read(length))))
        opaque = pixels[:, :, 3] > 0
        self.assertTrue(opaque.any()); self.assertFalse(opaque.all())
        rgb = pixels[:, :, :3][opaque]
        self.assertTrue(np.any(rgb[:, 2] > rgb[:, 0]+40))  # original blue glazing survives
        self.assertTrue(np.any(rgb[:, 0] > rgb[:, 2]+40))  # original warm wall survives
        impostors = [n for n in output.model['nodes'] if n.get('extras', {}).get('bakedAppearance')]
        self.assertEqual(len(impostors), 1)
        self.assertEqual(impostors[0]['extras']['streamCell'], plan.cells[plan.owners['A']]['id'])
        self.assertTrue(output.model['materials'][-1]['extras']['bakedAppearance'])

    def test_proxy_simplification_reduces_planar_grid_and_keeps_boundary(self):
        p = np.array([[x, 0, z] for z in range(12) for x in range(12)], dtype='<f4')
        indices = []
        for z in range(11):
            for x in range(11):
                a = z*12+x
                indices.extend([a, a+1, a+13, a, a+13, a+12])
        normal = np.tile([0, 1, 0], (len(p), 1)).astype('<f4')
        color = np.tile([.2, .4, .6, 1], (len(p), 1)).astype('<f4')
        simplifier = ProxySimplifier()
        try:
            reduced, _, _, compact = simplifier.simplify(p, normal, color, np.array(indices, dtype='<u4'))
        finally:
            simplifier.close()
        self.assertLess(len(compact), len(indices)//2)
        expected = {tuple(v) for v in p if v[0] in (0, 11) or v[2] in (0, 11)}
        self.assertTrue(expected.issubset({tuple(v) for v in reduced}))

    def test_perpendicular_proxy_faces_keep_hard_normals(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='<f4')
        index = np.array([0, 1, 2, 0, 3, 1], dtype='<u4')
        normal = np.zeros_like(p)
        color = np.ones((4, 4), dtype='<f4')
        simplifier = ProxySimplifier()
        try:
            q, normals, _, indices = simplifier.simplify(p, normal, color, index)
        finally:
            simplifier.close()
        self.assertEqual(len(q), 6)
        np.testing.assert_allclose(normals[indices[:3]], np.tile([0, 0, 1], (3, 1)))
        np.testing.assert_allclose(normals[indices[3:]], np.tile([0, 1, 0], (3, 1)))


if __name__ == '__main__':
    unittest.main()
