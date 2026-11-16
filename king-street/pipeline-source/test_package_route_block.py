import gzip
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from package_route_block import Source, compile_glb, package, route_box, building_id, scope_world
from package_startup import emit_chunks


def fixture():
    binary = bytearray(); views = []; accessors = []
    def accessor(values, component=5126, kind='VEC3', normalized=False):
        dtype = {5126:'<f4', 5121:'u1'}[component]
        a = np.array(values, dtype=dtype)
        views.append({'buffer':0, 'byteOffset':len(binary), 'byteLength':a.nbytes})
        binary.extend(a.tobytes()); binary.extend(bytes(-len(binary)%4))
        accessors.append({'bufferView':len(views)-1, 'componentType':component, 'type':kind,
                          'count':len(a), 'min':a.min(axis=0).tolist(), 'max':a.max(axis=0).tolist(),
                          **({'normalized':True} if normalized else {})})
        return len(accessors)-1
    def primitive(points, texture=False):
        p = {'attributes':{'POSITION':accessor(points)}, 'material':0}
        if texture:
            p['attributes']['TEXCOORD_0'] = accessor([[i/10, i/20] for i in range(len(points))], kind='VEC2')
            p['attributes']['COLOR_0'] = accessor([[i,128,255,255] for i in range(len(points))], component=5121, kind='VEC4', normalized=True)
        return p
    near = primitive([[0,0,0], [1,0,0], [0,1,0]])
    companion = primitive([[100,0,0], [101,0,0], [100,1,0]])
    distant = primitive([[200,0,0], [201,0,0], [200,1,0]])
    road = primitive([[200,0,0],[201,0,0],[200,0,-1], [-10,0,0],[10,0,0],[0,0,-10]], True)
    fan = primitive([[0,0,0],[50,0,0],[1,0,-1]])
    fan['indices'] = accessor([[0],[0],[0]], component=5121, kind='SCALAR')
    image = b'tiny-encoded-image'
    views.append({'buffer':0,'byteOffset':len(binary),'byteLength':len(image)})
    binary.extend(image); binary.extend(bytes(-len(binary)%4))
    model = {'asset':{'version':'2.0'}, 'scene':0, 'scenes':[{'nodes':[0]}],
             'nodes':[{'name':'world','children':[1,2,3,4,5]},
                      {'name':'facades__A___glass','mesh':0}, {'name':'roofs__A___roof','mesh':1},
                      {'name':'facades__B___plaster','mesh':2}, {'name':'roads__shared','mesh':3},
                      {'name':'vegetation___leaf','mesh':4,'extras':{'foliageFanCentres':True}}],
             'meshes':[{'primitives':[p]} for p in (near,companion,distant,road,fan)],
             'accessors':accessors, 'bufferViews':views, 'buffers':[{'byteLength':len(binary)}],
             'materials':[{'pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}, {'name':'unused'}],
             'textures':[{'source':0,'sampler':0}], 'samplers':[{'wrapS':33071}],
             'images':[{'bufferView':len(views)-1,'mimeType':'image/png'}]}
    return make_glb(model, binary), image


def make_glb(model, binary):
    text = json.dumps(model).encode(); text += b' '*(-len(text)%4)
    return struct.pack('<5I',0x46546c67,2,28+len(text)+len(binary),len(text),0x4e4f534a)+text+struct.pack('<II',len(binary),0x004e4942)+binary


class RouteBlockTests(unittest.TestCase):
    def test_building_group_triangles_fan_and_material_attributes(self):
        raw, image = fixture(); packed=io.BytesIO()
        stats = compile_glb(io.BytesIO(raw), io.BytesIO(), packed, [-2,-2,2,2])
        self.assertEqual(stats['meshes'],4)
        out = Source(io.BytesIO(packed.getvalue())); m = out.model
        self.assertEqual([n['name'] for n in m['nodes']], ['facades__A___glass','roofs__A___roof','roads__shared','vegetation___leaf'])
        self.assertEqual(len(m['materials']),1)
        self.assertEqual(m['samplers'],[{'wrapS':33071}])
        road = m['meshes'][2]['primitives'][0]
        self.assertEqual(out.array(road['attributes']['POSITION']).shape,(3,3))
        np.testing.assert_allclose(out.array(road['attributes']['TEXCOORD_0']), [[.3,.15],[.4,.2],[.5,.25]])
        self.assertTrue(m['accessors'][road['attributes']['COLOR_0']]['normalized'])
        np.testing.assert_array_equal(out.array(road['indices']).reshape(-1), [0,1,2])
        np.testing.assert_array_equal(out.array(road['attributes']['COLOR_0'])[:,0], [3,4,5])
        self.assertEqual(out.array(m['meshes'][3]['primitives'][0]['attributes']['POSITION']).shape,(2,3))
        offset, length = out.view_range(m['images'][0]['bufferView'])
        out.stream.seek(out.offset+offset); self.assertEqual(out.stream.read(length),image)

    def test_canonical_building_spans_boundary_with_foundation_and_observed_facade(self):
        raw, _ = fixture(); source = Source(io.BytesIO(raw)); model = source.model
        model['nodes'][1].update(name='facades__wrong__observed_7___glass', extras={'layer': 'facades', 'featureId': 'A'})
        # Foundation lies outside the route but belongs to its full building,
        # just like the far roof. Mesh extras also identify building owners.
        model['nodes'][3]['name'] = 'foundation display label'
        model['meshes'][2]['extras'] = {'layer': 'foundations', 'featureId': 'A'}
        model['nodes'][2]['name'] = 'roofs__A__observed_9___roof'
        packed = io.BytesIO()
        stats = compile_glb(io.BytesIO(make_glb(model, raw[source.offset:])), io.BytesIO(), packed, [-2,-2,2,2])
        self.assertEqual(stats['buildings'], 1)
        self.assertEqual(stats['buildingIds'], ['A'])
        out = Source(io.BytesIO(packed.getvalue()))
        self.assertEqual(len(out.model['meshes']), 5)
        foundation = out.model['meshes'][2]['primitives'][0]
        np.testing.assert_allclose(out.array(foundation['attributes']['POSITION']), [[200,0,0],[201,0,0],[200,1,0]])
        self.assertEqual(out.model['nodes'][2]['matrix'][12:15], [0,0,0])
        self.assertEqual(building_id({'name':'facades__A__observed_7___glass'}, {}), 'A')
        self.assertEqual(building_id({'name':'foundations__A___base'}, {}), 'A')
        self.assertEqual(building_id({'name':'facades__A___glass','extras':{'layer':'roads','featureId':'A'}}, {}), None)

    def test_world_records_follow_present_feature_ids_even_when_centers_are_outside(self):
        grid = {'min': -900, 'step': 3, 'n': 601}; ground = [2,3,4]
        world = {'center': {'lat': 37, 'lon': -122}, 'grid': grid, 'ground': ground,
                 'buildings': [{'id': 'A', 'center': [200,200,0]}, {'id': 'frontage', 'clippedAtBoundary': True, 'center': [50,50,0]}, {'id':'B','center':[0,0,0]}],
                 'counts': {'buildings': 949, 'windowAssemblies': 34597, 'lidarPoints': 20311808},
                 'streetViews': [{'position':[0,0]}, {'position':[500,500]}]}
        scope_world(world, {'A', 'frontage'}, [-2,-2,2,2])
        self.assertEqual([feature['id'] for feature in world['buildings']], ['A', 'frontage'])
        self.assertEqual(world['counts'], {'buildings': 2})
        self.assertEqual(world['sourceCounts']['buildings'], 949)
        self.assertNotIn('windowAssemblies', world['counts'])
        self.assertEqual(world['grid'], grid); self.assertIs(world['ground'], ground)
        self.assertEqual(world['center'], {'lat':37, 'lon':-122})
        self.assertEqual(world['buildings'][0]['center'], [200,200,0])
        self.assertEqual(len(world['streetViews']), 1)

    def test_parent_matrix_and_north_sign(self):
        raw, _ = fixture(); source=Source(io.BytesIO(raw)); m=source.model
        m['nodes'][0]['translation']=[10,0,-20]
        original_binary=raw[source.offset:]
        packed=io.BytesIO()
        stats=compile_glb(io.BytesIO(make_glb(m,original_binary)),io.BytesIO(),packed,[8,18,12,22])
        self.assertEqual(stats['buildings'],1)
        self.assertEqual(Source(io.BytesIO(packed.getvalue())).model['nodes'][0]['matrix'][12:15],[10,0,-20])

    def test_reject_unsupported_and_invalid_route(self):
        raw,_=fixture(); source=Source(io.BytesIO(raw)); source.model['animations']=[{}]
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            compile_glb(io.BytesIO(make_glb(source.model, raw[source.offset:])),io.BytesIO(),io.BytesIO(),[-2,-2,2,2])
        with self.assertRaises(ValueError): route_box([[0,0],[float('nan'),2]],42)

    def test_package_filters_metadata_and_preserves_frozen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            public=Path(directory); (public/'block-trial').mkdir(); (public/'compare').mkdir()
            raw,_=fixture(); packed=public/'input.glb'; packed.write_bytes(raw)
            with tempfile.TemporaryDirectory(dir=public) as scratch:
                portable=emit_chunks(packed,Path(scratch),public)
            portable['sourceHash']='a'*12
            (public/('startup-v3-'+'a'*12+'.json')).write_text(json.dumps(portable))
            original={'contentHash':'a'*12,'world':'/world.json','cars':'/cars.json','overhead':'/overhead.json'}
            frozen=public/'block-trial/frozen.json'; frozen.write_text(json.dumps(original))
            trial=public/'block-trial/trial.json'; trial.write_text(json.dumps({'manifest':'/block-trial/frozen.json','routeEN':[[0,0],[1,1]]}))
            before=(frozen.read_bytes(),trial.read_bytes())
            (public/'world.json').write_text(json.dumps({'grid':{'kept':True},'buildings':[{'id':'A'},{'id':'B'}],'counts':{'buildings':949,'windowAssemblies':500},'streetViews':[{'position':[0,0]},{'position':[500,500]}]}))
            (public/'cars.json').write_text(json.dumps({'types':[{'name':'sedan'}],'cars':[{'x':0,'y':0},{'x':500,'y':500}]}))
            (public/'overhead.json').write_text(json.dumps({'poles':[{'x':0,'y':0},{'x':500,'y':500}], 'wires':[{'p':[[-10,0,3],[10,0,3]]},{'p':[[500,500,3],[501,500,3]]}]}))
            result=package(public,'block-trial/trial.json',2)
            self.assertEqual(result['cars'],1); self.assertEqual(result['poles'],1); self.assertEqual(result['wires'],1)
            alias=json.loads((public/'compare/route-block.json').read_text())
            self.assertEqual(alias['scope'],'route-block'); self.assertEqual(alias['sourceWorldHash'],'a'*12)
            self.assertEqual(alias['buildingIds'], ['A']); self.assertEqual(alias['stats']['buildings'],1)
            self.assertEqual(alias['compilerVersion'],2)
            world=json.loads((public/alias['world'].lstrip('/')).read_text()); self.assertEqual(world['grid'],{'kept':True});self.assertEqual(len(world['streetViews']),1)
            self.assertEqual(world['buildings'], [{'id':'A'}]); self.assertEqual(world['counts'], {'buildings':1})
            self.assertEqual(world['sourceCounts'], {'buildings':949,'windowAssemblies':500})
            self.assertEqual(before,(frozen.read_bytes(),trial.read_bytes()))
            emitted=b''.join(gzip.decompress((public/u.lstrip('/')).read_bytes()) for u in alias['chunks'])
            self.assertEqual(len(emitted),result['uncompressedBytes']); self.assertEqual(len(Source(io.BytesIO(emitted)).model['meshes']),4)
            saved=(public/'compare/route-block.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'selected no meshes'):
                trial.write_text(json.dumps({'manifest':'/block-trial/frozen.json','routeEN':[[1000,1000],[1001,1001]]}))
                package(public,'block-trial/trial.json',2)
            self.assertEqual((public/'compare/route-block.json').read_bytes(),saved)


if __name__ == '__main__': unittest.main()
