from copy import deepcopy

from areas.ring.splice import splice, material_signature


def scene(pixel=b'old!', offset=0):
    return {
        'scene': 0, 'scenes': [{'nodes': [0]}],
        'nodes': [{'name': 'facades__sample___render', 'mesh': 0}],
        'meshes': [{'primitives': [{'attributes': {}, 'material': 0}]}],
        'materials': [{'name': 'render_same_recipe', 'pbrMetallicRoughness': {
            'baseColorTexture': {'index': 0}, 'roughnessFactor': .8}}],
        'textures': [{'source': 0}],
        'images': [{'bufferView': 0, 'mimeType': 'image/png'}],
        'bufferViews': [{'buffer': 0, 'byteOffset': offset, 'byteLength': len(pixel)}],
    }, b'\0' * offset + pixel


def test_rebuilt_maps_survive_same_named_old_material():
    base, bb = scene(); rebuilt, rb = scene(b'new!')
    binary, stats = splice(base, bb, rebuilt, rb, {'sample'})
    assert stats['newMaterials'] == 1
    material = base['meshes'][base['nodes'][base['scenes'][0]['nodes'][0]]['mesh']]['primitives'][0]['material']
    assert material == 1
    assert material_signature(base, binary, material) == material_signature(rebuilt, rb, 0)
    assert material_signature(base, binary, 0) != material_signature(rebuilt, rb, 0)


def test_identical_appearance_reuses_maps_despite_buffer_offsets():
    base, bb = scene(); rebuilt, rb = scene(offset=32)
    _, stats = splice(base, bb, rebuilt, rb, {'sample'})
    assert stats['newMaterials'] == 0
    assert stats['newImages'] == 0


def test_changed_roughness_survives_identical_maps():
    base, bb = scene(); rebuilt, rb = scene()
    rebuilt['materials'][0]['pbrMetallicRoughness']['roughnessFactor'] = .3
    _, stats = splice(base, bb, rebuilt, rb, {'sample'})
    assert stats['newMaterials'] == 1
    assert base['materials'][-1]['pbrMetallicRoughness']['roughnessFactor'] == .3
