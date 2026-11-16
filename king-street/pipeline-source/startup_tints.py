"""Conservatively bake tint-only wall materials before decoding their images.

This is a display derivative. The archival scene is unchanged. Unlike the live
sampled comparison, eligibility is checked at full image resolution and all
other texture bytes/material states must match exactly. Unsupported layouts,
observed materials, existing colors and mixed surface policies keep originals.
"""
import collections
import copy
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image

LAYERS = {'buildings', 'facades', 'foundations', 'roofs'}
# Classes with nearly uniform diffuse maps. Patterned surfaces never enter the
# approximation, even if an unusually small texture happened to pass the test.
CLASSES = {'stucco', 'stucco_red', 'render_white', 'render_gray', 'render_salmon'}
_CHANNELS = np.arange(256, dtype=np.float64) / 255
_LINEAR = np.where(_CHANNELS <= .04045, _CHANNELS / 12.92, ((_CHANNELS + .055) / 1.055) ** 2.4)


def compact_tints(model, binary):
    """Mutate static primitive/material references; return appended view bytes."""
    views, images, textures = model['bufferViews'], model.get('images', []), model.get('textures', [])
    users = collections.defaultdict(list)
    blocked = set()
    wall_counts = collections.defaultdict(collections.Counter)
    recesses = []
    mesh_users = collections.Counter(n.get('mesh') for n in model.get('nodes', []))
    # A material must have one unchanged finishing/weathering/depth policy.
    for node in model.get('nodes', []):
        if 'mesh' not in node:
            continue
        mesh = model['meshes'][node['mesh']]
        name = node.get('name', mesh.get('name', ''))
        layer = node.get('extras', {}).get('layer') or mesh.get('extras', {}).get('layer') or name.split('___')[0].split('__')[0]
        cls = name.split('___')[-1].split('#')[0].split('~obs1')[0].split('__')[0]
        feature = node.get('extras', {}).get('featureId') or mesh.get('extras', {}).get('featureId') or (name.split('___')[0].split('__') + ['', ''])[1]
        rank = 3 if layer == 'facades' else 2 if layer == 'roofs' else 1 if layer == 'buildings' and '__observed_' in name else 0 if layer == 'buildings' else -1
        for primitive in mesh.get('primitives', []):
            material = primitive.get('material')
            if material is None:
                continue
            if feature and layer in {'buildings', 'facades'} and re.match(r'^(brick|render|stucco|stone|limestone|cast_stone|concrete|terracotta|timber|panel_dark|venue_concrete)', cls):
                count = model['accessors'][primitive.get('indices', primitive['attributes']['POSITION'])]['count'] / 3
                wall_counts[feature][material] += count
            if cls == 'recess':
                recesses.append((feature, primitive, mesh))
            if (mesh_users[node['mesh']] != 1 or len(mesh.get('primitives', [])) != 1 or layer not in LAYERS or cls not in CLASSES or '~obs1' in name or
                    primitive.get('mode', 4) != 4 or primitive.get('extensions') or primitive.get('targets') or
                    any(k.startswith('COLOR') for k in primitive['attributes']) or mesh.get('weights') or 'skin' in node):
                blocked.add(material)
            users[material].append((primitive, (cls, rank)))

    image_bytes, digests = {}, {}
    def texture_key(index):
        texture = textures[index]
        image_id = texture['source']
        if image_id not in image_bytes:
            image = images[image_id]
            view = views[image['bufferView']]
            start = view.get('byteOffset', 0)
            data = binary[start:start + view['byteLength']]
            image_bytes[image_id] = data
            digests[image_id] = hashlib.sha256(data).hexdigest()
        # Texture/image extras and all sampler state remain part of identity.
        return [digests[image_id], {k:v for k,v in texture.items() if k != 'source'},
                {k:v for k,v in images[image_id].items() if k != 'bufferView'}]

    groups = collections.defaultdict(list)
    for material_id, entries in users.items():
        if material_id in blocked or len({policy for _, policy in entries}) != 1:
            continue
        material = copy.deepcopy(model['materials'][material_id])
        material.pop('name', None)
        pbr = material.get('pbrMetallicRoughness', {})
        base = pbr.get('baseColorTexture')
        # Existing emissive/vertex color/alpha behavior keeps the live path.
        if not base or material.get('alphaMode', 'OPAQUE') != 'OPAQUE' or material.get('extensions') or 'emissiveTexture' in material:
            continue
        texture_key(base['index'])
        for part in (material, pbr):
            for key, value in part.items():
                if key.endswith('Texture') and isinstance(value, dict) and 'index' in value:
                    value['index'] = texture_key(value['index'])
        # Only the base image pixels differ; retain its sampler/metadata policy.
        base['index'][0] = None
        groups[(entries[0][1], json.dumps(material, sort_keys=True))].append(material_id)

    new_views, color_accessors = {}, {}
    def paint(primitive, tint):
        count = model['accessors'][primitive['attributes']['POSITION']]['count']
        key = (count, tuple(tint))
        accessor = color_accessors.get(key)
        if accessor is None:
            color = np.empty((count, 3), dtype='<f4'); color[:] = tint
            data = color.tobytes()
            view_id = len(views); views.append({'buffer': 0, 'byteLength': len(data)})
            new_views[view_id] = data
            accessor = len(model['accessors'])
            model['accessors'].append({'bufferView': view_id, 'componentType': 5126, 'type': 'VEC3', 'count': count})
            color_accessors[key] = accessor
        primitive['attributes']['COLOR_0'] = accessor

    def pixels(material_id):
        texture = textures[model['materials'][material_id]['pbrMetallicRoughness']['baseColorTexture']['index']]
        try:
            with Image.open(io.BytesIO(image_bytes[texture['source']])) as image:
                # Browser color conversion is not reproduced for authored ICC assets.
                if image.info.get('icc_profile'):
                    return None
                rgba = np.asarray(image.convert('RGBA'))
                if np.any(rgba[..., 3] != 255):
                    return None
                return _LINEAR[rgba[..., :3]]
        except (OSError, ValueError):
            return None

    baked_tints, baked_materials = {}, {}
    for material_ids in groups.values():
        if len(material_ids) < 2:
            continue
        reference = max(material_ids, key=lambda i: len(users[i]))
        source = pixels(reference)
        if source is None:
            continue
        mean = source.mean(axis=(0, 1))
        luminance = source @ np.array([.2126, .7152, .0722])
        if luminance.std() / max(float(luminance.mean()), 1e-4) > .06:
            continue
        into = []
        for candidate in material_ids:
            if candidate == reference:
                continue
            image = pixels(candidate)
            if image is None or image.shape != source.shape:
                continue
            tint = image.mean(axis=(0, 1)) / np.maximum(mean, 1e-6)
            tint = tint.astype('<f4')
            error = np.max(np.abs(image - source * tint) / np.maximum(image, .002), axis=2)
            # Stricter than the live 98%-within-5% test, at full resolution.
            if np.count_nonzero(error >= .03) > int(error.size * .01) or np.max(error) >= .15:
                continue
            into.append((candidate, tint.tolist()))
        if not into:
            continue
        baked_tints[reference] = [1., 1., 1.]
        baked_materials[reference] = reference
        for primitive, _ in users[reference]:
            paint(primitive, [1., 1., 1.])
        for candidate, tint in into:
            baked_tints[candidate] = tint
            baked_materials[candidate] = reference
            for primitive, _ in users[candidate]:
                paint(primitive, tint)
                primitive['material'] = reference
    # The live viewer gives thin opening reveals their building's dominant
    # wall finish before material merging. Carry that wall's baked tint too.
    def array(index):
        accessor = model['accessors'][index]
        dtype = {5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}.get(accessor.get('componentType'))
        width = {'SCALAR':1,'VEC3':3}.get(accessor.get('type'))
        if not dtype or not width or accessor.get('sparse') or accessor.get('normalized') or 'bufferView' not in accessor:
            return None
        view = views[accessor['bufferView']]
        if accessor['bufferView'] in new_views:
            return None
        dtype = np.dtype(dtype)
        return np.ndarray((accessor['count'],width), dtype, buffer=binary, offset=view.get('byteOffset',0)+accessor.get('byteOffset',0), strides=(view.get('byteStride',dtype.itemsize*width),dtype.itemsize))
    for feature, primitive, mesh in recesses:
        donors = wall_counts.get(feature)
        if not donors or 'COLOR_0' in primitive['attributes']:
            continue
        donor = max(donors, key=donors.get)
        if donor not in baked_tints:
            continue
        position = array(primitive['attributes']['POSITION'])
        index = array(primitive['indices']) if 'indices' in primitive else None
        if position is None or ('indices' in primitive and index is None):
            continue
        triangles = position[index.reshape(-1)].reshape(-1,3,3) if index is not None else position.reshape(-1,3,3)
        triangles = triangles.astype(np.float64)
        areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        if len(areas) and np.partition(areas,len(areas)//2)[len(areas)//2] < .25 and areas.max() < 2:
            paint(primitive,baked_tints[donor])
            primitive['material'] = baked_materials[donor]
            mesh.setdefault('extras', {})['startupWallReveal'] = True
    return new_views
