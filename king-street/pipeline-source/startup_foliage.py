"""Pack the centres used by the viewer's foliage shader into its startup model.

The authoritative model retains the original textured leaf geometry. Collapse
only four-triangle fans whose centre is shared by all four triangles. Other
meshes keep the existing extraction path, including old two-triangle leaves.
"""
import numpy as np


def _array(model, binary, index, dtype, width):
    accessor = model['accessors'][index]
    view = model['bufferViews'][accessor['bufferView']]
    if accessor.get('sparse') or accessor.get('normalized') or view.get('extensions'):
        return None
    stride = view.get('byteStride', dtype.itemsize * width)
    offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    return np.ndarray((accessor['count'], width), dtype, buffer=binary,
                      offset=offset, strides=(stride, dtype.itemsize))


def _fan_centres(model, binary, primitive):
    if primitive.get('mode', 4) != 4 or primitive.get('extensions') or primitive.get('targets'):
        return None
    position_id = primitive.get('attributes', {}).get('POSITION')
    if position_id is None:
        return None
    position = model['accessors'][position_id]
    if position.get('componentType') != 5126 or position.get('type') != 'VEC3' or 'bufferView' not in position:
        return None
    vertices = _array(model, binary, position_id, np.dtype('<f4'), 3)
    if vertices is None:
        return None
    if 'indices' in primitive:
        index_id = primitive['indices']
        index = model['accessors'][index_id]
        dtype = {5121: 'u1', 5123: '<u2', 5125: '<u4'}.get(index['componentType'])
        if not dtype or 'bufferView' not in index:
            return None
        order = _array(model, binary, index_id, np.dtype(dtype), 1)
        if order is None or order.size % 12 or order.size == 0:
            return None
        centre_indices = order.reshape(-1, 12)[:, 2::3].T
        centres = vertices[centre_indices[0]]
        if not all(np.array_equal(centres, vertices[other]) for other in centre_indices[1:]):
            return None
    else:
        if len(vertices) % 12 or len(vertices) == 0:
            return None
        centres = vertices[2::12]
        if not all(np.array_equal(centres, vertices[start::12]) for start in (5, 8, 11)):
            return None
    if not np.isfinite(centres).all():
        return None
    return np.ascontiguousarray(centres)


def compact_foliage(model, bin_data):
    """Mutate the startup glTF and return newly appended buffer views by index."""
    new_views = {}
    nodes_by_mesh = {}
    for node in model.get('nodes', []):
        if 'mesh' in node:
            nodes_by_mesh.setdefault(node['mesh'], []).append(node)

    def append_view(data):
        index = len(model['bufferViews'])
        model['bufferViews'].append({'buffer': 0, 'byteLength': len(data)})
        new_views[index] = data
        return index

    # One degenerate triangle keeps the primitive a Mesh so the normal Three
    # loader preserves the material and renderer integration for leaf sources.
    # The renderer always hides this input mesh and displays the grown canopy.
    placeholder = None
    for i, mesh in enumerate(model.get('meshes', [])):
        nodes = nodes_by_mesh.get(i, [])
        if not nodes or len(mesh['primitives']) != 1:
            continue
        def is_leaf(node):
            name = node.get('name', mesh.get('name', ''))
            layer = node.get('extras', {}).get('layer') or mesh.get('extras', {}).get('layer') or name.split('__')[0]
            parts = name.split('___')
            return layer == 'vegetation' and len(parts) > 1 and parts[-1].startswith('leaf')
        if not all(is_leaf(node) for node in nodes) or mesh.get('weights') or any('skin' in n for n in nodes):
            continue
        primitive = mesh['primitives'][0]
        centres = _fan_centres(model, bin_data, primitive)
        if centres is None:
            continue
        if placeholder is None:
            view = append_view(bytes(3))
            placeholder = len(model['accessors'])
            model['accessors'].append({'componentType': 5121, 'count': 3, 'type': 'SCALAR',
                                       'min': [0], 'max': [0], 'bufferView': view})
        view = append_view(centres.tobytes())
        position = len(model['accessors'])
        model['accessors'].append({'componentType': 5126, 'count': len(centres), 'type': 'VEC3',
                                   'min': centres.min(axis=0).tolist(), 'max': centres.max(axis=0).tolist(),
                                   'bufferView': view})
        primitive['attributes'] = {'POSITION': position}
        primitive['indices'] = placeholder
        for node in nodes:
            node.setdefault('extras', {})['foliageFanCentres'] = True
        mesh.setdefault('extras', {})['foliageFanCentres'] = True
    return new_views
