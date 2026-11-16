import pytest

from regen.specs import _style


@pytest.mark.parametrize('kind', ['panel', 'glazed', 'grille'])
def test_vehicle_leaf_retains_its_observed_colour(kind):
    style = _style({'kind': 'garage', 'frame': '#6a3a36',
                    'door': {'kind': kind, 'hex': '#cfc8b8'}})
    assert style['material'] == style['slatMaterial'] == '#cfc8b8'
    assert style['frameMaterial'] == '#6a3a36'


def test_deep_observed_opening_is_not_flattened_to_window_default():
    assert _style({'kind': 'garage', 'recessM': 3.5})['recessM'] == 3.5


@pytest.mark.parametrize('depth', [-1, float('nan'), float('inf')])
def test_invalid_depth_cannot_make_inverted_or_unbounded_geometry(depth):
    with pytest.raises(ValueError, match='recess'):
        _style({'kind': 'garage', 'recessM': depth})


def test_vehicle_leaf_and_three_sided_casing_both_reach_geometry():
    import numpy as np
    from shapely.geometry import box
    from pipeline.facade_grammar import compile_wall
    meshes, blocks = [], []
    ctx = {'put': lambda mesh, key, mat: meshes.append((mesh, mat)),
           'cube': lambda center, size, key, mat, angle: blocks.append((center, size, mat)),
           'polys': lambda p: [p] if p.geom_type == 'Polygon' else list(p.geoms)}
    st = _style({'kind': 'garage', 'frame': '#6a3a36', 'recessM': .4,
                 'door': {'kind': 'panel', 'hex': '#cfc8b8'}})
    profile = {'id': 'view', 'normal': [0., -1.], 'sourceIds': ['photo'], 'material': 'stucco',
               'openings': [{'rect': [.2, .1, .8, .6], 'style': st}]}
    compile_wall(ctx, {'g': box(0, 0, 10, 10), 'id': 'building'},
                 np.array([0., 0.]), np.array([10., 0.]), 0., 10., profile, [])
    assert any(mat == '#cfc8b8' for _, _, mat in blocks)
    casing = [mesh for mesh, mat in meshes if mat == '#6a3a36']
    assert casing
    # The casing does not fill the opening or add a ground-level crossbar.
    tri = np.concatenate([m.triangles for m in casing])
    from shapely.geometry import Polygon
    projected = [Polygon(t[:, [0, 2]]) for t in tri]
    assert sum(p.intersection(box(3, 2, 7, 5)).area for p in projected) < 1e-7
    assert min(m.bounds[0, 2] for m in casing) >= 1.06 - 1e-6
