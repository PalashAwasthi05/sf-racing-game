import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.facade_grammar import (  # noqa: E402
    _component_material,
    compile_wall,
    head_shape,
    opening_outline,
)


def test_arch_is_shaped_before_segment_clip():
    opening = {'rect': [0.0, 0.0, 1.0, 1.0], 'style': {'head': 'arch', 'springRatio': 0.5}}
    full = opening_outline(opening, 10.0, 10.0)
    clipped = full.intersection(box(3.0, 0.0, 10.0, 10.0))

    # Rebuilding an arch from the clipped [3, 10] box moves its centre to 6.5.
    rebuilt = opening_outline(
        {'rect': [0.3, 0.0, 1.0, 1.0], 'style': opening['style']}, 10.0, 10.0
    )
    assert clipped.area != rebuilt.area
    assert clipped.contains(Point(5.0, 9.8))
    assert not rebuilt.contains(Point(5.0, 9.8))


def test_polygon_only_opening_compiles_and_keeps_contour():
    emitted = []

    def cube(size, key, material, angle):
        emitted.append(('cube', key, material, size))

    def put(mesh, key, material):
        emitted.append(('mesh', key, material, mesh))

    ctx = {
        'polys': lambda geom: [geom],
        'cube': cube,
        'put': put,
    }
    profile = {
        'id': 'polygon-window',
        'normal': [0.0, 1.0],
        'sourceIds': ['test-source'],
        'material': 'brick',
        'openings': [
            {
                'polygon': [[0.2, 0.1], [0.48, 0.1], [0.48, 0.45], [0.2, 0.62]],
                'style': {'frameWidthM': 0.05, 'glassMaterial': 'glass'},
            }
        ],
    }
    building = {'id': 'test-building', 'g': Polygon([(-5, 0), (5, 0), (5, 10), (-5, 10)])}
    result = compile_wall(
        ctx,
        building,
        np.array([5.0, 10.0]),
        np.array([-5.0, 10.0]),
        0.0,
        10.0,
        profile,
        [],
    )

    assert result is not None
    assert result['components'] == 1
    meshes = [item for item in emitted if item[0] == 'mesh']
    assert any(item[2] == 'glass' for item in meshes)
    # A polygon-only record must not be treated as a rectangular opening.
    glass_mesh = next(item[3] for item in meshes if item[2] == 'glass')
    assert len(glass_mesh.vertices) > 0
    assert len(glass_mesh.faces) > 0


def test_material_evidence_resolves_for_panels_and_bands():
    evidence = {
        'family': 'render',
        'finish': 'gray',
        'colorRgb': [12, 34, 56],
        'sourceIds': ['street-photo-1'],
    }
    assert _component_material({'material': 'stucco', 'materialEvidence': evidence}, 'brick') == 'render_gray#0c2238'
    assert _component_material({'materialEvidence': evidence}, 'stone') == 'render_gray#0c2238'


def test_flat_opening_path_remains_rectangular():
    opening = {'rect': [0.1, 0.2, 0.4, 0.8], 'style': {}}
    outline = opening_outline(opening, 10.0, 10.0)
    assert outline.equals(box(1.0, 2.0, 4.0, 8.0))
    assert head_shape(opening['style']) is None
