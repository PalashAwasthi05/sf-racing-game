import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

from pipeline import bays
from pipeline.facade_grammar import measured_bay_extent


def test_measured_bay_keeps_all_rows_above_short_host_return_and_closes_back():
    width, frame_height, host_top = 6.8, 12.0, 6.73
    extent = measured_bay_extent({'rect': [.2, .2, .6, .85]}, width, frame_height)
    assert extent['z1'] == 10.2
    assert extent['z1'] > host_top
    bay = {**extent, 'p': .45, 'shape': 'square', 'hostTopM': host_top,
           'hostMaterial': 'host-wall', 'cap': 'flat'}
    openings = []
    for lo, hi in ((3.0, 4.8), (6.9, 8.7), (9.2, 9.9)):
        outline = box(1.8, lo, 2.5, hi)
        openings.append({'rect': list(outline.bounds), '_outline': outline,
                         'style': {'frameMaterial': 'frame', 'sash': 'fixed'}})

    emitted = []
    point = lambda x, z, d: np.array([x, d, z], dtype=float)
    windows = bays.draw(point, lambda mesh, key, material: emitted.append((mesh, key, material)),
                        'wall', 'detail', bay, openings, 'bay-wall', 'trim')

    glass = [mesh for mesh, _, material in emitted if material == 'glass']
    assert windows == 3 and len(glass) == 3
    assert max(float(mesh.bounds[1, 2]) for mesh in glass[1:]) > host_top

    backing = [mesh for mesh, _, material in emitted if material == 'host-wall']
    assert backing
    back = max(backing, key=lambda mesh: mesh.bounds[1, 2])
    assert back.bounds[0, 2] >= host_top
    assert back.bounds[1, 2] == extent['z1']
    # The wall plane is opaque behind upper panes. It stays behind the pane so
    # an outside ray reaches glass first, while an inside ray cannot see through
    # the upper bay into the sky.
    assert back.bounds[0, 1] < 0.0 and back.bounds[1, 1] <= 1e-8
    aperture_center = Point(2.15, 7.8)
    back_faces = [Polygon(tri[:, [0, 2]]) for tri in back.vertices[back.faces]
                  if np.all(np.abs(tri[:, 1]) < 1e-7)]
    assert unary_union(back_faces).covers(aperture_center)
    pane_at_center = [mesh for mesh in glass if mesh.bounds[0, 0] <= 2.15 <= mesh.bounds[1, 0]
                      and mesh.bounds[0, 2] <= 7.8 <= mesh.bounds[1, 2]]
    assert pane_at_center and min(float(mesh.bounds[0, 1]) for mesh in pane_at_center) > back.bounds[1, 1]


def test_measured_bay_below_host_return_needs_no_extra_backing():
    extent = measured_bay_extent({'rect': [.1, .2, .5, .6]}, 5., 10.)
    assert extent == {'x0': .5, 'x1': 2.5, 'z0': 2., 'z1': 6.}
