import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pipeline import facade_detail


def render_roof_detail(*, roof_z, kind='mansard', H=12.95, Hs=9.76,
                       roof_shell=False, pitched_roof=False, openings=()):
    emitted = []
    ctx = {
        'block': lambda *args, **kwargs: None,
        'prism': lambda *args, **kwargs: None,
        'beam': lambda *args, **kwargs: None,
        'point': lambda x, z, depth=0.: np.array([x, depth, z]),
        'put': lambda mesh, key, material: emitted.append((key, material, mesh)),
        'detail': 'facade-detail', 'clip': None,
        'low': 0., 'high': 10., 'W': 10., 'Hs': Hs, 'H': H,
        'mat': 'wall', 'wall_mat': lambda x, z: 'wall',
        'roofShell': roof_shell, 'pitchedRoof': pitched_roof,
    }
    facade_detail.wall(ctx, {
        'sourceIds': ['mapillary:test-source'],
        'roofs': [{'x': [0., .6], 'z': roof_z, 'kind': kind,
                   'riseM': 1., 'pitchDeg': 70., 'material': 'roof-detail'}]
    }, list(openings))
    return emitted


@pytest.mark.parametrize('owner', ['roofShell', 'pitchedRoof'])
def test_upper_mansard_opens_only_over_source_measured_windows_when_wall_is_taller(owner):
    # This profile reads its roof at 7.4 m in a 9.76 m source frame, while
    # the host wall extent is 12.95 m. The old .6*H threshold was 7.77 m and
    # let this upper roof duplicate the shell; the source-frame threshold is
    # 5.856 m.
    opening = {'rect': [1., 7.5, 2., 8.1], 'evidence': 'visual-grammar',
               'style': {'kind': 'window', 'surround': {'widthM': .04}}}
    emitted = render_roof_detail(roof_z=7.4 / 9.76,
                                 roof_shell=owner == 'roofShell',
                                 pitched_roof=owner == 'pitchedRoof',
                                 openings=[opening])
    assert len(emitted) == 1
    # The clipped slab retains the mansard cheeks and a hole around the
    # opening, rather than dropping the entire roof reading.
    assert len(emitted[0][2].faces) > 12


def test_midwall_roof_detail_remains_when_shell_owns_only_upper_roof():
    emitted = render_roof_detail(roof_z=2.5 / 9.76, roof_shell=True)
    assert any(key == 'facade-detail' and material == 'roof-detail' for key, material, _ in emitted)


def test_high_gable_keeps_existing_wall_height_ownership_threshold():
    # A high source-frame gable is retained when the previous wall-height
    # predicate did not establish shell overlap; source height alone is not
    # enough to remove a real gable face or its edge trim.
    emitted = render_roof_detail(roof_z=6.0 / 6.95, kind='gable', H=12.38,
                                 Hs=6.95, roof_shell=True)
    assert any(material == 'roof-detail' for _, material, _ in emitted)


@pytest.mark.parametrize('panel', [
    box(0., 0., 10., 10.).difference(box(2., 2., 4., 4.)),
    Polygon([(0., 0.), (10., 0.), (10., 3.), (6., 3.), (6., 8.), (0., 8.)])
        .difference(box(1., 1., 2., 2.)),
])
def test_constrained_triangulation_preserves_projected_clipped_panel_area(panel):
    points, faces = facade_detail._constrained_polygon_triangles(panel)
    triangles = points[faces]
    area = sum(abs(float(np.linalg.det(np.stack((t[1] - t[0], t[2] - t[0]))))) * .5
               for t in triangles)
    assert area == pytest.approx(panel.area, abs=1e-7)
