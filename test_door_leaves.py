from pipeline.door_leaves import draw_garage_leaf, emit_grille, validate_cells
from regen.specs import _style


def _meshes(emitter):
    meshes = []
    def block(x, z, w, h, d, material, depth):
        meshes.append({'bounds': (x-w/2, z-h/2, x+w/2, z+h/2, depth-d/2, depth+d/2),
                       'material': material})
    emitter(block)
    return meshes


def _ray(meshes, x, z):
    hits = []
    # Ray travels along the wall-depth axis from in front of the facade.
    # These are the actual emitted cuboids, tested against their metric bounds.
    for mesh in meshes:
        x0, z0, x1, z1, depth0, depth1 = mesh['bounds']
        if x0 <= x <= x1 and z0 <= z <= z1 and depth0 <= 1.0:
            hits.append((1.0-depth1, mesh['material']))
    return sorted(hits)


def test_panelled_garage_is_opaque_and_unspecified_panels_are_not_glass():
    meshes = _meshes(lambda b: draw_garage_leaf(b, (0, 0, 3, 2),
                                                {'doorKind': 'panel', 'material': '#ddd'}, depth=-.4))
    assert len(meshes) == 1
    assert meshes[0]['material'] == '#ddd'
    assert _ray(meshes, 1.5, 1.)


def test_explicit_glass_cells_are_local_and_remaining_leaf_stays_opaque():
    cells = [{'rect': [.1, .2, .45, .8], 'kind': 'glass'}]
    meshes = _meshes(lambda b: draw_garage_leaf(b, (0, 0, 3, 2),
                                                {'doorKind': 'panel', 'material': '#ddd', 'doorCells': cells}, depth=-.4))
    assert any(m['material'] == 'glass' for m in meshes)
    # The pane is a real aperture: its ray hits glass only, while an untyped
    # cell remains the opaque leaf finish.
    assert [material for _, material in _ray(meshes, .8, 1.)] == ['glass']
    assert _ray(meshes, 2., 1.)[0][1] == '#ddd'


def test_grille_has_clear_bar_gaps_with_recessed_backing():
    meshes = _meshes(lambda b: emit_grille(b, (0, 0, 1, 2), depth=-.4, pitch=.14))
    # At the center height and between the regularly spaced bars, the first
    # intersection is the recessed dark backing, not a solid leaf at the gate.
    hits = _ray(meshes, .0625, 1.)
    assert hits and hits[0][1] == 'charcoal'
    assert hits[0][0] > 1.45
    # A bar position has a nearer metal intersection.
    bar_hits = _ray(meshes, .125, 1.)
    assert bar_hits and bar_hits[0][1] == 'charcoal'
    assert bar_hits[0][0] < hits[0][0]


def test_cells_reject_invalid_or_untyped_rectangles():
    for cells in ([{'rect': [0, 0, 1, 1]}], [{'rect': [0, 0, 2, 1], 'kind': 'glass'}]):
        try:
            validate_cells(cells)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid observed door-cell layout was accepted')


def test_style_preserves_grille_and_typed_door_cells_without_default_glass_grid():
    grille = _style({'kind': 'entry', 'frame': '#aaaaaa', 'door': {'kind': 'grille'}})
    assert grille['door']['kind'] == 'grille'
    entry = _style({'kind': 'entry', 'frame': '#aaaaaa',
                    'door': {'kind': 'panel', 'cells': [{'rect': [.1, .1, .9, .9], 'kind': 'glass'}]}})
    assert entry['door']['kind'] == 'panel'
    assert entry['door']['cells'][0]['kind'] == 'glass'

    garage = _style({'kind': 'garage', 'door': {'kind': 'panel', 'hex': '#dddddd'}})
    assert garage['doorKind'] == 'panel'
    assert 'doorCells' not in garage and 'glazed' not in garage
    assert 'slatPitchM' not in garage
