"""Frontage reading v27: detections become an elevation a person would draw. Synthetic detections only.

Windows fall into storeys and shared columns; a stretch hidden by a tree continues a regular storey, a plain stretch
that is seen does not; storeys hidden below the lowest seen one continue downward; transoms above a shopfront belong
to the ground storey.
"""
import numpy as np
from auto_read_frontages_v27 import regularise, PPM


def occluder(H, L, boxes=()):
    occ = np.zeros((int(H * PPM), int(L * PPM)), bool)
    for x0, z0, x1, z1 in boxes: occ[int((H - z1) * PPM):int((H - z0) * PPM), int(x0 * PPM):int(x1 * PPM)] = True
    return occ


def storey(z0, xs, w=1.2, h=1.6, jitter=0.):
    rng = np.random.default_rng(int(z0 * 10))
    return [(x - w / 2 + rng.normal(0, jitter), z0 + rng.normal(0, jitter), x + w / 2 + rng.normal(0, jitter), z0 + h + rng.normal(0, jitter), 'window') for x in xs]


def test_storeys_columns_and_ground_storey():
    H, L = 18., 20.; xs = [2, 5, 8, 11, 14, 17]
    ops = storey(5.5, xs, jitter=.08) + storey(9.2, xs, jitter=.08) + storey(12.9, xs, jitter=.08)
    ops += [(1., .2, 4., 3.4, 'window'), (6., .2, 9., 3.4, 'door'), (1.2, 3.7, 3.8, 4.4, 'window')]   # shopfronts and a transom
    gh, rows, gops = regularise(ops, occluder(H, L), H, L)
    assert 4.3 <= gh <= 5.3, gh                                   # the transom does not lower the ground storey
    assert len(rows) == 3 and all(len(r[2]) == 6 for r in rows)
    centres = [[round((a + b) / 2, 2) for a, b in r[2]] for r in rows]
    assert centres[0] == centres[1] == centres[2]                 # snapped to shared columns
    assert len(gops) == 2 and {g[4] for g in gops} == {'shopfront', 'entry'}


def test_hidden_stretch_filled_seen_stretch_left_plain():
    H, L = 14., 20.; xs = [2, 5, 8, 11, 14, 17]
    ops = storey(5.5, [2, 5, 14, 17]) + storey(9.2, [2, 5, 8, 11, 14, 17])
    gh, rows, _ = regularise(ops, occluder(H, L, [(6.5, 4., 12.5, 8.)]), H, L)      # a tree hides the gap of the lower storey
    assert len(rows[0][2]) == 6, rows[0][2]
    gh, rows, _ = regularise(ops, occluder(H, L), H, L)                             # seen and plain: stays plain
    assert len(rows[0][2]) == 4, rows[0][2]


def test_hidden_storeys_continue_downward_and_blank_walls_stay_blank():
    H, L = 20., 20.; xs = [2, 5, 8, 11, 14, 17]
    ops = storey(12.9, xs) + storey(16.6, xs)
    gh, rows, _ = regularise(ops, occluder(H, L, [(0., 4.5, 20., 12.)]), H, L)     # trees hide the storeys below
    assert len(rows) >= 4 and rows[0][0] < 7., [r[0] for r in rows]
    gh, rows, gops = regularise([], occluder(H, L), H, L)
    assert rows == [] and gops == []


def test_single_storey_wall_is_all_ground_storey():
    H, L = 5., 24.
    ops = [(1., .3, 10., 4.4, 'window'), (15., .2, 16.2, 2.4, 'door')]
    gh, rows, gops = regularise(ops, occluder(H, L), H, L)
    assert gh >= 4.5 and rows == []
    assert any(g[4] == 'shopfront' and g[3] - g[1] > 3.5 for g in gops), gops
