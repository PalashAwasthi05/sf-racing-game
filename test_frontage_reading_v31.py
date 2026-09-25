"""Frontage reading v31: colour zones by height, and the dark-photo guard. Synthetic images only."""
import numpy as np
import auto_read_frontages_v31 as V
from auto_read_frontages_v24 import PPM
from judge_facades_v27 import to_lab


def image(H, L, split, low, high):
    """A wall H x L metres: `low` colour below `split` m, `high` above (rows run top to bottom)."""
    img = np.zeros((int(H * PPM), int(L * PPM), 3), float); rows = np.arange(img.shape[0])
    z = H - (rows + .5) / PPM; img[z < split] = low; img[z >= split] = high
    return img


def test_two_tone_wall_gives_two_zones_and_one_colour_stays_one():
    H, L = 10., 12.
    img = image(H, L, 4., (225, 222, 214), (80, 74, 58)); lab = to_lab(img); M = np.ones(img.shape[:2], bool)
    zs = V.zones_of(lab, M, H)
    assert len(zs) == 2 and abs(zs[0][1] - 4.) <= .5, [(z[0], z[1]) for z in zs]
    img = image(H, L, 4., (180, 120, 100), (182, 121, 101)); zs = V.zones_of(to_lab(img), M, H)
    assert len(zs) == 1


def test_dark_reading_proves_neither_colour_nor_blankness():
    p = {'materialEvidence': {'family': 'metal-panel', 'finish': 'dark', 'colorRgb': [12, 11, 5], 'sourceIds': ['x']}, 'openings': [], 'groups': [], 'reading': {'blank': True}}
    q = V.guard_dark(p)
    assert 'materialEvidence' not in q and q.get('materialOnly') and 'blank' not in q['reading'] and q['reading']['darkPhoto']
    p = {'materialEvidence': {'family': 'render', 'finish': 'white', 'colorRgb': [214, 210, 206], 'sourceIds': ['x']}, 'reading': {}}
    assert V.guard_dark(p)['materialEvidence']['colorRgb'] == [214, 210, 206]
