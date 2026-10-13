"""Street surfaces follow the ground under the carriageway, not raised yards beside it (pipeline/street_ground). Synthetic."""
import sys, pathlib
import numpy as np
from shapely.geometry import box
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import street_ground


def _hill():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-60, 60, (60000, 2))
    z = .2 * xy[:, 0]                                   # a 20% grade running east
    z = z + np.where(xy[:, 1] > 8, 1.5, 0.)             # lots north of the street stand 1.5 m up behind a wall
    return np.c_[xy, z], box(-60, -4, 60, 4)            # an 8 m carriageway along y = 0


def test_a_sidewalk_beside_a_raised_lot_stays_at_its_kerb():
    ground, road = _hill(); step = 3.; lo = -60.; n = 41
    gx, gy = np.meshgrid(np.arange(n) * step + lo, np.arange(n) * step + lo)
    terrain = .2 * gx + np.where(gy > 6, 1.5, 0.)       # a coarse grid that has already smeared the wall
    elev, held = street_ground.build(ground, road, terrain, lo, step, n)
    x = np.linspace(-40, 40, 9)
    for y in (5., 7.):                                   # the sidewalk, 1-3 m from the kerb
        assert np.abs(elev(x, np.full_like(x, y)) - .2 * x).max() < .15
    assert np.abs(elev(x, np.full_like(x, 30.)) - (.2 * x + 1.5)).max() < .05   # the lot keeps its own height
    near = np.abs(gy) < 10
    assert (held[near] <= np.vectorize(lambda a, b: float(elev(a, b)))(gx[near], gy[near]) + 1e-6).all()
