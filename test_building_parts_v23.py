"""One outline holding a low and a tall building splits at the step; a pitched roof or a penthouse does not."""
import numpy as np
from shapely.geometry import box
from pipeline import building_parts

def grid(fn, w=40, d=60, step=.5):
    xs, ys = np.meshgrid(np.arange(.25, w, step), np.arange(.25, d, step))
    return np.c_[xs.ravel(), ys.ravel(), fn(xs.ravel(), ys.ravel())]

def test_low_and_tall_blocks_under_one_outline_become_two_parts():
    pts = grid(lambda x, y: np.where(y < 20, 8., 19.))
    parts = building_parts.split(box(0, 0, 40, 60), pts, 0.)
    assert len(parts) == 2 and sorted(round(p.area) for p in parts) == [800, 1600], [p.area for p in parts]

def test_pitched_roof_stays_one_building():
    pts = grid(lambda x, y: 8. + 10. * (1 - abs(x - 20) / 20))     # ridge 10 m above the eaves, smooth
    assert len(building_parts.split(box(0, 0, 40, 60), pts, 0.)) == 1

def test_penthouse_inside_the_roof_is_not_a_building():
    pts = grid(lambda x, y: np.where((abs(x - 20) < 6) & (abs(y - 30) < 6), 22., 15.))
    assert len(building_parts.split(box(0, 0, 40, 60), pts, 0.)) == 1
