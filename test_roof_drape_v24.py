"""A continuous measured slope becomes one draped surface closed by skirts, not a stack of tiers; a flat roof drapes nothing."""
import numpy as np
from shapely.geometry import box
from pipeline import roof_envelope

def test_bowl_slope_is_draped_and_closed():
    rows, cols = 30, 40
    rr = np.tile(np.linspace(8, 30, cols), (rows, 1))          # rises 22 m over 60 m: 0.37 m/m
    known = np.ones_like(rr, bool)
    keep, area = roof_envelope.drape_regions(rr, known, box(0, 0, 60, 45), 0, 45, 1.5)
    assert keep.mean() > .8 and area.area > 1500
    surface, skirt = roof_envelope.drape_mesh(rr, keep, 0, 45, 1.5, 8.)
    assert surface.bounds[1][2] > 28 and skirt is not None and skirt.bounds[0][2] == 8.

def test_flat_roof_drapes_nothing():
    rr = np.full((30, 40), 12.) + np.random.default_rng(1).normal(0, .03, (30, 40))
    keep, area = roof_envelope.drape_regions(rr, np.ones_like(rr, bool), box(0, 0, 60, 45), 0, 45, 1.5)
    assert area is None and not keep.any()

def test_flat_roof_with_plant_drapes_nothing():
    rng = np.random.default_rng(3)
    rr = np.full((30, 40), 12.) + rng.normal(0, .15, (30, 40))
    for i, j in rng.integers(2, 28, (12, 2)): rr[i:i + 2, j:j + 3] += rng.uniform(1.5, 3.5)   # plant boxes
    keep, area = roof_envelope.drape_regions(rr, np.ones_like(rr, bool), box(0, 0, 60, 45), 0, 45, 1.5)
    assert area is None, area.area
