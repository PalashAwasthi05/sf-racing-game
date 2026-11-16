"""Measured shallow arch heads must retain physical curvature after scaling."""
import numpy as np
import pytest
from shapely.geometry import box

from pipeline.facade_grammar import opening_outline


@pytest.mark.parametrize('width,height,rise', [(1., 2.5, .1), (2., 3., .12), (.7, 2.1, .08)])
def test_shallow_head_is_circular_in_world_metres(width, height, rise):
    outline = opening_outline({'rect': [0, 0, 1, 1],
                               'style': {'head': 'segmental', 'riseRatio': rise}}, width, height)
    sag = rise * height
    radius = ((width / 2) ** 2 + sag ** 2) / (2 * sag)
    center = np.array([width / 2, height - radius])
    pts = np.asarray(outline.exterior.coords)
    arc = pts[pts[:, 1] >= height - sag - 1e-8]
    assert np.linalg.norm(arc - center, axis=1) == pytest.approx(radius, abs=1e-8)
    assert outline.is_valid
    assert outline.bounds == pytest.approx((0, 0, width, height), abs=.002)
    # Unlike a flattened ellipse, a shallow circular head does not turn
    # almost vertically upward as it leaves each jamb.
    near_end = sorted(arc, key=lambda p: p[0])[:2]
    slope = (near_end[1][1] - near_end[0][1]) / (near_end[1][0] - near_end[0][0])
    assert slope < 2.


def test_clipping_half_of_measured_arch_keeps_original_crown_and_curvature():
    original = opening_outline({'rect': [.1, .2, .9, .9],
                                'style': {'head': 'segmental', 'riseRatio': .12}}, 3., 4.)
    left, right = original.intersection(box(0, 0, 1.5, 4)), original.intersection(box(1.5, 0, 3, 4))
    assert left.union(right).symmetric_difference(original).area < 1e-8
    assert left.bounds[3] == pytest.approx(right.bounds[3], abs=1e-8)
