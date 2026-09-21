"""Location-free regressions for measured, nested roof bands."""
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from pipeline.spatial_rules import roof_bands


TRANSFORM = from_origin(0, 30, 1.5, 1.5)
FOOTPRINT = box(0, 0, 45, 30)


def test_integer_level_bins_return_physical_heights():
    q = np.zeros((20, 30), np.int16)
    q[4:12, 6:18] = 1
    q[6:10, 8:16] = 2
    bands = roof_bands(q, TRANSFORM, FOOTPRINT, minimum_area=25, simplify_m=1.125,
                       cell_m=1.5, level_step_m=2.5)
    assert [(b["bottom"], b["top"]) for b in bands] == [(0.0, 2.5), (2.5, 5.0)]


def test_empty_wide_strip_does_not_promote_the_parent():
    q = np.zeros((20, 30), np.int16)
    q[4:8, 4:8] = 1
    q[4:8, 16:20] = 1
    measured = q > 0
    bands = roof_bands(q, TRANSFORM, FOOTPRINT, minimum_area=10, simplify_m=1.125,
                       cell_m=1.5, measured_mask=measured)
    assert len(bands) == 1
    g = bands[0]["geometry"]
    assert g.geom_type == "MultiPolygon" and len(g.geoms) == 2
    assert g.area < FOOTPRINT.area * 0.25
    assert not g.contains(Point(22.5, 15.0))


def test_measured_mask_keeps_an_unmeasured_channel_open():
    q = np.ones((20, 20), np.int16)
    measured = np.ones_like(q, bool)
    measured[:, 9:11] = False
    bands = roof_bands(q, from_origin(0, 30, 1.5, 1.5), box(0, 0, 30, 30),
                       minimum_area=10, simplify_m=1.125, cell_m=1.5,
                       measured_mask=measured)
    assert len(bands) == 1
    g = bands[0]["geometry"]
    assert len(g.geoms) == 2
    assert not g.contains(Point(14.25, 15.0))


def test_attached_narrow_strip_can_snap_but_disconnected_cell_cannot():
    q = np.zeros((20, 30), np.int16)
    q[4:12, 6:18] = 1
    q[12, 6:18] = 1       # attached one-cell outline drift
    q[15, 25] = 1         # disconnected one-cell noise
    measured = q > 0
    bands = roof_bands(q, TRANSFORM, FOOTPRINT, minimum_area=25, simplify_m=1.125,
                       cell_m=1.5, measured_mask=measured)
    assert len(bands) == 1
    g = bands[0]["geometry"]
    assert g.area > 200 and g.area < FOOTPRINT.area * 0.35
    assert not g.contains(Point(38.25, 6.75))
