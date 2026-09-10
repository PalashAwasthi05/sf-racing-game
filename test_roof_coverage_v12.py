import numpy as np
import trimesh
from shapely.geometry import Polygon

from general_world.roof_coverage import recover_roof_coverage


def roof(poly, z, parent=0):
    m = trimesh.creation.extrude_polygon(poly, .1, engine="earcut")
    m.apply_translation([0, 0, z - .05])
    return m


def points(poly, z, cells):
    out = []
    for x, y in cells:
        out += [[x + .2, y + .2, z], [x + .5, y + .5, z + .03], [x + .8, y + .7, z - .03]]
    return np.asarray(out)


def test_deleted_narrow_strip_is_recovered():
    fp = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    existing = [roof(Polygon([(0, 0), (2.9, 0), (2.9, 4), (0, 4)]), 10), roof(Polygon([(3.1, 0), (6, 0), (6, 4), (3.1, 4)]), 10)]
    made, report = recover_roof_coverage(existing, fp, points(fp, 10, [(2.8, 1), (2.8, 2), (2.8, 3)]), 5)
    assert made and report["supportedAreaM2"] > 0


def test_courtyard_hole_is_not_filled():
    fp = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)], [[(2, 2), (4, 2), (4, 4), (2, 4)]])
    made, report = recover_roof_coverage([roof(fp, 10)], fp, np.empty((0, 3)), 5)
    assert not made and report["gapAreaM2"] == 0


def test_different_height_points_do_not_support_plane():
    fp = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    existing = [roof(Polygon([(0, 0), (1.9, 0), (1.9, 4), (0, 4)]), 10)]
    made, report = recover_roof_coverage(existing, fp, points(fp, 14, [(2, 1), (2, 2), (2, 3)]), 5)
    assert not made and report["supportedAreaM2"] == 0


def test_unsupported_void_is_not_filled():
    fp = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    existing = [roof(Polygon([(0, 0), (1.9, 0), (1.9, 4), (0, 4)]), 10)]
    made, report = recover_roof_coverage(existing, fp, np.empty((0, 3)), 5)
    assert not made and report["unresolvedAreaM2"] > 0


def test_exact_no_overlap_existing_is_unresolved():
    fp = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    made, report = recover_roof_coverage([roof(Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]), 10)], fp, points(fp, 10, [(1, 1), (1, 2), (1, 3)]), 5)
    assert not made and report["status"] == "no-overlap-existing"
