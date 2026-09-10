import numpy as np
import pytest
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
    tops=[[[0,y,10.05],[6,y,10.05]] for y in (0,4)]
    cloud=np.array([[x,y,10.05] for x in (2.95,3.05) for y in (.2,.8,1.2,1.8,2.2,2.8,3.2,3.8)])
    made, report = recover_roof_coverage(existing, fp, cloud, 5,wall_top_segments=tops)
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


def test_sloped_patch_follows_existing_plane_exactly():
    fp = Polygon([(0, 0), (3.1, 0), (3.1, 4), (0, 4)])
    base = roof(Polygon([(0, 0), (2.9, 0), (2.9, 4), (0, 4)]), 10)
    # Existing cap has a z=10+0.5x plane.
    base.vertices[:, 2] += .5 * base.vertices[:, 0]
    cloud = np.asarray([[x, y, 10.05 + .5*x] for x in (2.95,3.05) for y in (.2,.5,.8,1.2, 1.5, 1.8, 2.2, 2.5, 2.8, 3.2, 3.5, 3.8)])
    tops=[[[0,y,10.05],[3.1,y,11.6]] for y in (0,4)]+[[[3.1,0,11.6],[3.1,4,11.6]]]
    made, report = recover_roof_coverage([base], fp, cloud, 5,wall_top_segments=tops)
    assert made and report["supportedAreaM2"] > 0
    patch = made[0]
    error = patch.vertices[:, 2] - (.5 * patch.vertices[:, 0] + 10.05)
    # Output is intentionally checked at GLB float32 precision.
    assert np.max(np.abs(error[error > -.01])) <= 2e-6
    assert np.min(error) == pytest.approx(-.02, abs=2e-6)


def test_far_disconnected_support_in_missing_domain_is_excluded():
    fp = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    existing = [roof(Polygon([(0, 0), (1.9, 0), (1.9, 4), (0, 4)]), 10)]
    far = points(fp, 10, [(7, 1), (7, 2), (7, 3)])
    made, report = recover_roof_coverage(existing, fp, far, 5)
    assert not made and report["supportedAreaM2"] == 0
