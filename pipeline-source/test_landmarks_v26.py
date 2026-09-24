"""Landmark layer: an explicit, keyed exception that never leaks into the general rules.

Location-free where it measures: synthetic rasters and point sets stand in for the survey. The registry test reads the
real registry only to check that it claims nothing but its own keys.
"""
import math
import numpy as np
import shapely
from scipy import ndimage as ndi
from shapely.geometry import box, Polygon
from pipeline import landmarks
from pipeline.landmarks import diamond, ballpark


def test_registry_claims_only_its_own_footprints():
    key = next(iter(landmarks.REGISTRY['landmarks']))
    assert landmarks.entry({'id': key}) is not None
    assert landmarks.entry({'id': key + '_p1', 'sourceId': key}) is not None          # a split part joins by source
    assert landmarks.entry({'id': 'any_other_outline'}) is None
    assert landmarks.entry({'id': 'x_p1', 'sourceId': 'any_other_outline'}) is None


def test_frontage_view_keeps_facade_planes_only():
    resolved = [{'frame': {}, 'profile': {'id': 'p', 'replaceDepthM': 35, 'assemblies': [
        {'op': 'bracedFrame', 'rearDepthM': 17, 'z': [0, 1]}, {'op': 'lightBank', 'z': [1, 2]}]},
        'nodes': [{'id': 'root', 'parentId': None, 'parentDepthM': -35}, {'id': 'child', 'parentId': 'root', 'parentDepthM': 0}]}]
    view = landmarks.frontage_view(resolved, {'frontageDepthM': 3})
    p = view[0]['profile']
    assert p['replaceDepthM'] == 3 and [a['op'] for a in p['assemblies']] == ['bracedFrame'] and 'rearDepthM' not in p['assemblies'][0]
    assert view[0]['nodes'][0]['parentDepthM'] == -3 and view[0]['nodes'][1]['parentDepthM'] == 0
    assert resolved[0]['profile']['replaceDepthM'] == 35 and len(resolved[0]['profile']['assemblies']) == 2   # original untouched
    assert resolved[0]['nodes'][0]['parentDepthM'] == -35


def synthetic_field(plate, bearing, base_paths=True, foul_cutouts=True, seed=3):
    xs, ys = np.meshgrid(np.arange(-20, 80, .5) + .25, np.arange(-50, 50, .5) + .25)
    t = diamond.template(plate, bearing, xs, ys, base_paths, foul_cutouts)
    rng = np.random.default_rng(seed)
    I = np.where(np.nan_to_num(t, nan=0.) > .5, 30., 60.) + rng.normal(0, 6, xs.shape)     # dirt reads darker than grass
    return I, xs, ys


def test_fit_recovers_plate_and_bearing():
    plate, bearing = np.array([5.3, -2.1]), math.radians(12.)
    I, xs, ys = synthetic_field(plate, bearing)
    got_p, got_b, score = diamond.fit(I, xs, ys, plate + (2., -1.5), bearing + math.radians(9))
    assert np.linalg.norm(got_p - plate) < .3, got_p
    assert abs(math.degrees(got_b - bearing)) < 1., math.degrees(got_b)
    assert score > 2.


def test_variant_reads_base_paths_and_foul_cutouts():
    plate, bearing = np.array([0., 0.]), math.radians(-20.)
    for paths, cut in ((True, False), (False, True), (False, False)):
        I, xs, ys = synthetic_field(plate, bearing, paths, cut)
        assert diamond.variant(I, xs, ys, plate, bearing) == (paths, cut)


def test_surface_keeps_decks_and_drops_masts():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 40, (60000, 2))
    z = np.where(xy[:, 0] > 20, 12., 0.)                                     # a deck 12 m up beside a field
    mast = (np.abs(xy[:, 0] - 10) < .4) & (np.abs(xy[:, 1] - 20) < .4)       # a mast 30 m tall on the field
    z[mast] = 30.
    grid, g = ballpark.surface_grid(np.c_[xy, z], (0, 0, 40, 40), .5)
    assert abs(ballpark.sample(grid, g, 30, 20) - 12) < .01                    # deck kept
    assert abs(ballpark.sample(grid, g, 10, 20)) < .01                         # mast gone
    assert abs(ballpark.sample(grid, g, 25, 20) - 12) < .01 and abs(ballpark.sample(grid, g, 15, 20)) < .01   # step kept sharp


def test_standing_clusters_group_what_stands_above_the_roofs():
    rng = np.random.default_rng(2)
    roof = np.c_[rng.uniform(0, 50, (5000, 2)), np.full(5000, 30.)]
    rig = np.c_[rng.uniform(10, 25, 400), rng.uniform(20, 21, 400), rng.uniform(50, 55, 400)]
    far = np.c_[rng.uniform(40, 41, 200), rng.uniform(5, 6, 200), rng.uniform(60, 64, 200)]
    P = np.r_[roof, rig, far]
    mask, groups = ballpark.standing_clusters(P, box(0, 0, 50, 50), 40.)
    assert mask.sum() == 600 and sorted(len(gr) for gr in groups) == [200, 400]


def test_triangulation_stays_inside_the_outline_and_faces_up():
    poly = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)], [[(10, 10), (20, 10), (20, 20), (10, 20)]])
    V, F = ballpark.triangulate(poly, .5)
    tris = shapely.polygons(V[F])
    assert shapely.covered_by(tris, poly.buffer(.01)).all()
    a = V[F[:, 1]] - V[F[:, 0]]; b = V[F[:, 2]] - V[F[:, 0]]
    assert (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0).all()
    assert abs(shapely.area(tris).sum() - poly.area) < 1e-6


def test_plateaus_level_roofs_and_keep_rakes():
    step = .5; yy, xx = np.mgrid[0:80, 0:80] * step
    H = np.where(xx < 20, 12., 12. + (xx - 20) * .5)                         # a roof at 12 m beside a 27 deg rake
    H[10:14, 10:14] += 1.2                                                    # an umbrella on the roof
    H += np.random.default_rng(4).normal(0, .03, H.shape)
    level, flat = ballpark.plateaus(ndi.median_filter(H, 5), step)
    assert np.ptp(level[:, 2:36][flat[:, 2:36]]) < 1e-9 and flat[:, 2:36].mean() > .9    # roof one exact level
    assert abs(level[12, 12] - 12) < .05                                      # umbrella gone
    assert not flat[:, 45:75].any()                                           # the rake is not levelled
