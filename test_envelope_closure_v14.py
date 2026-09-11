"""Envelope closure: a replaced frontage must stay a closed, visible shell.

Location-free: one synthetic footprint, one synthetic profile. Three defects
this guards against were measured in the v0.13 world at the user's viewpoint
(scan_frame_culled_v14.py): return sheets wound toward their own building and
culled by the viewer, host walls cut wider than the emitted skin, and host roof
removed over skins that never reach it.
"""
import numpy as np
from shapely.geometry import box, Point, Polygon
from pipeline import envelopes, facades

PROFILE = {'id': 'closure-test', 'match': {'featureId': 'test'}, 'normal': [0, 1], 'replaceDepthM': 4, 'levels': {},
           'nodes': [{'id': 'low-shop', 'span': [0.2, 0.6], 'z': [0, 6], 'material': 'stone', 'depthM': 0,
                      'children': [{'id': 'pier', 'rect': [0.1, 0, 0.3, 1], 'offsetM': 0.6},
                                   {'id': 'niche', 'rect': [0.5, 0.1, 0.9, 0.8], 'offsetM': -0.9}]},
                     {'id': 'tall-end', 'span': [0.6, 1.0], 'z': [0, 12], 'material': 'stone', 'depthM': 0}]}
FOOT = box(0, -10, 20, 0)          # frontage along y=0, facing +y (the street is at y>0)
HOST_HEIGHT = 9.0                  # host cap: low-shop (6 m) stays under it, tall-end (12 m) replaces it


def compile_profile():
    meshes = []
    ctx = {'put': lambda m, k, c: meshes.append((m, k, c)), 'cube': lambda *a: None, 'cylinder': lambda *a: None}
    c = {'id': 'test', 'g': FOOT}
    resolved = envelopes.prepare({}, c, 0, np.empty((0, 3)), [PROFILE])
    envelopes.compile(ctx, c, 0, resolved)
    return resolved, meshes


def sheets(meshes, key='buildings'):
    return [m for m, k, _ in meshes if k.startswith(key) and len(m.faces) == 2]


# frame() gives normal [0,1] the tangent u=(-1,0): frame x=0 is world x=20. In world x the low-shop box
# (span .2-.6) is 8..16 and tall-end (span .6-1) is 0..8; the street is at y>0, the slot reaches y=-4.


def test_root_returns_face_away_from_their_box():
    resolved, meshes = compile_profile()
    checked = 0
    for m in sheets(meshes):
        n = m.face_normals[0]; c = m.triangles_center[0]
        if abs(n[0]) > .99 and -4 < c[1] < -1.0:                  # a side return of a root box (spans y -4..-.24), not a niche wall
            centre = 12.0 if c[0] > 10 else 4.0                   # which box it belongs to
            assert np.sign(n[0]) == np.sign(c[0] - centre), (c, n)
            checked += 1
    assert checked >= 3, checked                                  # x=16 (low-shop), x=0 and x=8 above 6 m (tall-end)


def test_pier_return_faces_out_and_niche_return_faces_in():
    resolved, meshes = compile_profile()
    got = {'pier': [], 'niche': []}
    for m in sheets(meshes):
        n = m.face_normals[0]; c = m.triangles_center[0]
        if abs(n[0]) < .99 or c[2] > 6.5: continue
        if 0.0 < c[1] < 0.36:  got['pier'].append((round(c[0], 2), np.sign(n[0])))    # pier protrudes to +0.6 (slab .36..0.6): sheet in (0, .36)
        if -0.9 < c[1] < -0.24: got['niche'].append((round(c[0], 2), np.sign(n[0])))  # niche recessed to -0.9: sheet in (-0.9, -0.24)
    assert got['pier'] and got['niche'], got
    for x, sgn in got['pier']:
        assert sgn == (1 if x > 14.4 else -1), got['pier']         # pier spans world x 13.6..15.2; each side faces away from it
    for x, sgn in got['niche']:
        assert sgn == (-1 if x > 10.4 else 1), got['niche']        # niche spans world x 8.8..12; each wall faces into the niche


def test_every_root_box_has_a_rear_wall_facing_the_building():
    resolved, meshes = compile_profile()
    rear = [m for m in sheets(meshes) if np.allclose(m.triangles_center[0][1], -4, atol=1e-6)]
    assert len(rear) == 2, len(rear)
    for m in rear:
        assert m.face_normals[0][1] < -.99


def test_host_cut_matches_emitted_skin_not_the_padded_frame():
    resolved, _ = compile_profile()
    mask = envelopes.replacement_mask(resolved)
    x0, y0, x1, y1 = mask.bounds
    assert abs(x0 - 0.0) < 1e-6 and abs(x1 - 16.0) < 1e-6, mask.bounds   # skins span 20%..100% of the frontage: world x 0..16
    assert not mask.contains(Point(16.1, -1)) and mask.contains(Point(15.9, -1))


def test_roof_kept_over_low_skin_removed_over_tall_skin():
    resolved, _ = compile_profile()
    roof = envelopes.roof_mask(resolved, HOST_HEIGHT)
    assert not roof.contains(Point(12, -1))     # low-shop top (6 m) is under the 9 m cap: roof stays
    assert roof.contains(Point(4, -1))          # tall-end top (12 m) is the roof there


def test_host_band_closes_front_between_low_skin_and_cap():
    resolved, _ = compile_profile()
    spans = envelopes.coverage_spans(resolved, 0, HOST_HEIGHT)
    cubes = []
    ctx = {'cube': lambda center, size, key, mat, angle=0: cubes.append((np.round(center, 3), np.round(size, 3), key)),
           'cylinder': lambda *a: None, 'polys': lambda g: [g], 'lines': envelopes._lines,
           'envelopeMask': envelopes.replacement_mask(resolved), 'envelopeSpans': spans,
           'all_buildings': FOOT, 'boundary': Point(0, 0).buffer(500)}
    c = {'id': 'test', 'g': FOOT, 'osm': None, 'full_area': FOOT.area}
    facades.compile(ctx, c, 0, HOST_HEIGHT, HOST_HEIGHT, {}, [])
    bands = [(ce, sz) for ce, sz, key in cubes if key == 'buildings__test' and abs(ce[1] + .19) < 1e-3 and 8 < ce[0] < 16]
    assert bands, cubes[:5]
    ce, sz = bands[0]
    assert abs(ce[2] - 7.5) < 1e-6 and abs(sz[2] - 3.0) < 1e-6, bands   # band from 6 m (skin top) to 9 m (cap)
    assert abs(sz[0] - 8.0) < .05, bands                                   # exactly the low-shop run, split from tall-end's
    tall = [(ce, sz) for ce, sz, key in cubes if key == 'buildings__test' and 0 < ce[0] < 8 and abs(ce[1] + .19) < 1e-3]
    assert not tall, tall                                                  # tall-end owns the full host height: no band
