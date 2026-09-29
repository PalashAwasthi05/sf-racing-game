"""A low gabled building in front of a tall block under one outline (v0.35).

The outline is split along its own axes, so the tall block takes no strips beside the low building; a pitched roof
keeps its planes (a skylight on one half does not flatten it) and the two halves share one exact ridge; the end wall
under the gable stops at its eaves and the triangle above takes the wall's material. Tiers stand at their measured
height. A hand reading, an envelope skin or a reading filed under a stale part belongs only to the part whose wall it
describes."""
import math
import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import box
from rasterio.transform import from_origin
from pipeline import building_parts, roof_envelope, facades, envelopes, spatial_rules

ANGLE = 45.


def corner(step=.5):
    """24 m wide, 80 m deep outline turned 45 degrees: a gable over the first 8.5 m (eaves 6 m, ridge 10 m, the ridge
    running front to back), a 19 m block behind it."""
    u, v = np.meshgrid(np.arange(.25, 24, step), np.arange(.25, 80, step)); u, v = u.ravel(), v.ravel()
    z = np.where(v < 8.5, 6. + .33 * (12 - np.abs(u - 12)), 19.)
    a = math.radians(ANGLE); x = u * math.cos(a) - v * math.sin(a); y = u * math.sin(a) + v * math.cos(a)
    return affinity.rotate(box(0, 0, 24, 80), ANGLE, origin=(0, 0)), np.c_[x, y, z]


def local(g):
    """Coordinates of a geometry's vertices across the front (u) and back from the front (v)."""
    xy = np.concatenate([np.asarray(q.exterior.coords) for q in getattr(g, 'geoms', [g])]); a = math.radians(-ANGLE)
    return np.c_[xy[:, 0] * math.cos(a) - xy[:, 1] * math.sin(a), xy[:, 0] * math.sin(a) + xy[:, 1] * math.cos(a)]


def test_tall_block_takes_no_strips_beside_the_low_building():
    outline, pts = corner()
    parts = sorted(building_parts.split(outline, pts, 0.), key=lambda q: q.area)
    assert len(parts) == 2
    low, tall = parts
    assert 180 < low.area < 230, low.area
    # the tall part starts behind the low building across the whole width: nothing of it within 7 m of the front
    assert local(tall)[:, 1].min() > 7.5, local(tall)[:, 1].min()
    # and the low building spans the whole front
    lu = local(low); assert lu[:, 0].min() < .3 and lu[:, 0].max() > 23.7


def gable_points(skylight=False, valley=False, step=.5):
    u, v = np.meshgrid(np.arange(.25, 24, step), np.arange(.25, 9, step)); u, v = u.ravel(), v.ravel()
    z = 6. + .33 * (np.abs(u - 12) if valley else 12 - np.abs(u - 12))
    if skylight: z = z + np.where((abs(u - 17) < 1.5) & (abs(v - 4.5) < 1.5), .6, 0.)
    return box(0, 0, 24, 9), np.c_[u, v, z]


def test_skylight_does_not_flatten_a_gable_and_the_halves_share_a_ridge():
    g, pts = gable_points(skylight=True)
    patches, _ = roof_envelope.continuous_patches(g, pts, 1.5, .09, 150., 1.5)
    assert len(patches) == 1, [(p['geometry'].area, p['model']) for p in patches]
    p = patches[0]
    assert p['model']['operation'] == 'min' and len(p['model']['planes']) == 2 and p['geometry'].area > 190
    ridge = roof_envelope.evaluate(p['model'], np.array([[12., 4.5]]))[0]
    assert abs(ridge - 9.96) < .3, ridge


def test_a_valley_is_not_joined_as_a_ridge():
    g, pts = gable_points(valley=True)
    patches, _ = roof_envelope.continuous_patches(g, pts, 1.5, .09, 150., 1.5)
    assert all(len(p['model']['planes']) == 1 for p in patches)


def test_gable_end_wall_stops_at_its_eaves_under_a_pitched_roof():
    along = np.linspace(0, 24, 400); z = 6. + .33 * (12 - np.abs(along - 12))
    flat_rule = facades.wall_level(along, z, 24.)
    pitched = facades.wall_level(along, z, 24., pitched=True)
    assert abs(pitched - 6.) < .25 and flat_rule > pitched + 1., (pitched, flat_rule)
    assert facades.is_gable(facades.wall_profile(along, z, 24.), 24.)
    assert not facades.is_gable(facades.wall_profile(along, np.full(400, 9.), 24.), 24.)


def test_gable_faces_take_the_material_of_the_wall_below():
    mesh = trimesh.creation.box(extents=(10, 6, 2)); mesh.apply_translation((5, 3, 9))
    side = mesh.submesh([np.where(np.abs(mesh.face_normals[:, 2]) <= .05)[0]], append=True)
    records = [{'midpoint': [5., 0.], 'normal': [0., -1.], 'lengthM': 10., 'material': 'brick_modular#b2543c'}]
    groups = roof_envelope.side_materials(side, records, 'stucco')
    south = [i for i, n in enumerate(side.face_normals) if n[1] < -.9]
    assert sorted(groups['brick_modular#b2543c']) == sorted(south) and len(groups['stucco']) == len(side.faces) - len(south)


def tiers(rise_m):
    rr = np.full((20, 20), 33.); rr[:, :9] = 33. + rise_m; known = np.ones_like(rr, bool); tr = from_origin(0, 30, 1.5, 1.5)
    quant = np.maximum(0, np.round((rr - 33.) / 2.5)).astype(np.int16)
    bands = spatial_rules.roof_bands(quant, tr, box(0, 0, 30, 30), 100., simplify_m=1.125, cell_m=1.5, level_step_m=2.5, measured_mask=known)
    return spatial_rules.measured_tiers(bands, rr, known, tr, 33.)


def test_a_tier_stands_at_its_measured_height():
    t = tiers(3.4)   # the rounded bin (2.5 m) had called this storey rooftop plant
    assert len(t) == 1 and abs(t[0]['top'] - 3.4) < .05 and t[0]['bottom'] == 0.
    assert tiers(1.6) == []   # plant and parapets stay part of the roof


def part(cid, g, source):
    return {'id': cid, 'sourceId': 'b', 'partOf': 'b', 'g': g, 'sourceG': source, 'osm': None}


def test_envelope_skin_belongs_to_the_part_at_its_frontage():
    source = box(0, 0, 20, 30); front, back = part('b_p1', box(0, 0, 20, 8), source), part('b', box(0, 8, 20, 30), source)
    skin = {'id': 's', 'match': {'featureId': 'b'}, 'normal': [0., -1.]}
    assert envelopes.matches(front, skin) and not envelopes.matches(back, skin)
    assert envelopes.matches({'id': 'b', 'g': source, 'osm': None}, skin)   # an outline that is not split keeps it


def test_hand_reading_belongs_to_the_part_of_its_height():
    source = box(0, 0, 20, 30); low, tall = part('b_p1', box(0, 0, 20, 8), source), part('b', box(0, 8, 20, 30), source)
    hand = {'match': {'featureId': 'b'}, 'heightRefM': 8.4, 'normal': [0., -1.]}
    assert facades.belongs(hand, low, 6., 7.5) and not facades.belongs(hand, tall, 15., 15.3)
    assert facades.belongs(hand, low, 3.5, 7.5)   # a short tier wall of the low part is still that building
    assert facades.belongs(hand, {'id': 'b', 'g': source, 'osm': None}, 15., 20.)


def test_reading_filed_under_a_stale_part_does_not_follow_it():
    source = box(0, 0, 20, 30); low, tall = part('b_p1', box(0, 0, 20, 8), source), part('b', box(0, 8, 20, 30), source)
    # a side-wall reading (wall x = 0, facing -x) whose frame covers y 0-9.5: 8 m of it on the low part
    reading = {'match': {'featureId': 'b', 'partId': 'b'}, 'normal': [-1., 0.], 'frameU': [-9.5, 0.], 'planeV': 0., 'heightRefM': 15.}
    assert not facades.belongs(reading, tall, 15., 20.)
    assert facades.frame_on_part(reading, low)
