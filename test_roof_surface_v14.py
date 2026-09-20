"""Roof heights come from roof surfaces; walls never climb a roof slope; sloped fits need extent.

Synthetic, location-free. Guards the two defects reported on 2026-09-15: a wall raised into a sloped
roof (window grid on a diagonal) and a rooftop structure read as a roof tier and then as a gabled roof.
"""
import numpy as np
from pipeline.spatial_rules import surface_level, surface_raster
from pipeline.facades import wall_level

rng = np.random.default_rng(3)


def test_surface_level_ignores_sparse_tall_structure_above_dense_roof():
    roof = rng.normal(10.0, .05, 24); sign = np.linspace(11.5, 16.0, 7)      # billboard face returns above the roof
    level = surface_level(np.r_[roof, sign])
    assert abs(level - 10.0) < .2, level


def test_surface_level_keeps_a_sloped_cell_and_a_dense_upper_step():
    slope = np.linspace(9.8, 10.2, 20)                     # 15 degrees over a 1.5 m cell
    assert abs(surface_level(slope) - 10.16) < .1
    step = np.r_[rng.normal(10, .04, 8), rng.normal(13, .04, 12)]   # parapet/upper roof with many returns
    assert abs(surface_level(step) - 13.0) < .15


def test_surface_raster_drops_structure_returns_from_height_statistics():
    pts = np.array([[x, y, 10 + rng.normal(0, .05)] for x in np.arange(.2, 9, .35) for y in np.arange(.2, 9, .35)])
    sign = np.array([[4.5, y, z] for y in np.arange(3, 6, .3) for z in np.linspace(11.5, 17, 6)])   # thin tall sign
    allp = np.r_[pts, sign]
    rr, keep = surface_raster(allp, 0.0, 9.0, 1.5, 6, 6)
    assert np.nanmax(rr) < 10.5, np.nanmax(rr)
    assert np.quantile(allp[keep, 2], .97) < 10.5
    assert (~keep[len(pts):]).all()                       # every sign return excluded


def test_wall_level_parapet_partial_bay_and_slope():
    along = rng.uniform(0, 30, 1500)
    flat = np.full_like(along, 12.0) + rng.normal(0, .03, along.size)
    assert abs(wall_level(along, flat, 30.0) - 12.0) < .1
    partial = np.where(along < 18, 8.0, 12.0) + rng.normal(0, .03, along.size)   # taller bay over 40% of the wall
    assert abs(wall_level(along, partial, 30.0) - 8.0) < .15
    sloped = 6.0 + along / 30.0 * 7.0 + rng.normal(0, .03, along.size)           # shed roof 6 -> 13 m along the wall
    assert wall_level(along, sloped, 30.0) < 6.6                                 # eave, not the ridge


def test_sloped_fit_on_minor_cap_is_rejected():
    import trimesh
    from shapely.geometry import box
    from general_world import scene_roofs
    # one big flat roof at 10 m with a small 96 m2 tier at 14 m whose returns form a ridge
    big = trimesh.creation.extrude_polygon(box(0, 0, 40, 30), .1); big.apply_translation([0, 0, 9.9])
    small = trimesh.creation.extrude_polygon(box(15, 10, 25, 19.6), .1); small.apply_translation([0, 0, 13.9])
    roof = trimesh.util.concatenate([big, small]); roof.metadata = {'layer': 'roofs', 'featureId': 'b'}
    cloud = np.array([[x, y, 10 + rng.normal(0, .03)] for x in np.arange(.5, 40, .7) for y in np.arange(.5, 30, .7)])
    ridge = np.array([[x, y, 14 + 1.2 - abs(x - 20) * .24 + rng.normal(0, .03)] for x in np.arange(15.3, 25, .6) for y in np.arange(10.3, 19.6, .6)])
    scene = trimesh.Scene(); scene.add_geometry(roof, geom_name='roofs__b___roofing', node_name='roofs__b___roofing')
    report = scene_roofs.repair_scene(scene, np.r_[cloud, ridge])
    rejected = [r for r in report['records'] if r.get('repairRejection') == 'sloped-fit-on-minor-cap']
    assert report['capsReplaced'] == 0
    assert rejected or all(not r['accepted'] or not r.get('sloped') for r in report['records']), report['records']


def test_tiers_narrower_than_a_roof_volume_are_dropped():
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from pipeline.spatial_rules import roof_bands
    quant = np.zeros((20, 30), np.int16)
    quant[8:9, 5:25] = 3           # a 1.5 m wide, 30 m long raised strip: a sign or parapet, not a tier
    quant[12:18, 10:20] = 3        # a 9 x 15 m block: a real tier
    bands = roof_bands(quant, from_origin(0, 30, 1.5, 1.5), box(0, 0, 45, 30), minimum_area=25)
    assert len(bands) == 1
    g = bands[0]['geometry']
    assert g.contains(box(16, 4, 28, 10).centroid) and not g.intersects(box(7.5, 17, 37.5, 18.5).buffer(-.2))


def test_frontage_shared_by_low_wing_and_tall_block_is_split():
    from pipeline.facades import split_at_roof_steps, wall_level
    along = rng.uniform(0, 40, 2400)
    z = np.where(along < 24, 9.0, 21.0) + rng.normal(0, .04, along.size)
    pieces = split_at_roof_steps((0, 0), (40, 0), along, z)
    assert len(pieces) == 2, pieces
    (a0, b0), (a1, b1) = pieces
    assert abs(b0[0] - 24) < 2.6, pieces                       # cut within one bin of the step
    low = along < b0[0]
    assert abs(wall_level(along[low], z[low], b0[0]) - 9.0) < .2
    assert abs(wall_level(along[~low] - b0[0], z[~low], 40 - b0[0]) - 21.0) < .2


def test_diagonal_tier_outline_follows_the_building_not_the_grid():
    from rasterio.transform import from_origin
    from shapely.geometry import box
    from pipeline.spatial_rules import roof_bands
    quant = np.zeros((30, 30), np.int16)
    for i in range(4, 26):                      # a 6-cell-wide band running diagonally across the raster
        quant[i, max(0, i - 3):min(30, i + 3)] = 3
    coarse = roof_bands(quant, from_origin(0, 45, 1.5, 1.5), box(0, 0, 45, 45), minimum_area=25, simplify_m=.2)
    fine = roof_bands(quant, from_origin(0, 45, 1.5, 1.5), box(0, 0, 45, 45), minimum_area=25, simplify_m=1.5 * .75)
    n_coarse = len(coarse[0]['geometry'].exterior.coords); n_fine = len(fine[0]['geometry'].exterior.coords)
    assert n_fine <= 12 and n_fine < n_coarse / 3, (n_coarse, n_fine)
    assert abs(fine[0]['geometry'].area - coarse[0]['geometry'].area) < .15 * coarse[0]['geometry'].area


def test_thin_tall_structure_on_a_roof_becomes_a_panel_and_a_wide_one_does_not():
    from pipeline.spatial_rules import surface_raster, rooftop_panels
    roof = np.array([[x, y, 10 + rng.normal(0, .04)] for x in np.arange(.3, 30, .4) for y in np.arange(.3, 20, .4)])
    # a 14 m long, 0.3 m thin sign face rising 2..5 m above the roof, along x at y=10
    sign = np.array([[x, 10 + rng.normal(0, .15), 10 + z] for x in np.arange(8, 22, .35) for z in np.linspace(2, 5, 7)])
    pts = np.r_[roof, sign]
    rr, keep = surface_raster(pts, 0.0, 20.0, 1.5, 20, 14)
    panels = rooftop_panels(pts, keep, rr, 0.0, 20.0, 1.5)
    assert len(panels) == 1, panels
    p = panels[0]
    assert abs(p['lengthM'] - 14) < 2.5 and p['widthM'] < 1.0 and abs(p['topZ'] - 15) < .4 and abs(p['roofZ'] - 10) < .3, p
    assert abs(abs(p['axis'][0]) - 1) < .05
    # a 6 m wide, 12 m long raised block 2 m above the roof is a roof volume, not a panel
    block = np.array([[x, y, 12 + rng.normal(0, .04)] for x in np.arange(8, 20, .4) for y in np.arange(6, 12, .4)])
    pts2 = np.r_[roof[(roof[:, 0] < 8) | (roof[:, 0] > 20) | (roof[:, 1] < 6) | (roof[:, 1] > 12)], block]
    rr2, keep2 = surface_raster(pts2, 0.0, 20.0, 1.5, 20, 14)
    assert rooftop_panels(pts2, keep2, rr2, 0.0, 20.0, 1.5) == []
    assert np.nanmax(rr2) > 11.8                                  # the block reads as roof surface (a tier)


def test_wall_of_a_taller_roof_section_is_not_a_panel():
    from pipeline.spatial_rules import surface_raster, rooftop_panels
    low = np.array([[x, y, 10 + rng.normal(0, .04)] for x in np.arange(.3, 30, .4) for y in np.arange(.3, 10, .4)])
    high = np.array([[x, y, 15 + rng.normal(0, .04)] for x in np.arange(.3, 30, .4) for y in np.arange(10.3, 20, .4)])
    face = np.array([[x, 10 + rng.normal(0, .1), 10 + z] for x in np.arange(3, 27, .35) for z in np.linspace(1, 5, 9)])  # the step wall
    inside = np.r_[low, face]                     # this building's footprint holds the low roof and the wall face
    rr, keep = surface_raster(inside, 0.0, 10.0, 1.5, 20, 7)
    assert rooftop_panels(inside, keep, rr, 0.0, 10.0, 1.5, context=np.r_[low, high, face]) == []   # roof beside the top: a wall
    sign = np.array([[x, 5 + rng.normal(0, .1), 10 + z] for x in np.arange(8, 22, .35) for z in np.linspace(2, 6, 9)])
    inside2 = np.r_[low, sign]
    rr2, keep2 = surface_raster(inside2, 0.0, 10.0, 1.5, 20, 7)
    assert len(rooftop_panels(inside2, keep2, rr2, 0.0, 10.0, 1.5, context=inside2)) == 1              # air around the top: a sign


def test_panel_with_no_roof_cell_under_it_reads_the_roof_from_its_surroundings():
    from pipeline.spatial_rules import surface_raster, rooftop_panels
    roof = np.array([[x, y, 10 + rng.normal(0, .04)] for x in np.arange(.3, 30, .4) for y in np.arange(.3, 20, .4)])
    roof = roof[(roof[:, 1] < 9.2) | (roof[:, 1] > 10.8)]                      # no roof returns under the sign line
    sign = np.array([[x, 10 + rng.normal(0, .1), 10 + z] for x in np.arange(8, 22, .35) for z in np.linspace(2, 6, 9)])
    pts = np.r_[roof, sign]
    rr, keep = surface_raster(pts, 0.0, 20.0, 1.5, 20, 14)
    panels = rooftop_panels(pts, keep, rr, 0.0, 20.0, 1.5, context=pts)
    assert len(panels) == 1 and abs(panels[0]['roofZ'] - 10) < .3, panels


def test_a_slope_grows_no_tier_walls_but_a_step_does():
    from pipeline.spatial_rules import sloped_cells
    yy, xx = np.mgrid[0:20, 0:30]
    slope = 10 + xx * 1.5 * .25                 # 14 degrees across the raster (1.5 m cells)
    assert sloped_cells(slope, 1.5).mean() > .9
    stepped = np.where(xx < 15, 10.0, 14.0)     # two flat plateaus with a 4 m step
    marked = sloped_cells(stepped, 1.5)
    assert marked.mean() < .12                  # only the cells right at the jump can read as a gradient
    assert not marked[:, 5:12].any() and not marked[:, 18:].any()


def test_a_roof_of_two_slopes_with_a_step_gets_two_patch_solids():
    import trimesh
    from shapely.geometry import box
    from general_world import scene_roofs
    cap = trimesh.creation.extrude_polygon(box(0, 0, 40, 24), .1); cap.apply_translation([0, 0, 9.9])
    cap.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(cap.vertices), 2)), material=trimesh.visual.material.PBRMaterial(name='roofing'))
    cap.metadata = {'layer': 'roofs', 'featureId': 'b'}
    # west half rises 10 -> 13 across x, east half sits 3 m higher and rises 16 -> 19: one envelope cannot explain both
    pts = np.array([[x, y, (10 + x * .15 if x < 20 else 16 + (x - 20) * .15) + rng.normal(0, .03)] for x in np.arange(.4, 40, .5) for y in np.arange(.4, 24, .5)])
    scene = trimesh.Scene(); scene.add_geometry(cap, geom_name='roofs__b___roofing', node_name='roofs__b___roofing')
    report = scene_roofs.repair_scene(scene, pts)
    patches = [r for r in report['records'] if r.get('status') == 'held-out-plane-patch' and r.get('applied')]
    assert len(patches) == 2, [(r.get('status'), r.get('applied')) for r in report['records']]
    tops = [m for n, m in scene.geometry.items() if n.startswith('roof_plane')]
    assert tops and max(m.bounds[1][2] for m in tops) > 18.5
    # The two patches meet along one straight step and reach the walls: no lattice of notches (the comb of
    # fins) and no channel of old cap between them. Vertical faces: a handful per solid, not hundreds.
    covered = sum(r['areaM2'] for r in patches)
    assert covered >= .97 * 40 * 24, covered
    vertical = sum(int((np.abs(m.face_normals[:, 2]) < .05).sum()) for m in tops)
    assert vertical <= 40, vertical


def test_patch_outlines_follow_the_building_and_its_step_not_the_raster():
    """A diagonal shed of two sloped sections: outlines are the footprint edges and one straight step line."""
    import trimesh
    from shapely.geometry import box
    from shapely.affinity import rotate
    from general_world import scene_roofs
    from general_world.geometry import polygons
    foot = rotate(box(0, 0, 40, 24), 32, origin=(0, 0))
    cap = trimesh.creation.extrude_polygon(foot, .1); cap.apply_translation([0, 0, 9.9])
    cap.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(cap.vertices), 2)), material=trimesh.visual.material.PBRMaterial(name='roofing'))
    cap.metadata = {'layer': 'roofs', 'featureId': 'd'}
    c, s = np.cos(np.radians(32)), np.sin(np.radians(32))
    pts = []
    for sx in np.arange(.3, 40, .45):
        for sy in np.arange(.3, 24, .45):
            z = (10 + sx * .15 if sx < 20 else 16 + (sx - 20) * .15) + rng.normal(0, .03)
            pts.append([sx * c - sy * s, sx * s + sy * c, z])
    scene = trimesh.Scene(); scene.add_geometry(cap, geom_name='roofs__d___roofing', node_name='roofs__d___roofing')
    report = scene_roofs.repair_scene(scene, np.array(pts))
    patches = [r for r in report['records'] if r.get('status') == 'held-out-plane-patch' and r.get('applied')]
    assert len(patches) == 2, [(r.get('status'), r.get('repairRejection')) for r in report['records']]
    assert sum(r['areaM2'] for r in patches) >= .95 * foot.area
    fit = next(r for r in report['records'] if r.get('patches'))['patches']
    # each section is a quadrilateral, give or take the raster's own resolution at the corners
    assert all(p['outlineVertices'] <= 8 for p in fit['patches']), fit['patches']
    tops = [m for n, m in scene.geometry.items() if n.startswith('roof_plane')]
    vertical = sum(int((np.abs(m.face_normals[:, 2]) < .05).sum()) for m in tops)
    assert vertical <= 40, vertical


def test_partition_simplifies_a_raster_staircase_into_one_shared_straight_boundary():
    from general_world.roof_planes import _partition
    grid = np.full((20, 30), -1, int)
    for y in range(20):
        for x in range(30):
            grid[y, x] = 0 if x + .6 * y < 18 else 1   # a diagonal boundary rasterised at 1 m
    regions = _partition(grid, np.array([0., 0.]), 1., tolerance=1.1)
    assert set(regions) == {0, 1}
    assert all(len(g.exterior.coords) - 1 <= 6 for g in regions.values()), {k: len(g.exterior.coords) for k, g in regions.items()}
    # conformal: the two regions still tile the raster with no sliver between them and no overlap
    assert abs(regions[0].area + regions[1].area - 600) < 1e-6
    assert regions[0].intersection(regions[1]).area < 1e-9



def test_a_plane_through_a_staircase_of_tiers_is_not_a_roof_patch():
    """Three flat tiers 1.5 m apart: a tilted plane can pass the hold-out test on the cells it claims, but
    interior cells under its outline sit on the steps and are not explained; no patch is built."""
    from shapely.geometry import box
    from general_world.roof_planes import fit_patches, RoofPolicy
    pts = np.array([[x, y, 10 + 1.5 * int(x // 10) + rng.normal(0, .03)] for x in np.arange(.4, 30, .5) for y in np.arange(.4, 20, .5)])
    patches, report = fit_patches(box(0, 0, 30, 20), pts, RoofPolicy(minimum_cells=18))
    assert not any(np.hypot(*p['model']['planes'][0][:2]) >= .05 for p in patches), report
    # the three flat tiers may be reported as flat planes; every sloped candidate must carry the rejection
    assert all(r.get('rejected') == 'interior-cells-not-explained' for r in report['patches'] if r['sloped'] and r['areaM2'] >= 40), report


def test_edge_cells_under_a_taller_block_do_not_join_the_patch():
    """A sloped roof beside a taller block: cells at the shared edge measure the block's roof (8 m higher)
    and stay out of the patch, so its solid never reaches under the block."""
    from shapely.geometry import box
    from general_world.roof_planes import fit_patches, RoofPolicy
    foot = box(0, 0, 30, 20)
    pts = [[x, y, 10 + x * .2 + rng.normal(0, .03)] for x in np.arange(.4, 30, .5) for y in np.arange(.4, 17.6, .5)]
    pts += [[x, y, 24 + rng.normal(0, .03)] for x in np.arange(.4, 30, .5) for y in np.arange(17.6, 20, .5)]   # the block's roof inside the same cap outline
    patches, report = fit_patches(foot, np.array(pts), RoofPolicy(minimum_cells=18))
    sloped = [p for p in patches if np.hypot(*p['model']['planes'][0][:2]) >= .05]
    assert len(sloped) == 1, report
    assert sloped[0]['geometry'].bounds[3] <= 18.0 + 1e-6, sloped[0]['geometry'].bounds


def test_a_slope_rising_from_a_flat_roof_meets_it_along_a_straight_eave():
    """Flat roof for x < 15, then a slope rising .2 per metre: the sloped patch's eave edge is one straight line
    within a cell of x = 15, not a saw of teeth where cells fit both surfaces."""
    from shapely.geometry import box
    from general_world.roof_planes import fit_patches, RoofPolicy
    from general_world.geometry import polygons
    pts = np.array([[x, y, (10 if x < 15 else 10 + (x - 15) * .2) + rng.normal(0, .04)] for x in np.arange(.4, 40, .5) for y in np.arange(.4, 24, .5)])
    patches, report = fit_patches(box(0, 0, 40, 24), pts, RoofPolicy(minimum_cells=18))
    sloped = [p for p in patches if np.hypot(*p['model']['planes'][0][:2]) >= .05]
    assert len(sloped) == 1, report
    g = sloped[0]['geometry']
    ring = np.concatenate([np.asarray(q.exterior.coords)[:-1] for q in polygons(g)])
    assert len(ring) <= 8, len(ring)
    eave = ring[ring[:, 0] < 20]
    assert len(eave) >= 2 and np.all(np.abs(eave[:, 0] - 15) <= 1.0), eave
    assert g.area >= .9 * 25 * 24, g.area


def test_a_structure_standing_on_a_sloped_roof_keeps_its_walls():
    """A 4 x 4 m plant room on a validated slope: the roof solid is built, the cap under it is replaced, but the
    room's walls, which the returns measured, are not trimmed to the plane."""
    import trimesh
    from shapely.geometry import box
    from general_world import scene_roofs
    foot = box(0, 0, 40, 24)
    cap = trimesh.creation.extrude_polygon(foot, .1); cap.apply_translation([0, 0, 9.9])
    cap.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(cap.vertices), 2)), material=trimesh.visual.material.PBRMaterial(name='roofing'))
    cap.metadata = {'layer': 'roofs', 'featureId': 'e'}
    room = trimesh.creation.box(extents=[4, 4, 3]); room.apply_translation([20, 12, 10 + 20 * .15 + 1.5])
    room.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(room.vertices), 2)), material=trimesh.visual.material.PBRMaterial(name='brick'))
    room.metadata = {'layer': 'buildings', 'featureId': 'e'}
    pts = []
    for x in np.arange(.3, 40, .45):
        for y in np.arange(.3, 24, .45):
            on_room = 18 <= x <= 22 and 10 <= y <= 14
            pts.append([x, y, (10 + 20 * .15 + 3 if on_room else 10 + x * .15) + rng.normal(0, .03)])
    scene = trimesh.Scene(); scene.add_geometry(cap, geom_name='roofs__e___roofing', node_name='roofs__e___roofing'); scene.add_geometry(room, geom_name='buildings__e', node_name='buildings__e')
    before = scene.geometry['buildings__e'].bounds[1][2]
    report = scene_roofs.repair_scene(scene, np.array(pts))
    applied = [r for r in report['records'] if r.get('applied')]
    assert applied and applied[0].get('protectedStructureM2', 0) >= 12, applied
    assert abs(scene.geometry['buildings__e'].bounds[1][2] - before) < 1e-6, (scene.geometry['buildings__e'].bounds[1][2], before)


def test_wall_profile_despikes_crowns_but_keeps_parapets_and_towers():
    from pipeline.facades import despike
    import numpy as np
    # a crown or pole two bins wide at the end of a 9 m wall, a crown in the middle, a spike at the start
    assert np.allclose(despike([9, 9, 9, 9, 17.4, 12.4]), [9, 9, 9, 9, 9, 9])
    assert np.allclose(despike([9, 9, 17, 9, 9]), [9, 9, 9, 9, 9])
    assert np.allclose(despike([16, 13, 9, 9, 9]), [9, 9, 9, 9, 9])
    # a false front the length of the wall, a tower three bins wide and a step up to a plateau all stay
    assert np.allclose(despike([30.6, 30.6, 30.6, 30.6, 30.6, 30.6]), [30.6] * 6)
    assert np.allclose(despike([24, 24, 50, 50, 50, 24]), [24, 24, 50, 50, 50, 24])
    assert np.allclose(despike([9, 9, 9, 20, 20, 20, 20]), [9, 9, 9, 20, 20, 20, 20])
    # a lower notch is not a spike; unmeasured bins are left alone
    out = despike([20, 20, np.nan, 12, 20, 20])
    assert out[3] == 12 and np.isnan(out[2])
