import math

from shapely.geometry import LineString, box

from pipeline import mobility


def test_length_parser_handles_metric_and_imperial_osm_widths():
    assert math.isclose(mobility.length_m("12' 0\""), 3.6576, rel_tol=1e-8)
    assert math.isclose(mobility.length_m('19 ft 4 in'), 5.8928, rel_tol=1e-8)
    assert math.isclose(mobility.length_m('6"'), .1524, rel_tol=1e-8)
    assert math.isclose(mobility.length_m('2.4m'), 2.4, rel_tol=1e-8)
    assert mobility.number('2m of unknown units', 7) == 7
    assert mobility.number(float('nan'), 7) == 7


def test_explicit_blue_colour_stays_a_colour_variant_not_water():
    assert mobility.surface_colour({'surface:colour': 'blue'}) == 'stone#356aaf'
    assert mobility.surface_colour({'surface:colour': '#0066ff'}) == 'stone#0066ff'


def test_spec_keeps_explicit_total_width_and_known_side_order():
    tags = {
        'highway': 'residential', 'lanes': '2', 'width': "30'",
        'cycleway:right': 'lane', 'cycleway:right:width': "6'",
        'parking:lane:left': 'parallel', 'parking:lane:left:width': "7'",
    }
    result = mobility.spec(tags)

    assert math.isclose(result['widthM'], 30 * .3048, rel_tol=1e-8)
    assert math.isclose(result['bike']['right'], 6 * .3048, rel_tol=1e-8)
    assert math.isclose(result['parking']['left'], 7 * .3048, rel_tol=1e-8)
    assert result['leftM'] > result['rightM']
    assert math.isclose(result['laneWidthM'], result['vehicleWidthM'] / 2, rel_tol=1e-8)


def test_oneway_generic_cycleway_uses_only_known_directional_side():
    assert mobility.spec({'oneway': 'yes', 'cycleway': 'lane'})['bike'] == {'left': 0, 'right': 1.8}
    assert mobility.spec({'oneway': '-1', 'cycleway': 'lane'})['bike'] == {'left': 1.8, 'right': 0}
    # An explicit side tag is stronger than the generic one-way convention.
    assert mobility.spec({'oneway': 'yes', 'cycleway': 'lane', 'cycleway:left': 'lane'})['bike']['left'] == 1.8


def test_road_polygon_follows_curved_centerline_and_is_valid():
    centerline = LineString([(0, 0), (6, 0), (10, 4), (10, 10)])
    polygon = mobility.road_polygon(centerline, mobility.spec({'lanes': '2', 'width': '8 m'}))
    assert polygon.is_valid
    assert polygon.area > centerline.length * 3
    assert polygon.contains(centerline.interpolate(0.45, normalized=True))


def _compile_context(road_tags, *, cycleway=None):
    boundary = box(-3, -6, 24, 6)
    buildings = box(9, -2, 13, 2)
    road = {'id': 'road', 'tags': road_tags, 'g': LineString([(0, 0), (20, 0)])}
    ways = [] if cycleway is None else [cycleway]
    road_spec = mobility.spec(road_tags)
    captured = []

    def surface(geometry, semantic, material, offset):
        captured.append((geometry, semantic, material, offset))

    def lines(geometry):
        return [geometry] if geometry.geom_type == 'LineString' else list(geometry.geoms)

    ctx = {
        'surface': surface, 'cube': lambda *args: None, 'cylinder': lambda *args: None,
        'elev': lambda x, y: 0., 'boundary': boundary, 'all_buildings': buildings,
        'ways': ways, 'roads': [(road, road_spec['widthM'], road_spec['lanes'])],
        'road_specs': {'road': road_spec}, 'road_area': mobility.road_polygon(road['g'], road_spec).intersection(boundary).difference(buildings),
        'lines': lines, 'counts': {},
    }
    return ctx, captured


def test_green_cycle_lane_requires_explicit_colour_and_is_clipped_to_road():
    ctx, captured = _compile_context({'highway': 'residential', 'lanes': '2', 'cycleway:right': 'lane'})
    mobility.compile(ctx)
    assert not any(semantic == 'cycleways' and material == 'greenpaint' for _, semantic, material, _ in captured)

    ctx, captured = _compile_context({'highway': 'residential', 'lanes': '2', 'cycleway:right': 'lane',
                                      'cycleway:right:surface:colour': 'green'})
    mobility.compile(ctx)
    green = [g for g, semantic, material, _ in captured if semantic == 'cycleways' and material == 'greenpaint']
    assert green
    assert all(not g.intersects(ctx['all_buildings']) for g in green)


def test_mapped_cycleway_uses_explicit_imperial_width_without_inventing_green():
    cycle = {'id': 'cycle', 'tags': {'highway': 'cycleway', 'width': "12' 0\"", 'oneway': 'yes'},
             'g': LineString([(1, 4), (20, 4)])}
    ctx, captured = _compile_context({'highway': 'residential', 'lanes': '2'}, cycleway=cycle)
    records = mobility.compile(ctx)
    mapped = next(record for record in records if record['rule'] == 'mapped-cycleway')
    assert math.isclose(mapped['widthM'], 3.6576, rel_tol=1e-8)
    assert mapped['surfaceColour'] is None
    assert not any(material == 'greenpaint' for _, _, material, _ in captured)


def test_shared_ownership_clips_cycle_markings_and_path_surfaces():
    cycle = {'id': 'cycle', 'tags': {'highway': 'cycleway', 'width': '2 m'},
             'g': LineString([(1, 4), (20, 4)])}
    path = {'id': 'path', 'tags': {'highway': 'footway', 'width': '2 m'},
            'g': LineString([(1, 0), (20, 0)])}
    ctx, captured = _compile_context({'highway': 'residential', 'lanes': '2'}, cycleway=cycle)
    ctx['ways'].append(path)
    road_owned = ctx['road_area']
    cycle_owned = cycle['g'].buffer(.45, cap_style=2)
    sidewalk_owned = box(1, 3.5, 20, 4.5)
    ctx['street_plan'] = {'road': road_owned, 'cycle': cycle_owned, 'sidewalk': sidewalk_owned}
    mobility.compile(ctx)
    cycle_surfaces = [g for g, semantic, _, _ in captured if semantic == 'cycleways']
    path_surfaces = [g for g, semantic, _, _ in captured if semantic == 'paths']
    assert cycle_surfaces and all(g.difference(cycle_owned).area < 1e-8 for g in cycle_surfaces)
    assert not path_surfaces


def test_reverse_oneway_has_no_yellow_median_and_bollards_use_shared_space():
    road_tags = {'highway': 'residential', 'lanes': '2', 'oneway': '-1'}
    barrier = {'id': 'bollards', 'tags': {'barrier': 'bollard'},
               'g': LineString([(3, 0), (18, 0)])}
    ctx, captured = _compile_context(road_tags)
    ctx['ways'] = [barrier]
    cylinders = []
    ctx['cylinder'] = lambda *args: cylinders.append(args)
    ctx['street_plan'] = {'road': ctx['road_area'], 'cycle': box(0, 4, 1, 5),
                          'sidewalk': box(0, 3, 20, 5),
                          'forbidden': ctx['road_area'].union(ctx['all_buildings']),
                          'occupied': [], 'records': []}
    mobility.compile(ctx)
    assert not any(material == 'yellow' for _, semantic, material, _ in captured if semantic == 'markings')
    assert not cylinders

    path_barrier = {'id': 'path-bollards', 'tags': {'barrier': 'bollard'},
                    'g': LineString([(3, 4), (18, 4)])}
    ctx, captured = _compile_context(road_tags)
    ctx['ways'] = [path_barrier]
    cylinders = []
    ctx['cylinder'] = lambda *args: cylinders.append(args)
    ctx['street_plan'] = {'road': ctx['road_area'], 'cycle': box(0, 6, 1, 7),
                          'sidewalk': box(0, 3, 20, 5),
                          'forbidden': ctx['road_area'].union(ctx['all_buildings']),
                          'occupied': [], 'records': []}
    mobility.compile(ctx)
    assert cylinders
    assert all(record['status'] == 'mapped' for record in ctx['street_plan']['records'])
