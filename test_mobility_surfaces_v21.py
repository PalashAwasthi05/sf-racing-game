import math

from shapely.geometry import LineString, box

from pipeline import mobility


def test_length_parser_handles_metric_and_imperial_osm_widths():
    assert math.isclose(mobility.length_m("12' 0\""), 3.6576, rel_tol=1e-8)
    assert math.isclose(mobility.length_m('19 ft 4 in'), 5.8928, rel_tol=1e-8)
    assert math.isclose(mobility.length_m('2.4m'), 2.4, rel_tol=1e-8)


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

