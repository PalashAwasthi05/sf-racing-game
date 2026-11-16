"""Geometric clearance for facade access; all coordinates come from the scene.

A stair inferred from a threshold must not extend into a carriageway. A conflict
requests a source reading of the flights/landings; it never compresses a flight
or invents a switchback to make it fit.
"""
from shapely.geometry import LineString


def carriageway_limit(point, road, center, width, start_depth, search_m=40.):
    if road is None or road.is_empty:
        return None
    distances = []
    for x in (center-width/2, center, center+width/2):
        a = point(x, 0., start_depth)[:2]
        b = point(x, 0., start_depth+search_m)[:2]
        line = LineString([a, b])
        hit = line.intersection(road)
        if hit.is_empty:
            continue
        def coordinates(g):
            if hasattr(g, 'geoms'):
                return [p for part in g.geoms for p in coordinates(part)]
            return list(g.coords)
        from shapely.geometry import Point
        distances.extend(start_depth + line.project(Point(p)) for p in coordinates(hit))
    return min(distances) if distances else None
