"""Locate named street ends and intersections along the mapped driving path."""
import argparse
import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform


def load(path):
    return json.loads(path.read_text())


def junctions(area):
    proj = Transformer.from_crs(4326, load(area / 'area-source-job.json')['projectedCrs'], always_xy=True)
    route = transform(proj.transform, shape(load(area / 'route-drive.geojson')))
    rows = load(area / 'data/streets.json')
    marks = {}
    lane_band = route.buffer(16)
    for row in rows:
        if not row.get('active') or not row.get('line') or not row.get('streetname'):
            continue
        source = transform(proj.transform, shape(row['line']))
        if source.geom_type != 'LineString' or source.intersection(lane_band).length > .75 * source.length:
            continue
        intersections=source.intersection(route)
        for at in getattr(intersections, 'geoms', [intersections]):
            if at.is_empty or at.geom_type != 'Point':
                continue
            station=route.project(at)
            if 2 < station < route.length-2:
                marks[(row['streetname'],round(station/5))]={'street':row['streetname'],'stationM':round(station,1)}
        # A side street often ends at the median, several metres from the driving lane.
        for i in (0, -1):
            at = Point(source.coords[i])
            station = route.project(at)
            if at.distance(route.interpolate(station)) > 15 or not 2 < station < route.length-2:
                continue
            ahead = route.interpolate(min(route.length,station+5))
            behind = route.interpolate(max(0,station-5))
            edge = source.interpolate(min(10,source.length) if i == 0 else max(0,source.length-10))
            ux,uy = edge.x-at.x, edge.y-at.y
            vx,vy = ahead.x-behind.x, ahead.y-behind.y
            if abs(ux*vy-uy*vx) <= .4*(ux*ux+uy*uy)**.5*(vx*vx+vy*vy)**.5:
                continue
            key = (row['streetname'], round(station/5))
            marks[key] = {'street':row['streetname'], 'stationM':round(station,1)}
    grouped = {}
    for mark in sorted(marks.values(), key=lambda m:m['stationM']):
        previous = next(reversed(grouped),None)
        station = previous if previous is not None and mark['stationM']-previous<10 else mark['stationM']
        grouped.setdefault(station,set()).add(mark['street'])
    return [{'stationM':station,'streets':sorted(names)} for station,names in sorted(grouped.items())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('area', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(junctions(args.area), indent=2)+'\n')
