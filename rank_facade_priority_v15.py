#!/usr/bin/env python
"""Building-level facade priority from a driver's point of view (v15).

Ports the wall-level visibility queue of pipeline/plan_observations.py and
aggregates it per building so the cheap question "which buildings deserve an
image-derived facade?" can be answered from geometry alone.

Method
  * Stations every 8 m along drivable OSM highways within the study radius.
  * For every facade wall and every station 3..160 m away: incidence
    (cos between the outward wall normal and the direction to the station)
    must be >= .15, and the 2D sight line from the station to a target point
    .3 m in front of the wall midpoint must not cross another building
    footprint (STRtree). Weight = length * min(height, 12) * incidence /
    max(25, d^2). Per wall the best 12 stations are summed (capped variant,
    like the reference) and all stations are summed (uncapped variant).
  * Building score = sum of its walls. Ranking is sorted by score, ties by id.

Generality
  * No location facts: the UTM zone is derived from the world centre, every
    threshold is a parameter, and building / way ids are only join keys.
  * Footprints are not trusted by id alone: the world consolidates outlines
    across sources, so each building's footprint is selected as the union of
    candidate polygons (OSM building ways + municipal parts) whose boundary its
    own walls lie on. The selection reports how much wall length it explains.
  * Every summation is order-independent (sorted math.fsum), so shuffling the
    input order cannot change the ranking; the script verifies that itself.

Usage
  rank_facade_priority_v15.py [--out DIR] [--radius 200] [--report-radius 265]
      [--include-service] [--no-context-occluders] [--self-occlusion]
      [--shuffle-seed N] [--no-shuffle-check]
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULTS = {
    'world': ROOT / 'king-street/public/world.json',
    'evidence': ROOT / 'detailed-v14i/king-street-evidence.json',
    'osm': ROOT / 'data/osm.xml',
    'datasf': ROOT / 'data/buildings.json',
    'out': ROOT / 'facade-priority-v15',
}
DRIVABLE = ('primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street')
# OSM building values that carry no eye-level wall (canopies, bridges): neither facade owners nor occluders.
WALL_LESS_BUILDINGS = ('roof', 'bridge')
PARAMS = {
    'stationSpacingM': 8.0,
    'stationStartM': 3.0,
    'minDistanceM': 3.0,
    'maxDistanceM': 160.0,
    'minIncidence': 0.15,
    'heightCapM': 12.0,
    'distanceFloorM2': 25.0,
    'targetOffsetM': 0.3,
    'bestStations': 12,
    'driverEyeHeightM': 1.25,
    'footprintMatchToleranceM': 0.75,
    'duplicateOverlapFraction': 0.25,
}
PRIOR_ONLY_CLASS = 'Unverified prior'
IMAGE_STATUSES = ('visual-fit', 'image-layout-fit', 'visual-grammar', 'envelope-graph')


# ----------------------------------------------------------------------------- geometry helpers
def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat); derived, never hard-coded."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_local(lon, lat):
    epsg = utm_epsg(lon, lat)
    ll = Transformer.from_crs(4326, epsg, always_xy=True)
    cx, cy = ll.transform(lon, lat)

    def local(x, y, z=None):
        a, b = ll.transform(x, y)
        return np.asarray(a) - cx, np.asarray(b) - cy

    return local, epsg


def polygons_of(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if hasattr(g, 'geoms'):
        return [p for a in g.geoms for p in polygons_of(a)]
    return []


def clean_polygon(g):
    g = make_valid(g) if not g.is_valid else g
    parts = polygons_of(g)
    if not parts:
        return None
    return unary_union(parts)


def fsum_sorted(values):
    """Order-independent floating sum."""
    return math.fsum(sorted(float(v) for v in values))


# ----------------------------------------------------------------------------- loading
def load_osm(path, local, road_classes):
    """Return (footprints{id: polygon}, roads[(id, class, LineString)], skippedWallLess)."""
    root = ET.parse(path).getroot()
    nodes = {n.attrib['id']: (float(n.attrib['lon']), float(n.attrib['lat'])) for n in root.findall('node')}
    footprints, roads, skipped = {}, [], 0
    for way in root.findall('way'):
        tags = {a.attrib['k']: a.attrib['v'] for a in way.findall('tag')}
        coords = [nodes[a.attrib['ref']] for a in way.findall('nd') if a.attrib['ref'] in nodes]
        if len(coords) < 2:
            continue
        x, y = local(*zip(*coords))
        xy = np.c_[x, y]
        if tags.get('building') and len(coords) > 3:
            if tags['building'] in WALL_LESS_BUILDINGS:
                skipped += 1
            else:
                poly = clean_polygon(Polygon(xy))
                if poly is not None and poly.area > 0:
                    footprints['osm_' + way.attrib['id']] = poly
        hw = tags.get('highway')
        if hw in road_classes:
            roads.append(('osm_' + way.attrib['id'], hw, LineString(xy)))
    return footprints, roads, skipped


def load_datasf(path, local):
    out = {}
    for b in json.load(open(path)):
        g = clean_polygon(transform(local, shape(b['shape'])))
        if g is not None and g.area > 0:
            out['sf_' + b['sf16_bldgid'].replace('.', '_')] = g
    return out


def road_stations(roads, radius, spacing, start):
    """Points every `spacing` m along road lines within `radius` of the origin."""
    origin = Point(0.0, 0.0)
    pts, owners = [], []
    for rid, hw, line in roads:
        if line.distance(origin) >= radius:
            continue
        for s in np.arange(start, line.length, spacing):
            p = line.interpolate(s)
            if p.distance(origin) < radius:
                pts.append((p.x, p.y))
                owners.append(rid)
    if not pts:
        return np.zeros((0, 2)), []
    return np.asarray(pts, dtype=float), owners


# ----------------------------------------------------------------------------- footprint selection
def wall_segment(wall):
    m = np.asarray(wall['midpoint'], dtype=float)
    n = np.asarray(wall['normal'], dtype=float)
    n = n / max(np.linalg.norm(n), 1e-12)
    t = np.array([-n[1], n[0]])
    h = float(wall['lengthM']) / 2.0
    return m - t * h, m + t * h, m, n


def select_footprints(walls_by_building, candidates, tol):
    """Greedy set cover: per building, choose candidate polygons whose boundary
    its walls lie on (midpoint within `tol` of the boundary, normal pointing
    out of the polygon). Returns {id: {'geometry', 'candidateIds', 'explainedFraction', ...}}.
    Independent of input order: candidates are ranked by explained length then id."""
    cand_ids = sorted(candidates)
    tree = STRtree([candidates[c] for c in cand_ids])
    result = {}
    for bid in sorted(walls_by_building):
        walls = walls_by_building[bid]
        total_len = fsum_sorted(w['lengthM'] for w in walls)
        probes = []
        for w in walls:
            _, _, m, n = wall_segment(w)
            probes.append((Point(m), Point(m - n * 0.5), Point(m + n * 0.5), float(w['lengthM'])))
        near = set()
        for mid, _, _, _ in probes:
            near.update(int(i) for i in tree.query(mid.buffer(tol + 1.0)))
        explained_by = {}
        for i in sorted(near):
            poly = candidates[cand_ids[i]]
            boundary = poly.boundary
            covered = []
            for k, (mid, inside, outside, length) in enumerate(probes):
                if boundary.distance(mid) <= tol and poly.contains(inside) and not poly.contains(outside):
                    covered.append(k)
            if covered:
                explained_by[cand_ids[i]] = covered
        chosen, remaining = [], set(range(len(walls)))
        while remaining and explained_by:
            best = max(explained_by, key=lambda c: (fsum_sorted(probes[k][3] for k in explained_by[c] if k in remaining), c))
            gain = [k for k in explained_by[best] if k in remaining]
            if not gain:
                break
            chosen.append(best)
            remaining -= set(gain)
            del explained_by[best]
        geometry = unary_union([candidates[c] for c in chosen]) if chosen else None
        explained_len = fsum_sorted(probes[k][3] for k in range(len(walls)) if k not in remaining)
        result[bid] = {
            'geometry': geometry,
            'candidateIds': chosen,
            'explainedFraction': (explained_len / total_len) if total_len > 0 else 0.0,
            'unexplainedWalls': len(remaining),
        }
    return result


# ----------------------------------------------------------------------------- scoring
def score_walls(walls, stations, occluder_polys, occluder_owner, params, self_occlusion=False):
    """Per wall: capped and uncapped visibility-weighted area, visible station count.
    Sight lines run from station to (midpoint + offset * normal). Occluders whose
    polygon contains the target point are ignored for that wall: a point .3 m in
    front of a facade cannot be inside another building, so such a hit is a
    source disagreement, not a real occlusion."""
    tree = STRtree(occluder_polys) if occluder_polys else None
    kd = cKDTree(stations) if len(stations) else None
    out = []
    dmin, dmax = params['minDistanceM'], params['maxDistanceM']
    for wall in walls:
        _, _, m, n = wall_segment(wall)
        area = float(wall['lengthM']) * min(float(wall['heightM']), params['heightCapM'])
        target = m + n * params['targetOffsetM']
        rec = {'buildingId': wall['buildingId'], 'edge': wall.get('edge', wall.get('grammarId', 'region')),
               'status': wall['status'], 'lengthM': wall['lengthM'], 'heightM': wall['heightM'], 'areaCappedM2': round(area, 3),
               'scoreCapped': 0.0, 'scoreUncapped': 0.0, 'visibleStations': 0, 'candidateStations': 0, 'bestStation': None,
               'bestDistanceM': None, 'ignoredOccluders': 0}
        if kd is None:
            out.append(rec)
            continue
        idx = np.asarray(sorted(kd.query_ball_point(m, dmax)), dtype=int)
        if idx.size == 0:
            out.append(rec)
            continue
        p = stations[idx]
        v = p - m
        d = np.linalg.norm(v, axis=1)
        keep = d >= dmin
        idx, p, v, d = idx[keep], p[keep], v[keep], d[keep]
        if idx.size == 0:
            out.append(rec)
            continue
        incidence = (v @ n) / d
        keep = incidence >= params['minIncidence']
        idx, p, d, incidence = idx[keep], p[keep], d[keep], incidence[keep]
        rec['candidateStations'] = int(idx.size)
        if idx.size == 0:
            out.append(rec)
            continue
        visible = np.ones(idx.size, dtype=bool)
        if tree is not None:
            ignore = set()
            for gi in tree.query(Point(target), predicate='within'):
                ignore.add(int(gi))
            if not self_occlusion:
                ignore.update(i for i, owner in enumerate(occluder_owner) if owner == wall['buildingId'])
            rec['ignoredOccluders'] = len(ignore)
            lines = [LineString([tuple(pp), tuple(target)]) for pp in p]
            pairs = tree.query(lines, predicate='intersects')
            for li, gi in zip(pairs[0], pairs[1]):
                if int(gi) not in ignore:
                    visible[li] = False
        scores = area * incidence / np.maximum(params['distanceFloorM2'], d ** 2)
        scores = np.where(visible, scores, 0.0)
        vis_scores = sorted((float(s) for s in scores[visible]), reverse=True)
        rec['visibleStations'] = int(visible.sum())
        rec['scoreUncapped'] = fsum_sorted(vis_scores)
        rec['scoreCapped'] = fsum_sorted(vis_scores[:params['bestStations']])
        if vis_scores:
            best = int(np.argmax(scores))
            rec['bestStation'] = [round(float(p[best][0]), 2), round(float(p[best][1]), 2)]
            rec['bestDistanceM'] = round(float(d[best]), 2)
        out.append(rec)
    return out


def aggregate(world_buildings, wall_records, footprints, roads, stations, params):
    by = {}
    for r in wall_records:
        by.setdefault(r['buildingId'], []).append(r)
    road_lines = [(rid, hw, line) for rid, hw, line in roads]
    kd = cKDTree(stations) if len(stations) else None
    rows = []
    for b in world_buildings:
        bid = b['id']
        recs = by.get(bid, [])
        fp = footprints.get(bid, {}).get('geometry')
        ref = fp if fp is not None else Point(b['center'][0], b['center'][1])
        nearest = min(((line.distance(ref), hw, rid) for rid, hw, line in road_lines), default=(None, None, None))
        nearest_station = None
        if kd is not None:
            probe = np.asarray(ref.exterior.coords if ref.geom_type == 'Polygon' else
                               ([c for g in polygons_of(ref) for c in g.exterior.coords] if ref.geom_type == 'MultiPolygon' else [ref.coords[0]]), dtype=float)
            nearest_station = round(float(kd.query(probe)[0].min()), 2)
        rows.append({
            'nearestStationDistanceM': nearest_station,
            'id': bid,
            'name': b.get('name'),
            'score': fsum_sorted(r['scoreCapped'] for r in recs),
            'scoreUncapped': fsum_sorted(r['scoreUncapped'] for r in recs),
            'frontageLengthVisibleM': fsum_sorted(r['lengthM'] for r in recs if r['visibleStations'] > 0),
            'frontageLengthTotalM': fsum_sorted(r['lengthM'] for r in recs),
            'wallsTotal': len(recs),
            'wallsVisible': sum(1 for r in recs if r['visibleStations'] > 0),
            'wallsInferred': sum(1 for r in recs if r['status'] == 'inferred'),
            'wallsImageDerived': sum(1 for r in recs if r['status'] in IMAGE_STATUSES),
            'nearestRoadDistanceM': None if nearest[0] is None else round(float(nearest[0]), 2),
            'nearestRoadClass': nearest[1],
            'evidenceClass': b.get('facadeEvidence'),
            'priorOnly': b.get('facadeEvidence') == PRIOR_ONLY_CLASS,
            'fittedWalls': b.get('fittedWalls'),
            'totalWalls': b.get('totalWalls'),
            'heightM': b.get('height'),
            'areaM2': b.get('area'),
            'footprintCandidates': footprints.get(bid, {}).get('candidateIds', []),
            'footprintExplainedFraction': round(footprints.get(bid, {}).get('explainedFraction', 0.0), 3),
        })
    return rows


def rank_rows(rows, key='score'):
    rows = sorted(rows, key=lambda r: (-round(r[key], 9), r['id']))
    total = fsum_sorted(r[key] for r in rows)
    cum = 0.0
    for i, r in enumerate(rows):
        share = (r[key] / total) if total > 0 else 0.0
        cum += share
        r['rank'] = i + 1
        r['share'] = share
        r['cumulativeShare'] = min(cum, 1.0)
    return rows, total


def count_for_share(rows, threshold):
    """Smallest N such that the top-N rows reach `threshold` of the total."""
    for r in rows:
        if r['cumulativeShare'] >= threshold - 1e-12:
            return r['rank']
    return len(rows)


# ----------------------------------------------------------------------------- pipeline
def run(args, world, walls, footprint_candidates, roads, shuffle_seed=None):
    params = dict(PARAMS)
    buildings = list(world['buildings'])
    walls = list(walls)
    cand_items = sorted(footprint_candidates.items())
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(buildings)
        rng.shuffle(walls)
        rng.shuffle(cand_items)
    candidates = dict(cand_items)

    walls_by_building = {}
    for w in walls:
        walls_by_building.setdefault(w['buildingId'], []).append(w)
    world_ids = [b['id'] for b in buildings]

    footprints = select_footprints(walls_by_building, candidates, params['footprintMatchToleranceM'])
    selected = {bid: f['geometry'] for bid, f in footprints.items() if f['geometry'] is not None}
    for bid in world_ids:  # fallback: raw geometry by id when walls explain nothing
        if bid not in selected and bid in candidates:
            selected[bid] = candidates[bid]
            footprints.setdefault(bid, {})['geometry'] = candidates[bid]
            footprints[bid].setdefault('candidateIds', [bid])
            footprints[bid].setdefault('explainedFraction', 0.0)
            footprints[bid].setdefault('unexplainedWalls', len(walls_by_building.get(bid, [])))
    world_union = unary_union(list(selected.values())) if selected else None
    used = set(c for f in footprints.values() for c in f.get('candidateIds', []))

    occluder_polys, occluder_owner = [], []
    for bid in sorted(selected):
        occluder_polys.append(selected[bid])
        occluder_owner.append(bid)
    context_count = 0
    reach = max(args.radius, args.report_radius) + params['maxDistanceM']
    if not args.no_context_occluders and world_union is not None:
        origin = Point(0.0, 0.0)
        for cid in sorted(candidates):
            if cid in used or cid in selected:
                continue
            g = candidates[cid]
            if g.distance(origin) > reach:
                continue
            overlap = g.intersection(world_union).area / max(g.area, 1e-9)
            if overlap > params['duplicateOverlapFraction']:
                continue
            occluder_polys.append(g)
            occluder_owner.append(None)
            context_count += 1

    results = {}
    for radius in sorted({args.radius, args.report_radius}):
        stations, _ = road_stations(roads, radius, params['stationSpacingM'], params['stationStartM'])
        wall_records = score_walls(walls, stations, occluder_polys, occluder_owner, params, self_occlusion=args.self_occlusion)
        rows = aggregate(buildings, wall_records, footprints, roads, stations, params)
        rows, total = rank_rows(rows, 'score')
        results[radius] = {'stations': int(len(stations)), 'rows': rows, 'total': total, 'walls': wall_records}

    primary = results[args.radius]
    report = results[args.report_radius]
    by_id_report = {r['id']: r for r in report['rows']}
    unc_rows, unc_total = rank_rows([dict(r) for r in primary['rows']], 'scoreUncapped')
    unc_rank = {r['id']: (r['rank'], r['share'], r['cumulativeShare']) for r in unc_rows}
    n85_report = count_for_share(report['rows'], 0.85)
    for r in primary['rows']:
        rr = by_id_report[r['id']]
        r['scoreReportRadius'] = rr['score']
        r['rankReportRadius'] = rr['rank']
        r['shareReportRadius'] = rr['share']
        r['nearestStationDistanceReportRadiusM'] = rr['nearestStationDistanceM']
        r['rankUncapped'], r['shareUncapped'], r['cumulativeShareUncapped'] = unc_rank[r['id']]
        r['zeroScore'] = r['score'] <= 0.0
        r['zeroScoreAtReportRadius'] = rr['score'] <= 0.0
        r['neverDriverVisible'] = r['zeroScore'] and r['zeroScoreAtReportRadius']
        r['in85pctReportRadius'] = rr['rank'] <= n85_report
    n85 = count_for_share(primary['rows'], 0.85)
    n95 = count_for_share(primary['rows'], 0.95)
    for r in primary['rows']:
        r['in85pct'] = r['rank'] <= n85
        r['in95pct'] = r['rank'] <= n95

    diagnostics = {
        'stations': {str(k): v['stations'] for k, v in results.items()},
        'occluders': {'worldFootprints': len(selected), 'contextFootprints': context_count, 'candidatePolygons': len(candidates)},
        'footprintSelection': {
            'buildingsWithSelectedFootprint': sum(1 for bid in world_ids if footprints.get(bid, {}).get('candidateIds')),
            'buildingsFallingBackToRawId': sum(1 for bid in world_ids if footprints.get(bid, {}).get('explainedFraction', 0) == 0 and bid in selected),
            'wallsUnexplainedByAnyCandidate': sum(f.get('unexplainedWalls', 0) for f in footprints.values()),
            'medianExplainedFraction': round(float(np.median([footprints[b]['explainedFraction'] for b in world_ids if b in footprints])), 3) if footprints else None,
            'minExplainedFraction': round(float(min(footprints[b]['explainedFraction'] for b in world_ids if b in footprints)), 3) if footprints else None,
        },
        'wallsWithIgnoredForeignOccluder': sum(1 for w in primary['walls'] if w['ignoredOccluders'] > (0 if args.self_occlusion else 1)),
    }
    return {'primary': primary, 'report': report, 'results': results, 'n85': n85, 'n95': n95, 'n85ReportRadius': n85_report,
            'n95ReportRadius': count_for_share(report['rows'], 0.95), 'diagnostics': diagnostics, 'params': params}


def signature(rows):
    return [(r['id'], round(r['score'], 6), round(r['scoreUncapped'], 6), round(r['scoreReportRadius'], 6)) for r in rows]


def write_markdown(path, out, args, sanity, inputs):
    rows = out['primary']['rows']
    lines = []
    lines.append('# Facade priority ranking v15 (driver view)')
    lines.append('')
    lines.append(f"Stations every {PARAMS['stationSpacingM']:g} m on drivable OSM highways ({', '.join(inputs['roadClasses'])}) within {args.radius} m of the local origin "
                 f"({out['primary']['stations']} stations; {out['report']['stations']} at {args.report_radius} m). Driver eye height {PARAMS['driverEyeHeightM']} m. "
                 f"Wall weight = length x min(height, {PARAMS['heightCapM']:g} m) x incidence / max({PARAMS['distanceFloorM2']:g}, d^2), stations {PARAMS['minDistanceM']:g}..{PARAMS['maxDistanceM']:g} m, "
                 f"incidence >= {PARAMS['minIncidence']}, 2D footprint occlusion, best {PARAMS['bestStations']} stations per wall (uncapped variant also reported).")
    lines.append('')
    zero = [r for r in rows if r['zeroScore']]
    never = [r['id'] for r in rows if r['neverDriverVisible']]
    edge = [r['id'] for r in rows if r['zeroScore'] and not r['neverDriverVisible']]
    lines.append(f"Total score {out['primary']['total']:.2f}. Buildings covering 85%: **{out['n85']}**; 95%: **{out['n95']}** of {len(rows)} "
                 f"(at {args.report_radius:g} m: {out['n85ReportRadius']} and {out['n95ReportRadius']}). "
                 f"Zero-score at {args.radius:g} m: {len(zero)}, of which never driver-visible at either radius: {len(never)}.")
    prior85 = [r['id'] for r in rows if r['priorOnly'] and r['in85pct']]
    lines.append('')
    lines.append(f"'{PRIOR_ONLY_CLASS}' buildings inside the 85% set ({len(prior85)}): " + (', '.join(f'`{i}`' for i in prior85) if prior85 else 'none'))
    lines.append('')
    lines.append(f"Never driver-visible (zero at both radii; leave as priors) ({len(never)}): " + (', '.join(f'`{i}`' for i in never) if never else 'none'))
    lines.append('')
    lines.append(f"Zero only at {args.radius:g} m because their roads lie outside the station disc ({len(edge)}): " + (', '.join(f'`{i}`' for i in edge) if edge else 'none'))
    lines.append('')
    lines.append('Sanity: walls processed {wallsProcessed} of {facadeWallsCount}; shuffle-invariant ranking: {shuffleInvariant}.'.format(**sanity))
    lines.append('')
    lines.append('| rank | id | score | share | cum | rank uncapped | rank @%g m | visible frontage m | walls vis/total | nearest road m (class) | nearest station m | evidence | flags |' % args.report_radius)
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|')
    for r in rows:
        flags = []
        if r['neverDriverVisible']:
            flags.append('never-visible')
        elif r['zeroScore']:
            flags.append('zero-at-%g' % args.radius)
        if r['priorOnly'] and r['in85pct']:
            flags.append('prior-in-85')
        elif r['priorOnly']:
            flags.append('prior')
        if r['in85pct']:
            flags.append('top85')
        elif r['in95pct']:
            flags.append('top95')
        lines.append(f"| {r['rank']} | `{r['id']}` | {r['score']:.3f} | {100*r['share']:.1f}% | {100*r['cumulativeShare']:.1f}% | {r['rankUncapped']} | {r['rankReportRadius']} | "
                     f"{r['frontageLengthVisibleM']:.1f} / {r['frontageLengthTotalM']:.1f} | {r['wallsVisible']}/{r['wallsTotal']} | {r['nearestRoadDistanceM']} ({r['nearestRoadClass']}) | {r['nearestStationDistanceM']} | {r['evidenceClass']} | {' '.join(flags)} |")
    lines.append('')
    lines.append('Scores are geometric predictions of driver-visible facade area, not verified sightings; occlusion is a 2D footprint approximation.')
    path.write_text('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', default=str(DEFAULTS['world']))
    ap.add_argument('--evidence', default=str(DEFAULTS['evidence']))
    ap.add_argument('--osm', default=str(DEFAULTS['osm']))
    ap.add_argument('--datasf', default=str(DEFAULTS['datasf']))
    ap.add_argument('--out', default=str(DEFAULTS['out']))
    ap.add_argument('--radius', type=float, default=None, help='station radius (default: world studyRadiusM)')
    ap.add_argument('--report-radius', type=float, default=265.0, help='second radius reported alongside')
    ap.add_argument('--include-service', action='store_true', help='also sample highway=service')
    ap.add_argument('--no-context-occluders', action='store_true', help='occlude only with study-set footprints')
    ap.add_argument('--self-occlusion', action='store_true', help='let a building\'s own footprint block its walls (concave outlines)')
    ap.add_argument('--shuffle-seed', type=int, default=None, help='shuffle input order before processing')
    ap.add_argument('--no-shuffle-check', action='store_true', help='skip the internal shuffled re-run')
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--bottom', type=int, default=5)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    world = json.load(open(args.world))
    evidence = json.load(open(args.evidence))
    walls = evidence['facadeWalls']
    if args.radius is None:
        args.radius = float(world.get('studyRadiusM', 200))
    if args.report_radius == args.radius:
        args.report_radius = args.radius
    local, epsg = make_local(world['center']['lon'], world['center']['lat'])
    road_classes = tuple(DRIVABLE) + (('service',) if args.include_service else ())
    osm_footprints, roads, skipped_wall_less = load_osm(args.osm, local, road_classes)
    sf_footprints = load_datasf(args.datasf, local)
    candidates = {**osm_footprints, **sf_footprints}
    inputs = {'world': args.world, 'evidence': args.evidence, 'osm': args.osm, 'datasf': args.datasf, 'epsg': epsg,
              'roadClasses': list(road_classes), 'roadWays': len(roads), 'osmFootprints': len(osm_footprints), 'osmWallLessBuildingsSkipped': skipped_wall_less,
              'datasfFootprints': len(sf_footprints), 'worldBuildings': len(world['buildings']), 'facadeWalls': len(walls)}

    out = run(args, world, walls, candidates, roads, shuffle_seed=args.shuffle_seed)
    rows = out['primary']['rows']

    sanity = {'wallsProcessed': len(out['primary']['walls']), 'facadeWallsCount': len(walls),
              'wallsProcessedMatchesInput': len(out['primary']['walls']) == len(walls),
              'buildingsRanked': len(rows), 'worldBuildings': len(world['buildings']),
              'shuffleInvariant': None, 'shuffleSeeds': None}
    if not args.no_shuffle_check:
        seeds = [(args.shuffle_seed or 0) + 1, (args.shuffle_seed or 0) + 2]
        sig0 = signature(rows)
        ok = True
        for s in seeds:
            alt = run(args, world, walls, candidates, roads, shuffle_seed=s)
            ok = ok and signature(alt['primary']['rows']) == sig0
        sanity['shuffleInvariant'] = bool(ok)
        sanity['shuffleSeeds'] = seeds

    zero = [r['id'] for r in rows if r['zeroScore']]
    never = [r['id'] for r in rows if r['neverDriverVisible']]
    prior85 = [r['id'] for r in rows if r['priorOnly'] and r['in85pct']]
    prior95 = [r['id'] for r in rows if r['priorOnly'] and r['in95pct']]
    summary = {
        'totalScore': out['primary']['total'],
        'totalScoreUncapped': fsum_sorted(r['scoreUncapped'] for r in rows),
        'totalScoreReportRadius': out['report']['total'],
        'buildingsFor85pct': out['n85'],
        'buildingsFor95pct': out['n95'],
        'buildingsFor85pctReportRadius': out['n85ReportRadius'],
        'buildingsFor95pctReportRadius': out['n95ReportRadius'],
        'priorOnlyBuildings': sum(1 for r in rows if r['priorOnly']),
        'priorOnlyIn85pct': prior85,
        'priorOnlyIn95pct': prior95,
        'priorOnlyIn85pctReportRadius': [r['id'] for r in rows if r['priorOnly'] and r['in85pctReportRadius']],
        'imageDerivedOutside85pct': [r['id'] for r in rows if not r['priorOnly'] and not r['in85pct']],
        'zeroScoreBuildings': zero,
        'zeroScorePriorOnly': [r['id'] for r in rows if r['zeroScore'] and r['priorOnly']],
        'zeroScoreImageDerived': [r['id'] for r in rows if r['zeroScore'] and not r['priorOnly']],
        'neverDriverVisibleBuildings': never,
        'zeroOnlyAtStudyRadius': [i for i in zero if i not in never],
    }

    def rounded(r):
        o = {}
        for k, v in r.items():
            o[k] = round(v, 6) if isinstance(v, float) else v
        return o

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        'schemaVersion': 1,
        'generator': pathlib.Path(__file__).name,
        'method': {**out['params'], 'occlusion': '2D footprint sight-line test (shapely STRtree), own footprint excluded' + (' except --self-occlusion' if args.self_occlusion else ''),
                   'stationRadiusM': args.radius, 'reportRadiusM': args.report_radius, 'roadClasses': list(road_classes),
                   'contextOccluders': not args.no_context_occluders, 'scoreDefinition': 'sum over walls of the best-12-station sum of length*min(height,cap)*incidence/max(floor,d^2)'},
        'inputs': inputs,
        'summary': summary,
        'sanity': sanity,
        'diagnostics': out['diagnostics'],
        'ranking': [rounded(r) for r in rows],
        'elapsedSeconds': None,
    }
    wall_payload = [rounded(w) for w in out['primary']['walls']]
    elapsed = time.perf_counter() - t0
    payload['elapsedSeconds'] = round(elapsed, 2)
    (outdir / 'ranking.json').write_text(json.dumps(payload, indent=1))
    (outdir / 'wall-scores.json').write_text(json.dumps({'stationRadiusM': args.radius, 'walls': wall_payload}, indent=1))
    write_markdown(outdir / 'ranking.md', out, args, sanity, inputs)

    print(f"walls processed {sanity['wallsProcessed']} / facadeWalls {sanity['facadeWallsCount']} ; buildings ranked {len(rows)} / {len(world['buildings'])}")
    print(f"stations: {out['diagnostics']['stations']} ; occluders: {out['diagnostics']['occluders']} ; footprint selection: {out['diagnostics']['footprintSelection']}")
    print(f"total score {out['primary']['total']:.3f} ; buildings for 85%: {out['n85']} ; for 95%: {out['n95']} ; zero-score at {args.radius:g} m: {len(zero)} "
          f"(never visible at either radius: {len(never)}) ; prior-only in 85%: {len(prior85)} ; at {args.report_radius:g} m: 85% -> {out['n85ReportRadius']}, 95% -> {out['n95ReportRadius']}")
    print(f"shuffle-invariant: {sanity['shuffleInvariant']} (seeds {sanity['shuffleSeeds']})")
    print(f"top {args.top}:")
    for r in rows[:args.top]:
        print(f"  {r['rank']:2d} {r['id']:<22} score {r['score']:9.3f} share {100*r['share']:5.1f}% cum {100*r['cumulativeShare']:5.1f}%  vis {r['frontageLengthVisibleM']:6.1f} m  {r['evidenceClass']}")
    print(f"bottom {args.bottom}:")
    for r in rows[-args.bottom:]:
        print(f"  {r['rank']:2d} {r['id']:<22} score {r['score']:9.3f} share {100*r['share']:5.1f}% cum {100*r['cumulativeShare']:5.1f}%  vis {r['frontageLengthVisibleM']:6.1f} m  {r['evidenceClass']}")
    print(f"wrote {outdir / 'ranking.json'}, {outdir / 'ranking.md'}, {outdir / 'wall-scores.json'} in {elapsed:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
