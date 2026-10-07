"""Overhead street equipment for an area's local preview, from OpenStreetMap:
  trolleybus wires where a street is tagged trolley_wire=yes: a pair of contact wires over each travel direction, 5.6 m
    above the road, hung every 30 m from a span wire between two steel poles standing at the kerbs;
  street lamps along lit streets (lit=yes, or an untagged city street; none where lit=no): a cobra-head lamp on an 8.5 m
    pole every 36 m, staggered across the street. On a trolley street the span poles carry the lamps.
Poles stand 0.45 m behind the kerb of the built road surface (found as areas/parking.py finds it) and keep clear of
intersections, crossings and the garage doors the photographed facades show; the wires follow the street's own slope.
Written beside the preview's manifest (<label>-overhead.json, manifest 'overhead') for the viewer. Location-free: every
rule reads the area's own data.
Usage: overhead.py <area dir> <build label>     (SFW_AREA set to the area; any working directory)"""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frontage_crops_v15 as fc
from parking import roads_outline, kerbs, garage_doors, COMPARE

STREETS = ('trunk', 'trunk_link', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'living_street', 'primary_link', 'secondary_link', 'tertiary_link')
TROLLEY_SPAN, LAMP_SPAN = 30., 36.
WIRE_Z, SPAN_Z, PAIR, SAG = 5.6, 7.2, .3, .18
BEHIND_KERB, PARKING_LANE = .45, 2.4


def main():
    area, label = Path(sys.argv[1]).resolve(), sys.argv[2]
    build = area / label.split('-')[-1] if (area / label.split('-')[-1]).exists() else area / label
    world, frame = fc.load_world()
    road = roads_outline(build / 'materials/candidate.glb'); inside, kerb = kerbs(road)
    root = ET.parse(fc.OSM_PATH).getroot()
    nodes = {n.get('id'): np.array(frame.local(float(n.get('lon')), float(n.get('lat'))), float) for n in root.iter('node')}
    ntags = {n.get('id'): {t.get('k'): t.get('v') for t in n.iter('tag')} for n in root.iter('node')}
    ways, use = [], {}
    for w in root.iter('way'):
        tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
        if tags.get('highway') not in STREETS: continue
        refs = [nd.get('ref') for nd in w.iter('nd') if nd.get('ref') in nodes]
        if len(refs) < 2: continue
        ways.append((tags, refs))
        for r in refs: use[r] = use.get(r, 0) + 1
    clear = [(nodes[r], 8.) for r, k in use.items() if k >= 2] + [(nodes[i], 3.) for i, t in ntags.items() if i in nodes and t.get('highway') in ('crossing', 'traffic_signals')]
    clear_p = np.array([p for p, _ in clear]); clear_r = np.array([r for _, r in clear])
    doors = garage_doors(json.load(open(area / 'regen/profiles.json'))['profiles'])
    door_p = np.array([c for c, _ in doors]) if doors else np.zeros((0, 2)); door_r = np.array([h + .5 for _, h in doors]) if doors else np.zeros(0)
    z_at = lambda p: float(fc.ground_at(p))

    from shapely.strtree import STRtree
    outlines = [g for g in fc.load_footprints(frame).values() if not g.is_empty]; otree = STRtree(outlines)
    def free(p):
        if any(outlines[j].contains(Point(*p)) for j in otree.query(Point(*p))): return False      # never inside a building
        return not ((len(clear_p) and (np.hypot(*(clear_p - p).T) < clear_r).any()) or (len(door_p) and (np.hypot(*(door_p - p).T) < door_r + 3.).any()))

    def frame_at(line, s):
        a = np.asarray(line.interpolate(s).coords[0]); b = np.asarray(line.interpolate(min(line.length, s + .5)).coords[0])
        if s + .5 > line.length: b = a + (a - np.asarray(line.interpolate(max(0., s - .5)).coords[0]))
        d = b - a; d /= np.linalg.norm(d); return a, d, np.array([-d[1], d[0]])

    def kerbs_at(line, s):
        """(left kerb, right kerb) as (point, distance) at station s, or None where the road is not a plain street there."""
        a, d, n = frame_at(line, s)
        if not inside.contains(Point(*a)): return None
        kl, kr = kerb(a, n), kerb(a, -n)
        if kl is None or kr is None or not (2.5 < kl[1] < 14.) or not (2.5 < kr[1] < 14.): return None
        for t in (s - 3., s + 3.):          # a kerb that jumps 3 m either side is a corner or a cross street
            if 0 <= t <= line.length:
                b, _, m = frame_at(line, t); jl, jr = kerb(b, m), kerb(b, -m)
                if jl is None or jr is None or abs(jl[1] - kl[1]) > .8 or abs(jr[1] - kr[1]) > .8: return None
        return a, d, n, kl, kr

    poles, wires, stats = [], [], {'trolleyM': 0., 'litM': 0.}
    for tags, refs in ways:
        line = LineString([tuple(nodes[r]) for r in refs]); L = line.length
        if L < 12 or not line.intersects(road): continue
        trolley = tags.get('trolley_wire') == 'yes'
        lit = tags.get('lit') == 'yes' or (tags.get('lit') != 'no' and tags.get('highway') in ('residential', 'tertiary', 'secondary', 'primary', 'unclassified', 'living_street'))
        if not (trolley or lit): continue
        span = TROLLEY_SPAN if trolley else LAMP_SPAN
        stations = []
        for k, s in enumerate(np.arange(span / 2, L - 2., span)):
            q = kerbs_at(line, s)
            if q is None: continue
            a, d, n, kl, kr = q
            pl, pr = kl[0] + n * BEHIND_KERB, kr[0] - n * BEHIND_KERB
            if trolley:
                if not (free(pl) and free(pr)): continue
                stations.append((s, kl[1], kr[1]))
                for side, p, out in ((0, pl, n), (1, pr, -n)):
                    poles.append({'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2), 'z': round(z_at(p) + .2, 2), 'h': 9.,
                                  'arm': [round(float(-out[0]), 3), round(float(-out[1]), 3)] if side == k % 2 else None})
                zl, zr = z_at(pl) + SPAN_Z, z_at(pr) + SPAN_Z; mid = (pl + pr) / 2
                wires.append({'r': .008, 'p': [[round(float(pl[0]), 2), round(float(pl[1]), 2), round(zl, 2)], [round(float(mid[0]), 2), round(float(mid[1]), 2), round((zl + zr) / 2 - .25, 2)], [round(float(pr[0]), 2), round(float(pr[1]), 2), round(zr, 2)]]})
            else:
                side = k % 2; p, out = (pl, n) if side == 0 else (pr, -n)
                if not free(p): continue
                poles.append({'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2), 'z': round(z_at(p) + .2, 2), 'h': 8.5, 'arm': [round(float(-out[0]), 3), round(float(-out[1]), 3)]})
        if trolley and len(stations) >= 2:
            # the contact wires run the street between its first and last span, over the middle of each travel lane, hung
            # at the spans and sagging between them; a one-way street carries one pair, a two-way street a pair each way
            s0, s1 = stations[0][0], stations[-1][0]; half = float(np.median([min(l, r) for _, l, r in stations]))
            lane = max(1.6, (half - PARKING_LANE) / 2) if half > PARKING_LANE + 3.2 else half / 2
            offsets = [0.] if tags.get('oneway') in ('yes', '-1') else [lane, -lane]
            spans = np.array([st[0] for st in stations])
            for off in offsets:
                for pair in (-PAIR, PAIR):
                    pts = []
                    for s in np.arange(s0, s1 + .01, 2.):
                        a, d, n = frame_at(line, s); p = a + n * (off + pair)
                        j = int(np.searchsorted(spans, s, side='right')) - 1; j = min(max(j, 0), len(spans) - 2)
                        t = (s - spans[j]) / max(1., spans[j + 1] - spans[j]); z = z_at(p) + .09 + WIRE_Z - SAG * 4 * t * (1 - t)
                        pts.append([round(float(p[0]), 2), round(float(p[1]), 2), round(z, 2)])
                    wires.append({'r': .011, 'p': pts})
            stats['trolleyM'] += s1 - s0
        elif lit: stats['litM'] += L
    out = COMPARE / f'{label}-overhead.json'; out.write_text(json.dumps({'poles': poles, 'wires': wires}))
    manifest = COMPARE / f'{label}.json'; m = json.loads(manifest.read_text()); m['overhead'] = f'/compare/{label}-overhead.json'; manifest.write_text(json.dumps(m))
    print(json.dumps({'poles': len(poles), 'lamps': sum(p['arm'] is not None for p in poles), 'wires': len(wires), 'trolleyM': round(stats['trolleyM']), 'litM': round(stats['litM'])}))


if __name__ == '__main__':
    main()
