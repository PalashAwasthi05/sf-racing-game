"""Parked cars for an area's local preview: where the kerbs of the built road surface meet the streets OpenStreetMap
marks for parking, a row of parked cars, written beside the preview's manifest for the viewer to instance.

The kerb is the edge of the road surface the generator built (its `roads` mesh), found square to each street's
centreline on either side; the parking lane runs 1.15 m inside it. A side is parked where its tags say so
(parking:<side> / parking:both = lane | street_side ..., or the older parking:lane:<side> = parallel | diagonal ...),
and nowhere they say no; an untagged residential, tertiary, secondary or unclassified street is parked both sides, a
primary or service road is not. The row stops short of intersections (12 m), crossings (5 m), signals (10 m) and bus
stops (15 m), and leaves every garage door the photographed facades show clear. Cars face the traffic of their own
side and take the slope of the street they stand on. Location-free: every rule reads the area's own data.
Usage: parking.py <area dir> <build label> [--glb model.glb] [--profiles profiles.json]
       (SFW_AREA set to the area, unset for King Street; any working directory)"""
import json, math, struct, sys, os, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree
from shapely.ops import unary_union
WORK = Path(__file__).resolve().parents[1]; COMPARE = WORK / 'king-street/public/compare'
sys.path.insert(0, str(WORK))
import frontage_crops_v15 as fc

ROADS = ('trunk', 'trunk_link', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'living_street', 'service', 'primary_link', 'secondary_link', 'tertiary_link')
PARKED_BY_DEFAULT = ('secondary', 'tertiary', 'residential', 'unclassified', 'living_street')
ALLOWED = {'lane', 'street_side', 'on_street', 'half_on_kerb', 'on_kerb', 'parallel', 'diagonal', 'perpendicular', 'marked', 'yes', 'shoulder'}
REFUSED = {'no', 'no_parking', 'no_stopping', 'no_standing', 'fire_lane', 'separate'}
# body types: length, width, height (m), and how common each is on a city street
TYPES = [('sedan', 4.8, 1.82, 1.45, .30), ('hatch', 4.15, 1.76, 1.48, .18), ('suv', 4.75, 1.9, 1.72, .26), ('wagon', 4.7, 1.8, 1.5, .08),
         ('pickup', 5.5, 2.0, 1.85, .06), ('van', 5.1, 2.0, 2.0, .05), ('coupe', 4.5, 1.8, 1.32, .07)]
# paint colours in roughly the shares cars are sold in (white, black, greys and silver dominate)
PAINT = [('#f2f2ef', .24), ('#18191b', .21), ('#6b6e72', .16), ('#b8bbbe', .13), ('#24395e', .09), ('#7e1418', .08), ('#3c4a3a', .02),
         ('#8a6a3c', .02), ('#5a2a3c', .02), ('#c9b48a', .02), ('#2f6f8f', .01)]
LANE_FROM_KERB, STEP = 1.15, .5


def ground_triangles(glb, prefix):
    """The triangles (n, 3 corners, xyz) of the model's meshes whose node names start with `prefix` ('roads',
    'sidewalks'), vertical faces left out. Vertices are in the world frame (x east, y north, z up); every node carries
    only the turn to glTF's y-up, which the viewer applies."""
    f = open(glb, 'rb'); f.read(12); jl, _ = struct.unpack('<II', f.read(8)); j = json.loads(f.read(jl)); f.read(8); b0 = f.tell()
    def read(acc):
        a = j['accessors'][acc]; v = j['bufferViews'][a['bufferView']]; f.seek(b0 + v.get('byteOffset', 0) + a.get('byteOffset', 0))
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a['componentType']]; k = {'VEC3': 3, 'SCALAR': 1}[a['type']]
        return np.frombuffer(f.read(a['count'] * k * np.dtype(dt).itemsize), dt).reshape(-1, k) if k > 1 else np.frombuffer(f.read(a['count'] * np.dtype(dt).itemsize), dt)
    tris = []
    for node in (n for n in j['nodes'] if n.get('name', '').startswith(prefix) and 'mesh' in n):
        for p in j['meshes'][node['mesh']]['primitives']:
            V = read(p['attributes']['POSITION']); I = read(p['indices']).reshape(-1, 3) if 'indices' in p else np.arange(len(V)).reshape(-1, 3)
            tris.append(V[I].astype(float))
    T = np.concatenate(tris)
    plan = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
    return T[np.abs(plan) > 1e-6]


def roads_outline(glb):
    """The built road surface as a 2-D outline, from the model's `roads` mesh."""
    return unary_union([Polygon(t[:, :2]) for t in ground_triangles(glb, 'roads')]).buffer(.02).buffer(-.02)


def surface(tris):
    """z(p): the height of the highest of these triangles over the plan point p, or None off them. Parked cars stand on
    the built road and poles on the built sidewalk, wherever the generator draped them."""
    tree = STRtree([Polygon(t[:, :2]) for t in tris])
    def z(p):
        best = None
        for i in tree.query(Point(float(p[0]), float(p[1]))):
            (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = tris[i]; d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            l0 = ((y1 - y2) * (p[0] - x2) + (x2 - x1) * (p[1] - y2)) / d; l1 = ((y2 - y0) * (p[0] - x2) + (x0 - x2) * (p[1] - y2)) / d
            if min(l0, l1, 1. - l0 - l1) < -1e-6: continue
            h = l0 * z0 + l1 * z1 + (1. - l0 - l1) * z2
            if best is None or h > best: best = h
        return best
    return z


def inputs():
    """(area dir, build label, model GLB, photo readings) from the command line. The model defaults to the build's
    materials/candidate.glb and the readings to the area's regen/profiles.json (none when it has none); --glb and
    --profiles name others (a published model reassembled from its chunks, a world outside areas/)."""
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('area'); ap.add_argument('label'); ap.add_argument('--glb'); ap.add_argument('--profiles')
    a = ap.parse_args(); area, label = Path(a.area).resolve(), a.label
    build = area / label.split('-')[-1] if (area / label.split('-')[-1]).exists() else area / label
    readings = Path(a.profiles) if a.profiles else area / 'regen/profiles.json'
    return area, label, Path(a.glb) if a.glb else build / 'materials/candidate.glb', json.load(open(readings))['profiles'] if readings.exists() else []


def kerbs(road, reach=14.):
    """(prepared road, kerb(a, n)): kerb gives the nearest point of the road's edge on the ray from a along n, and its
    distance, or None within `reach` metres."""
    inside = prep(road)
    segs = [LineString([a, b]) for ring in getattr(road.boundary, 'geoms', [road.boundary]) for a, b in zip(ring.coords, ring.coords[1:])]
    tree = STRtree(segs)
    def kerb(a, n):
        ray = LineString([tuple(a), tuple(a + n * reach)]); best = None
        for g in tree.query(ray):
            h = ray.intersection(segs[g])
            for q in getattr(h, 'geoms', [h]):
                if q.is_empty or q.geom_type != 'Point': continue
                q = np.asarray(q.coords[0]); dd = float(np.linalg.norm(q - a))
                if best is None or dd < best[1]: best = (q, dd)
        return best
    return inside, kerb


def side_rule(tags, side, highway):
    """('parallel' | 'diagonal' | 'perpendicular') when a side is parked, else None."""
    for key in (f'parking:{side}', 'parking:both', f'parking:lane:{side}', 'parking:lane:both'):
        v = tags.get(key)
        if v is None: continue
        if v in REFUSED or tags.get(f'parking:{side}:restriction', tags.get('parking:both:restriction')) in REFUSED: return None
        if v in ALLOWED:
            o = tags.get(f'parking:{side}:orientation', tags.get('parking:both:orientation', v if v in ('parallel', 'diagonal', 'perpendicular') else 'parallel'))
            return o if o in ('parallel', 'diagonal', 'perpendicular') else 'parallel'
    return 'parallel' if highway in PARKED_BY_DEFAULT else None


def garage_doors(profiles):
    """Points (world x, y) and half-widths of the garage doors the photographed facades show."""
    out = []
    for p in profiles:
        n = np.asarray(p['normal'][:2], float); n /= np.linalg.norm(n); t = np.array([-n[1], n[0]]); u0, u1 = p['frameU']; W = u1 - u0
        for o in p.get('openings', []) + [{'rect': g['region'], 'style': g.get('style', {})} for g in p.get('groups', [])]:
            if o.get('style', {}).get('kind') != 'rollup': continue
            x0, _, x1, _ = o['rect']; c = n * p['planeV'] + t * (u0 + (x0 + x1) / 2 * W)
            out.append((c, (x1 - x0) * W / 2))
    return out


def main():
    area, label, glb, profiles = inputs()
    world, frame = fc.load_world(); rng = np.random.default_rng(1906)
    road = roads_outline(glb); inside, kerb = kerbs(road); road_z = surface(ground_triangles(glb, 'roads'))
    root = ET.parse(fc.OSM_PATH).getroot()
    nodes = {n.get('id'): frame.local(float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
    ntags = {n.get('id'): {t.get('k'): t.get('v') for t in n.iter('tag')} for n in root.iter('node')}
    ways = []
    for w in root.iter('way'):
        tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
        if tags.get('highway') not in ROADS: continue
        refs = [nd.get('ref') for nd in w.iter('nd') if nd.get('ref') in nodes]
        if len(refs) >= 2: ways.append((tags, refs))
    use = {}
    for _, refs in ways:
        for r in refs: use[r] = use.get(r, 0) + 1
    def pts(ids): return [np.array([float(nodes[i][0]), float(nodes[i][1])]) for i in ids]
    keep_clear = [(p, 12.) for p in pts([r for r, k in use.items() if k >= 2])]
    for i, tg in ntags.items():
        if i not in nodes: continue
        hw = tg.get('highway'); r = {'crossing': 5., 'traffic_signals': 10., 'bus_stop': 15., 'stop': 4.}.get(hw) or (15. if tg.get('public_transport') == 'platform' else None)
        if r: keep_clear.append((np.array([float(nodes[i][0]), float(nodes[i][1])]), r))
    doors = garage_doors(profiles)
    clear_pts = np.array([p for p, _ in keep_clear]); clear_r = np.array([r for _, r in keep_clear])
    door_pts = np.array([c for c, _ in doors]) if doors else np.zeros((0, 2)); door_r = np.array([h + .9 for _, h in doors]) if doors else np.zeros(0)
    types = [t for t in TYPES]; tw = np.array([t[4] for t in TYPES]); tw /= tw.sum(); pw = np.array([c[1] for c in PAINT]); pw /= pw.sum()
    cars = []; sides_parked = 0
    for tags, refs in ways:
        line = LineString(pts(refs))
        if line.length < 10: continue
        for side, sgn in (('right', -1.), ('left', 1.)):
            if side_rule(tags, side, tags.get('highway')) is None: continue
            # stations along the centreline, each with its kerb on this side; a station whose ray finds no kerb
            # within 14 m, or a kerb that jumps from its neighbours (a cross street, a corner), parks nothing
            lane = []
            for s in np.arange(0., line.length, STEP):
                a = np.asarray(line.interpolate(s).coords[0]); b = np.asarray(line.interpolate(min(line.length, s + .5)).coords[0])
                if s + .5 > line.length: b = a + (a - np.asarray(line.interpolate(s - .5).coords[0]))
                d = b - a; d /= np.linalg.norm(d); n = np.array([-d[1], d[0]]) * sgn
                if not inside.contains(Point(*a)): lane.append(None); continue
                hit = kerb(a, n)
                if hit is None: lane.append(None); continue
                k, dist = hit
                if dist < 3.2: lane.append(None); continue          # too narrow to park on this side
                lane.append((k - n * LANE_FROM_KERB, d * (1. if side == 'right' else -1.), dist, n))
            ok = []
            for i, q in enumerate(lane):
                if q is None: ok.append(False); continue
                nb = [lane[j][2] for j in (i - 2, i + 2) if 0 <= j < len(lane) and lane[j] is not None]
                c = q[0]
                # a garage door stands back across the sidewalk: it keeps clear the stretch of kerb in front of it
                rel = door_pts - c; along = np.abs(rel @ q[1]) if len(rel) else rel; across = rel @ q[3] if len(rel) else rel
                bad = any(abs(q[2] - x) > .8 for x in nb) or (len(clear_pts) and (np.hypot(*(clear_pts - c).T) < clear_r).any()) or (len(door_pts) and ((along < door_r) & (across > 0) & (across < 10.)).any())
                ok.append(not bad)
            # fill each clear run with cars, end to end with natural gaps, some spaces left empty
            i = 0; parked_here = 0
            while i < len(lane):
                if not ok[i]: i += 1; continue
                j = i
                while j < len(lane) and ok[j]: j += 1
                run = lane[i:j]; pos = rng.uniform(.3, 1.5)
                while True:
                    t = types[rng.choice(len(types), p=tw)]; L = t[1] + rng.uniform(-.15, .15)
                    if pos + L > len(run) * STEP: break
                    if rng.random() < .86:
                        mid = run[min(len(run) - 1, int((pos + L / 2) / STEP))]; c, hd = mid[0], mid[1]
                        zs = [road_z(q) for q in (c + hd * L / 2, c - hd * L / 2, c)]
                        zf, zb, zc = zs if None not in zs else [float(fc.ground_at(q)) + .09 for q in (c + hd * L / 2, c - hd * L / 2, c)]
                        pitch = math.atan2(zf - zb, L)          # nose up on a rising street
                        cars.append({'x': round(float(c[0]), 2), 'y': round(float(c[1]), 2), 'z': round(float(zc), 2), 'heading': round(math.atan2(hd[1], hd[0]), 4),
                                     'pitch': round(float(pitch), 4), 'type': t[0], 'paint': PAINT[rng.choice(len(PAINT), p=pw)][0]})
                        parked_here += 1
                    pos += L + rng.uniform(.7, 2.6)
                i = j
            sides_parked += parked_here > 0
    # where two ways of one street meet mid-block, their rows can overlap: a car overlapping one already kept is dropped
    from shapely.geometry import Polygon
    kept, boxes, tw_ = [], [], {t[0]: t for t in TYPES}
    for c in cars:
        t = tw_[c['type']]; u = np.array([np.cos(c['heading']), np.sin(c['heading'])]); n = np.array([-u[1], u[0]]); m = np.array([c['x'], c['y']])
        b = Polygon([m + u * a * t[1] / 2 + n * e * t[2] / 2 for a, e in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        if any(b.intersects(q) and b.intersection(q).area > .05 for q in boxes[-60:]) or any(b.intersects(q) and b.intersection(q).area > .05 for q in boxes[:-60] if q.distance(b) < .5): continue
        kept.append(c); boxes.append(b)
    dropped = len(cars) - len(kept); cars = kept
    out = COMPARE / f'{label}-cars.json'
    out.write_text(json.dumps({'types': [{'name': t[0], 'length': t[1], 'width': t[2], 'height': t[3]} for t in TYPES], 'cars': cars}))
    manifest = COMPARE / f'{label}.json'; m = json.loads(manifest.read_text()); m['cars'] = f'/compare/{label}-cars.json'; manifest.write_text(json.dumps(m))
    print(json.dumps({'cars': len(cars), 'overlapsDropped': dropped, 'streetSidesParked': sides_parked, 'garageDoorsKeptClear': len(doors), 'keepClearPoints': len(keep_clear)}))


if __name__ == '__main__':
    main()
