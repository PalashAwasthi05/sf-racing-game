"""Fetch the map inputs of one model area into <area>/data, in the same form the pilot's data/ holds them.

An area is a folder with its own config.json (region anchor way, study and context radii); everything it downloads
lands under it, nothing is shared with another area. City datasets come from DataSF (within a box round the centre),
the map from the OpenStreetMap Overpass API (full metadata, as the generator reads osm.xml), and the LiDAR points from
the public USGS 2023 San Francisco survey (EPT tiles to depth 8, clipped to the context radius plus a margin).
Already-fetched files are kept. Usage: area_fetch.py <area dir> --lat LAT --lon LON [--skip-lidar]"""
import argparse, concurrent.futures, json, math, pathlib, ssl, sys, time, urllib.parse, urllib.request
import numpy as np

UA = {'User-Agent': 'SFWorldResearchPilot/0.2'}
# this runtime ships no CA bundle: verify against certifi's, or the system's
try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
CITY = [('buildings', 'ynuv-fyni', 'shape'), ('trees', 'tkzw-k3nq', 'latlon'), ('streets', '3psu-pn9h', 'line'),
        ('signs', 'm48z-6ji4', 'shape'), ('row', 'h8n7-e4ns', 'the_geom'), ('curbs', 'emxt-b6yg', 'the_geom'),
        ('sidewalk-widths', '4g86-grxu', 'shape'), ('trash-cans', '5buk-jp7b', 'point'), ('bike-network', 'ygmz-vaxd', 'shape')]
META = ['buildings', 'row', 'signs', 'streets', 'trees']
EPT = 'https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SanFrancisco_1_B23/'

def get(url, data=None, timeout=180, tries=3):
    for k in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=timeout, context=CTX).read()
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(5 * (k + 1))

def city(data, name, dataset, field, box):
    dest = data / f'{name}.json'
    if dest.exists(): return name, 'cached', dest.stat().st_size
    s, w, n, e = box; rows, offset = [], 0
    while True:   # page through: a dense area holds more than one page of trees or curb segments
        where = (f'latitude::number between {s} and {n} and longitude::number between {w} and {e}' if field == 'latlon'   # a point column held as text
                 else f'within_box({field}, {n}, {w}, {s}, {e})')
        q = {'$where': where, '$limit': '20000', '$offset': str(offset), '$order': ':id'}
        url = f'https://data.sf.gov/resource/{dataset}.json?' + urllib.parse.urlencode(q)
        page = json.loads(get(url)); rows += page; offset += len(page)
        if len(page) < 20000: break
    if name == 'trees':   # the inventory renamed its columns; the pipeline reads the earlier names
        for t in rows:
            for k in ('species', 'siteinfo', 'legalstatus', 'caretaker', 'address'):
                if k in t and 'q' + k not in t: t['q' + k] = t[k]
            if 'mapdbh' in t and 'dbh' not in t: t['dbh'] = t['mapdbh']
    dest.write_text(json.dumps(rows))
    (data / f'{name}-query.json').write_text(json.dumps({'dataset': dataset, 'geometryField': field, 'box': box, 'rows': len(rows)}, indent=2))
    return name, len(rows), dest.stat().st_size

def meta(data, name, dataset):
    dest = data / f'{name}-meta.json'
    if not dest.exists(): dest.write_bytes(get(f'https://data.sf.gov/api/views/{dataset}.json'))

def osm(data, box):
    dest = data / 'osm.xml'
    if dest.exists(): return dest.stat().st_size
    s, w, n, e = box
    q = f'[timeout:300][maxsize:536870912];(node({s},{w},{n},{e});way({s},{w},{n},{e});relation({s},{w},{n},{e}););(._;>;);out meta;'
    raw = get('https://overpass-api.de/api/interpreter', data=urllib.parse.urlencode({'data': q}).encode(), timeout=400)
    if b'<osm' not in raw[:500]: raise RuntimeError('overpass returned no map')
    dest.write_bytes(raw); return len(raw)

def lidar(data, lat, lon, radius):
    import laspy
    from pyproj import Transformer
    cache = data / 'lidar'; cache.mkdir(exist_ok=True)
    def download(rel):
        f = cache / rel.replace('/', '_')
        if not f.exists(): f.write_bytes(get(EPT + rel, timeout=60))
        return f
    if not (data / 'ept.json').exists(): (data / 'ept.json').write_bytes(get(EPT + 'ept.json'))
    bounds = json.loads((data / 'ept.json').read_text())['bounds']
    cx, cy = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    r = radius / math.cos(math.radians(lat)); clip = [cx - r, cy - r, cx + r, cy + r]; maxdepth = 8
    def overlaps(k):
        d, x, y, z = map(int, k.split('-')); s = (bounds[3] - bounds[0]) / 2 ** d
        lo = [bounds[0] + x * s, bounds[1] + y * s, bounds[2] + z * s]
        return lo[0] <= clip[2] and lo[0] + s >= clip[0] and lo[1] <= clip[3] and lo[1] + s >= clip[1] and lo[2] < 300 and lo[2] + s > -100
    pending, visited, keys = ['0-0-0-0'], set(), []
    while pending:
        k = pending.pop()
        if k in visited: continue
        visited.add(k)
        for node, count in json.loads(download('ept-hierarchy/' + k + '.json').read_text()).items():
            if int(node.split('-')[0]) > maxdepth or not overlaps(node): continue
            if count == -1: pending.append(node)
            elif count > 0: keys.append(node)
    keys = sorted(set(keys)); print('lidar nodes', len(keys), flush=True)
    def process(key):
        las = laspy.read(download('ept-data/' + key + '.laz')); x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        m = (x >= clip[0]) & (x <= clip[2]) & (y >= clip[1]) & (y <= clip[3]) & (z > -100) & (z < 300)
        return np.stack([x[m], y[m], z[m], np.asarray(las.classification)[m], np.asarray(las.intensity)[m]], 1)
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, arr in enumerate(pool.map(process, keys)):
            out.append(arr)
            if i % 25 == 0: print('lidar', i + 1, '/', len(keys), sum(len(a) for a in out), 'points', flush=True)
    pts = np.concatenate(out); np.savez_compressed(data / 'points.npz', points=pts); return pts.shape

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('area'); ap.add_argument('--lat', type=float, required=True); ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--skip-lidar', action='store_true'); a = ap.parse_args()
    area = pathlib.Path(a.area); data = area / 'data'; data.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(area / 'config.json')); reach = float(cfg['region']['contextRadiusM']) + 60.   # margin for streets and trees at the edge
    dlat = reach / 111320.; dlon = reach / (111320. * math.cos(math.radians(a.lat)))
    box = (a.lat - dlat, a.lon - dlon, a.lat + dlat, a.lon + dlon)
    print('box', [round(v, 6) for v in box], flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for f in concurrent.futures.as_completed([pool.submit(city, data, *j, box) for j in CITY]): print(f.result(), flush=True)
    for name, dataset, _ in CITY:
        if name in META: meta(data, name, dataset)
    print('osm bytes', osm(data, box), flush=True)
    if not a.skip_lidar: print('lidar points', lidar(data, a.lat, a.lon, reach), flush=True)
    print('done', flush=True)
