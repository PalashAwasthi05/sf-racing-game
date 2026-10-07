"""Build the backdrop around a model area for the viewer (work/king-street/app/horizon.ts): the real ground and water
outside San Francisco, out to 60 km. San Francisco itself is not built here: the city is what the project builds, so its
land outside the model is only ground at its real height (the viewer shades it neutral, with nothing on it).

Everything drawn is measured:
- ground and water: USGS 3DEP 1 arc-second elevation (work/data/horizon, fetched by fetch_horizon.py) on a polar grid
  round the model's centre, lowered for the earth's curvature (with standard refraction, radius 7/6 of the earth's);
  cells at or below sea level are water (the bay and the ocean). The ground there is sunk 3 m under the level the model
  draws its own water at, and the viewer lays one water surface at that level over everything: the shore is where the two
  meet, a smooth line, not the elevation grid's 30 m staircase.
  North of 38 N the tiles end: that ground is marked as having no data and left to the haze.
- San Francisco's mainland: from the DataSF building footprints (all within the city): their coverage, closed over parks
  and open spaces (700 m), holes filled, kept to land, and only the piece the model stands on. The viewer shades it as
  plain ground. The city's islands (Treasure Island, Yerba Buena, Alcatraz) are shaded as the land round the bay is.
Output: <out>.json (metadata) and <out>.bin (gzip): the grid's ring radii and per-vertex height, water class, slope and
aspect; the San Francisco mask raster. Usage:
  build_horizon.py --world <world.json> --out <dir>/<name> [--outer 60000] [--angles 1024] [--rings 360]"""
import argparse, gzip, json, math, pathlib, time
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data' / 'horizon'
TILES = ['n38w123', 'n38w122']
R_EFF = 6371000 * 7 / 6          # refraction-corrected earth radius for the horizon drop
WATER_BELOW = 1.3                # NAVD88 m: at or below this the DEM cell is water (mean sea level at SF is about 0.95)
WATER_LEVEL = 0.0                # NAVD88 m: the level the generator draws its own water at (the model's water meshes)

class Dem:
    """Bilinear samples of the mosaic of 1 arc-second tiles (NAD83 geographic, NAVD88 metres); `valid` marks covered points."""
    def __init__(self):
        self.tiles = []
        for t in TILES:
            with rasterio.open(DATA / f'USGS_1_{t}.tif') as d:
                a = d.read(1).astype(np.float32); a[a < -1000] = np.nan
                self.tiles.append((d.transform, a))
    def __call__(self, lon, lat):
        lon = np.asarray(lon, float).ravel(); lat = np.asarray(lat, float).ravel(); out = np.full(lon.shape, np.nan, np.float32)
        for tr, a in self.tiles:
            col = (lon - tr.c) / tr.a - .5; row = (lat - tr.f) / tr.e - .5
            ok = (col >= 0) & (row >= 0) & (col < a.shape[1] - 1) & (row < a.shape[0] - 1) & np.isnan(out)
            c0 = np.floor(col[ok]).astype(int); r0 = np.floor(row[ok]).astype(int); fc = col[ok] - c0; fr = row[ok] - r0
            out[ok] = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr) + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)
        self.valid = ~np.isnan(out)
        return np.where(self.valid, out, WATER_LEVEL)

def sf_mainland(to_utm, cx, cy, dem, to_ll, cell=10., close_m=700.):
    """San Francisco's mainland as a raster in local metres (x east, y north; row 0 is the north edge): building coverage
    closed over parks, holes filled, kept to land, the component holding the origin (the model)."""
    rows = json.load(open(DATA / 'sf-building-footprints.json'))
    local = lambda x, y, z=None: (np.asarray(to_utm.transform(x, y)[0]) - cx, np.asarray(to_utm.transform(x, y)[1]) - cy)
    shapes = []
    for r in rows:
        sh = r.get('shape')
        if not sh: continue
        try: shapes.append(shp_transform(local, shape(sh)))
        except Exception: continue
    b = np.array([s.bounds for s in shapes])
    pad = close_m + 200; x0, y0, x1, y1 = b[:, 0].min() - pad, b[:, 1].min() - pad, b[:, 2].max() + pad, b[:, 3].max() + pad
    W, H = int(math.ceil((x1 - x0) / cell)), int(math.ceil((y1 - y0) / cell)); tr = from_origin(x0, y1, cell, cell)
    cover = features.rasterize(((mapping(s), 1) for s in shapes), out_shape=(H, W), transform=tr, all_touched=True, dtype=np.uint8).astype(bool)
    r = close_m / cell
    dil = ndimage.distance_transform_edt(~cover) <= r                 # closing by distance transforms (a disc of close_m)
    closed = ndimage.distance_transform_edt(dil) > r
    closed |= cover; closed = ndimage.binary_fill_holes(closed)
    # land only: the bay and the ocean stay water
    gx = x0 + (np.arange(W) + .5) * cell; gy = y1 - (np.arange(H) + .5) * cell; GX, GY = np.meshgrid(gx, gy)
    lon, lat = to_ll.transform(cx + GX.ravel(), cy + GY.ravel()); h = dem(lon, lat).reshape(GX.shape)
    land = closed & (h > WATER_BELOW)
    lab, n = ndimage.label(land); ci, cj = int(y1 / cell), int(-x0 / cell)
    keep = lab[ci, cj] if lab[ci, cj] else int(np.bincount(lab.ravel())[1:].argmax()) + 1
    main = lab == keep
    main = ndimage.binary_dilation(main, iterations=4)      # 40 m over the shore: the land's last metres are the city's too
    return main.astype(np.uint8) * 255, (float(x0), float(y0), float(x1), float(y1)), int(n)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--world', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--outer', type=float, default=60000); ap.add_argument('--angles', type=int, default=1024); ap.add_argument('--rings', type=int, default=360)
    a = ap.parse_args(); t0 = time.time()
    world = json.load(open(a.world)); lat0, lon0 = world['center']['lat'], world['center']['lon']; vo = float(world['verticalOrigin'])
    inner = float(world.get('contextRadiusM') or 265)
    to_utm = Transformer.from_crs(4326, 32610, always_xy=True); to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    cx, cy = to_utm.transform(lon0, lat0); dem = Dem()
    # the grid: rings from the model's edge out, growing geometrically; angles counter-clockwise from east
    r0 = inner; k = math.log(a.outer / r0) / a.rings
    radii = r0 * np.exp(k * np.arange(a.rings + 1)); th = np.arange(a.angles) / a.angles * 2 * math.pi
    RR, TT = np.meshgrid(radii, th, indexing='ij'); X = RR * np.cos(TT); Y = RR * np.sin(TT)
    lon, lat = to_ll.transform(cx + X.ravel(), cy + Y.ravel()); h = dem(lon, lat).reshape(X.shape); valid = dem.valid.reshape(X.shape)
    water = (h <= WATER_BELOW).astype(np.uint8); water[~valid] = 2          # 2: no elevation data (left to the haze)
    hl = np.where(water > 0, WATER_LEVEL - 3., h) - vo - RR ** 2 / (2 * R_EFF)
    # slope (degrees) and how far the ground faces north (-1..1, weighted by slope), for the shading
    hs = np.where(water > 0, WATER_LEVEL, h) - vo - RR ** 2 / (2 * R_EFF)       # slopes of the surface as seen (water flat)
    gy = np.gradient(hs, axis=1) / (radii[:, None] * (2 * math.pi / a.angles)); gx = np.gradient(hs, radii, axis=0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    north = -(gx * np.sin(TT) + gy * np.cos(TT)) / np.maximum(np.hypot(gx, gy), 1e-6)
    mask, (mx0, my0, mx1, my1), parts = sf_mainland(to_utm, cx, cy, dem, to_ll)
    payload = {
        'terrain.radii': radii.astype(np.float32), 'terrain.height': hl.astype(np.float32), 'terrain.water': water,
        'terrain.slope': np.clip(slope, 0, 89).astype(np.uint8), 'terrain.north': (np.clip(north * np.clip(slope / 20, 0, 1), -1, 1) * 127).astype(np.int8),
        'sf.mask': mask,
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = {'angles': a.angles, 'rings': a.rings + 1, 'center': world['center'], 'verticalOrigin': vo, 'innerRadiusM': inner, 'outerRadiusM': a.outer,
            'waterLevelNAVD88': WATER_LEVEL, 'waterLevelLocal': WATER_LEVEL - vo, 'earthRadiusEffM': R_EFF, 'sfMask': {'bounds': [mx0, my0, mx1, my1], 'shape': list(mask.shape), 'landParts': parts},
            'arrays': {}, 'sources': [
                {'name': 'USGS 3DEP 1 arc-second DEM', 'tiles': TILES, 'url': 'https://www.usgs.gov/3d-elevation-program'},
                {'name': 'DataSF Building Footprints (city extent only)', 'id': 'ynuv-fyni', 'url': 'https://data.sfgov.org/d/ynuv-fyni'}]}
    blobs, off = [], 0
    for name, arr in payload.items():
        b = np.ascontiguousarray(arr).tobytes(); pad = (-off) % 4
        if pad: blobs.append(b'\0' * pad); off += pad
        meta['arrays'][name] = {'offset': off, 'dtype': str(arr.dtype), 'shape': list(arr.shape)}
        blobs.append(b); off += len(b)
    raw = b''.join(blobs); out.with_suffix('.bin').write_bytes(gzip.compress(raw, 6))
    meta['bytes'] = len(raw); meta['bin'] = out.name + '.bin'
    out.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    # the viewer finds an area's backdrop by its centre
    idx_path = out.parent / 'index.json'; index = json.loads(idx_path.read_text()) if idx_path.exists() else {'areas': []}
    index['areas'] = [e for e in index['areas'] if e['id'] != out.name] + [{'id': out.name, 'center': world['center'], 'meta': '/horizon/' + out.with_suffix('.json').name}]
    idx_path.write_text(json.dumps(index, indent=1))
    print(json.dumps({'terrainVertices': int(hl.size), 'water': round(float((water == 1).mean()), 3), 'noData': round(float((water == 2).mean()), 3), 'sfMask': list(mask.shape),
                      'rawMB': round(len(raw) / 1e6, 1), 'gzMB': round(out.with_suffix('.bin').stat().st_size / 1e6, 2), 'seconds': round(time.time() - t0)}))

if __name__ == '__main__':
    main()
