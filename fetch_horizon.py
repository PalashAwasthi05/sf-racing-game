"""Fetch the inputs of the viewer's real horizon into work/data/horizon (kept local, not in git):
- USGS 3DEP 1 arc-second (~30 m) elevation tiles covering San Francisco, the bay, Marin and the East Bay hills
  (n38w123: 37-38 N, 123-122 W; n38w122: 37-38 N, 122-121 W), public domain;
- DataSF Building Footprints (ynuv-fyni) for the whole city, with only the fields the horizon's plain massing needs:
  the outline, the median and maximum LiDAR height above the ground, and the ground level.
Already-fetched files are kept. Usage: fetch_horizon.py"""
import json, pathlib, ssl, time, urllib.parse, urllib.request

UA = {'User-Agent': 'SFWorldResearchPilot/0.2'}
try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
OUT = pathlib.Path(__file__).resolve().parent / 'data' / 'horizon'
DEM = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/{t}/USGS_1_{t}.tif'
TILES = ['n38w123', 'n38w122']

def get(url, timeout=300, tries=4):
    for k in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=CTX).read()
        except Exception:
            if k == tries - 1: raise
            time.sleep(5 * (k + 1))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for t in TILES:
        dest = OUT / f'USGS_1_{t}.tif'
        if not dest.exists(): dest.write_bytes(get(DEM.format(t=t))); print('fetched', dest.name, dest.stat().st_size)
        else: print('cached', dest.name)
    dest = OUT / 'sf-building-footprints.json'
    if not dest.exists():
        rows, offset = [], 0
        while True:
            q = {'$select': 'sf16_bldgid,hgt_median_m,hgt_maxcm,gnd_min_m,shape', '$limit': '50000', '$offset': str(offset), '$order': ':id'}
            page = json.loads(get('https://data.sf.gov/resource/ynuv-fyni.json?' + urllib.parse.urlencode(q)))
            rows += page; offset += len(page); print('footprints', offset)
            if len(page) < 50000: break
        dest.write_text(json.dumps(rows))
        (OUT / 'sf-building-footprints-query.json').write_text(json.dumps({'dataset': 'ynuv-fyni', 'source': 'https://data.sfgov.org/d/ynuv-fyni', 'fields': q['$select'], 'rows': len(rows), 'fetched': time.strftime('%Y-%m-%d')}, indent=2))
    print('footprints at', dest, dest.stat().st_size)

if __name__ == '__main__':
    main()
