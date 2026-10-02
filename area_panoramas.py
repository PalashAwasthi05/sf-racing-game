"""Street panoramas of one model area, into <area>/imagery in the pilot's form (coverage.json, entities/, images/,
source-index-registered.json with the provider's computed pose, registration grade 'unregistered').

  discover: the provider's coverage tiles over the area's context radius (+100 m) -> coverage.json, and a count of
            the posed-panorama candidates (captured 2023 or later, within the context radius + margin) to fetch.
  fetch:    full-resolution panoramas of those candidates; an interrupted run keeps what it fetched and resumes.

Credential handling is the pilot's: the access token is read from stdin with echo off and kept in memory, sent only
to the provider API (and its coverage tiles), never to the image CDN (signed downloads, fetched without credentials);
neither the token nor any provider URL is printed or logged.
Usage: area_panoramas.py <area dir> discover|fetch [--radius-margin 20] [--workers 4] [--limit N]"""
import argparse, hashlib, json, math, sys, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapillary_acquire as ma, fetch_panoramas_v32 as fp

SINCE_MS = 1672531200000   # 2023-01-01

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('area', type=Path); ap.add_argument('step', choices=['discover', 'fetch'])
    ap.add_argument('--radius-margin', type=float, default=20.); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); img = a.area / 'imagery'; (img / 'images').mkdir(parents=True, exist_ok=True); (img / 'entities').mkdir(exist_ok=True)
    job = json.loads((a.area / 'area-source-job.json').read_text()); lat0, lon0 = job['center']['lat'], job['center']['lon']
    radius = job['contextRadiusM'] + a.radius_margin
    def dist(lon, lat): return math.hypot((lon - lon0) * 111320 * math.cos(math.radians(lat0)), (lat - lat0) * 111320)
    print('Ready for in-memory imagery credential (input hidden)', flush=True)
    token = fp.read_token()
    if not token: sys.exit('no credential on stdin')
    if a.step == 'discover':
        dy = (job['contextRadiusM'] + 100) / 111320; dx = dy / math.cos(math.radians(lat0))
        manifest = ma.discover([lon0 - dx, lat0 - dy, lon0 + dx, lat0 + dy], token, img)
    reg_path = img / 'source-index-registered.json'; reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    have = {k.split(':', 1)[1] for k in reg}
    cov = json.loads((img / 'coverage.json').read_text())['images']
    cands = [str(c['id']) for c in cov if c.get('is_pano') and c.get('captured_at', 0) >= SINCE_MS and str(c['id']) not in have and dist(*c['tile_location']) <= radius + 30]
    print(json.dumps({'coverageImages': len(cov), 'panoramaCandidates': len(cands), 'alreadyCached': len(have)}), flush=True)
    if a.step == 'discover': return
    if a.limit: cands = cands[:a.limit]
    stats = {'fetched': 0, 'rejected': 0, 'failed': 0, 'bytes': 0}; new = {}
    def one(iid):
        e = fp.api(iid, token)
        cg, rot = e.get('computed_geometry'), e.get('computed_rotation')
        if not (e.get('is_pano') and e.get('camera_type') == 'spherical' and cg and rot and e.get('thumb_original_url')): return iid, None, 0
        lon, lat = cg['coordinates']
        if dist(lon, lat) > radius: return iid, None, 0
        body = fp.download(e['thumb_original_url'])
        (img / 'images' / f'{iid}.jpg').write_bytes(body); (img / 'entities' / f'{iid}.json').write_text(json.dumps(e))
        rec = {'provider': 'Mapillary', 'sourceUrl': f'https://www.mapillary.com/app/?pKey={iid}', 'imagePath': str((img / 'images' / f'{iid}.jpg').resolve()),
               'imageSha256': hashlib.sha256(body).hexdigest(), 'captureDate': datetime.fromtimestamp(e['captured_at'] / 1000, tz=timezone.utc).date().isoformat(),
               'viewGroup': e.get('sequence'),
               'camera': {'model': 'spherical', 'parameters': [], 'rotationAngleAxis': rot, 'locationWgs84': [lon, lat], 'altitudeM': e.get('computed_altitude'),
                          'worldRegistrationValidated': False, 'providerPoseConsistent': True, 'orientation': 1, 'clusterId': (e.get('sfm_cluster') or {}).get('id'),
                          'registrationGrade': 'unregistered', 'registrationEvidence': {'reason': 'provider computed pose, fetched by area_panoramas.py; not registered against the world'}},
               'admission': 'pose-registration-required', 'batch': 'area_panoramas'}
        return iid, rec, len(body)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, iid) for iid in cands]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                iid, rec, n = f.result()
                if rec: new['mapillary:' + iid] = rec; stats['fetched'] += 1; stats['bytes'] += n
                else: stats['rejected'] += 1
            except Exception:
                stats['failed'] += 1
            if k % 50 == 0 or k == len(futs):
                print(json.dumps({'done': k, 'of': len(futs), **stats}), flush=True)
                if new: reg.update(new); reg_path.write_text(json.dumps(reg, indent=1)); new = {}
    reg.update(new); reg_path.write_text(json.dumps(reg, indent=1)); print(json.dumps({'final': stats, 'indexRecords': len(reg)}), flush=True)

if __name__ == '__main__':
    main()
