"""Fetch the recent street panoramas of the model area that are not cached yet, into imagery-street-v12 in the same
form as the first 160 (entity JSON, full-resolution panorama, a record in source-index-registered.json with the
provider's computed pose, registration grade 'unregistered').

Credential handling: the access token is read from stdin (not echoed) and kept in memory; it is sent only to
graph.mapillary.com in an Authorization header, never to the image CDN (thumbnail URLs are signed downloads and are
fetched without credentials), and neither the token nor any provider URL is printed or logged.
Candidates: coverage.json panoramas captured 2023 or later within the context radius (+ margin).
Usage: <token source> | fetch_panoramas_v32.py [--radius-margin 20] [--limit N] [--workers 4]"""
import argparse, hashlib, json, math, sys, termios, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

W = Path(__file__).resolve().parent; IMG = W / 'imagery-street-v12'
FIELDS = 'id,captured_at,computed_geometry,geometry,computed_altitude,altitude,computed_compass_angle,compass_angle,computed_rotation,camera_type,width,height,is_pano,sequence,sfm_cluster,thumb_original_url,thumb_2048_url'
CDN_HOSTS = ('fbcdn.net', 'mapillary.com', 'fbsbx.com')


def read_token():
    settings = None
    if sys.stdin.isatty():
        settings = termios.tcgetattr(sys.stdin); quiet = termios.tcgetattr(sys.stdin); quiet[3] &= ~termios.ECHO; termios.tcsetattr(sys.stdin, termios.TCSANOW, quiet)
    try: return sys.stdin.readline().strip()
    finally:
        if settings: termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)


def api(image_id, token):
    url = 'https://graph.mapillary.com/' + urllib.parse.quote(str(image_id)) + '?' + urllib.parse.urlencode({'fields': FIELDS})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': 'OAuth ' + token}), timeout=50) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3: time.sleep(2 * (attempt + 1)); continue
            raise RuntimeError('provider HTTP %d' % e.code) from None
        except Exception:
            if attempt < 3: time.sleep(1 + attempt); continue
            raise RuntimeError('provider transport failure') from None


def download(url):
    """A signed CDN download: no credential header, https to the provider's image hosts only."""
    host = urllib.parse.urlsplit(url).hostname or ''
    if urllib.parse.urlsplit(url).scheme != 'https' or not host.endswith(CDN_HOSTS): raise RuntimeError('unexpected image host')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r: return r.read()
        except Exception:
            if attempt < 3: time.sleep(2 * (attempt + 1)); continue
            raise RuntimeError('image download failure') from None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--radius-margin', type=float, default=20.); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(); token = read_token()
    if not token: sys.exit('no credential on stdin')
    job = json.loads((W / 'area-source-job.json').read_text()); lat0, lon0 = job['center']['lat'], job['center']['lon']; radius = job['contextRadiusM'] + a.radius_margin
    reg_path = IMG / 'source-index-registered.json'; reg = json.loads(reg_path.read_text()); have = {k.split(':', 1)[1] for k in reg}
    def dist(lon, lat): return math.hypot((lon - lon0) * 111320 * math.cos(math.radians(lat0)), (lat - lat0) * 111320)
    cov = json.loads((IMG / 'coverage.json').read_text())['images']
    cands = [str(c['id']) for c in cov if c.get('is_pano') and c.get('captured_at', 0) >= 1672531200000 and str(c['id']) not in have and dist(*c['tile_location']) <= radius + 30]
    if a.limit: cands = cands[:a.limit]
    print(json.dumps({'candidates': len(cands), 'alreadyCached': len(have)}), flush=True)
    stats = {'fetched': 0, 'rejected': 0, 'failed': 0}; new = {}
    def one(iid):
        e = api(iid, token)
        cg, rot = e.get('computed_geometry'), e.get('computed_rotation')
        if not (e.get('is_pano') and e.get('camera_type') == 'spherical' and cg and rot and e.get('thumb_original_url')): return iid, None, 'not a posed panorama'
        lon, lat = cg['coordinates']
        if dist(lon, lat) > radius: return iid, None, 'outside the area'
        body = download(e['thumb_original_url'])
        (IMG / 'images' / f'{iid}.jpg').write_bytes(body); (IMG / 'entities' / f'{iid}.json').write_text(json.dumps(e))
        rec = {'provider': 'Mapillary', 'sourceUrl': f'https://www.mapillary.com/app/?pKey={iid}', 'imagePath': str(IMG / 'images' / f'{iid}.jpg'), 'imageSha256': hashlib.sha256(body).hexdigest(),
               'captureDate': datetime.fromtimestamp(e['captured_at'] / 1000, tz=timezone.utc).date().isoformat(), 'viewGroup': e.get('sequence'),
               'camera': {'model': 'spherical', 'parameters': [], 'rotationAngleAxis': rot, 'locationWgs84': [lon, lat], 'altitudeM': e.get('computed_altitude'), 'worldRegistrationValidated': False,
                          'providerPoseConsistent': True, 'orientation': 1, 'clusterId': (e.get('sfm_cluster') or {}).get('id'), 'registrationGrade': 'unregistered',
                          'registrationEvidence': {'reason': 'provider computed pose, fetched by fetch_panoramas_v32.py; not registered against the world'}},
               'admission': 'pose-registration-required', 'batch': 'fetch_panoramas_v32'}
        return iid, rec, None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, iid) for iid in cands]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                iid, rec, why = f.result()
                if rec: new['mapillary:' + iid] = rec; stats['fetched'] += 1
                else: stats['rejected'] += 1
            except Exception as exc:
                stats['failed'] += 1
            if k % 50 == 0 or k == len(futs):
                print(json.dumps({'done': k, **stats}), flush=True)
                if new:                                            # save progress: an interrupted run keeps what it fetched
                    reg.update(new); reg_path.write_text(json.dumps(reg, indent=1)); new = {}
    reg.update(new); reg_path.write_text(json.dumps(reg, indent=1)); print(json.dumps({'final': stats, 'indexRecords': len(reg)}), flush=True)


if __name__ == '__main__':
    main()
