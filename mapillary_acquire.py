"""Area-wide imagery acquisition. Spatial input is data; no location rules.

Discover from coverage vector tiles instead of repeatedly querying dense bbox
history. Rank within spatial cells by capture time and camera type. Credentials
remain in memory. The manifest is evidence discovery, not calibrated geometry.
"""
import argparse
import json
import math
import sys
import termios
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import mercantile
import mapbox_vector_tile


def read_token():
    settings = None
    if sys.stdin.isatty():
        settings = termios.tcgetattr(sys.stdin)
        quiet = termios.tcgetattr(sys.stdin)
        quiet[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSANOW, quiet)
    print('Ready for in-memory imagery credential', flush=True)
    try:
        return sys.stdin.readline().strip()
    finally:
        if settings:
            termios.tcsetattr(sys.stdin, termios.TCSANOW, settings)


def fetch(url, token, binary=False):
    # Never log provider URLs: they may contain signatures or credentials.
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'Authorization': 'OAuth ' + token})
            with urllib.request.urlopen(req, timeout=50) as r:
                body = r.read()
            return body if binary else json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(1 + attempt)
                continue
            raise RuntimeError('Provider HTTP ' + str(e.code)) from None
        except Exception:
            if attempt < 2:
                continue
            raise RuntimeError('Provider transport or decoding failure') from None


def discover(bbox, token, out):
    rows = {}
    tiles = list(mercantile.tiles(*bbox, zooms=14))
    for tile in tiles:
        url = f'https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{tile.z}/{tile.x}/{tile.y}/?' + urllib.parse.urlencode({'access_token': token})
        raw = fetch(url, token, binary=True)
        layers = mapbox_vector_tile.decode(raw)
        layer = layers.get('image')
        if layer is None:
            raise RuntimeError('Coverage tile has no image layer')
        bounds = mercantile.xy_bounds(tile)
        extent = layer['extent']
        for item in layer['features']:
            if item['geometry']['type'] != 'Point':
                continue
            x, y = item['geometry']['coordinates']
            # Decoder defaults to Cartesian Y-up; Web Mercator bounds also Y-up.
            mx = bounds.left + x / extent * (bounds.right - bounds.left)
            my = bounds.bottom + y / extent * (bounds.top - bounds.bottom)
            lon, lat = mercantile.lnglat(mx, my)
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
            rec = dict(item['properties'])
            rec['id'] = str(rec['id'])
            rec['tile_location'] = [lon, lat]
            rows[rec['id']] = rec
        print(json.dumps({'tilesCompleted': tiles.index(tile) + 1, 'tiles': len(tiles), 'discoveredImages': len(rows)}), flush=True)
    manifest = {'bbox': bbox, 'retrievedAt': datetime.now(timezone.utc).isoformat(), 'images': list(rows.values()), 'source': 'Mapillary computed coverage tiles; locations quantized to tile grid'}
    (out/'coverage.json').write_text(json.dumps(manifest, separators=(',', ':')))
    return manifest


def select(manifest, spacing=20, cap=160):
    bbox = manifest['bbox']
    lat0 = (bbox[1] + bbox[3])/2
    cells = {}
    for r in manifest['images']:
        lon, lat = r['tile_location']
        cell = (int((lon-bbox[0])*111320*math.cos(math.radians(lat0))/spacing), int((lat-bbox[1])*111320/spacing))
        # Freshness first. Within the same calendar month prefer panorama.
        rank = (int(r.get('captured_at', 0))//(30*86400000), bool(r.get('is_pano')), int(r.get('captured_at', 0)))
        if cell not in cells or rank > cells[cell][0]:
            cells[cell] = (rank, r)
    candidates = [v[1] for v in cells.values()]
    if len(candidates) <= cap:
        return candidates
    # Farthest-point selection retains geographical coverage under a hard budget.
    chosen = [max(candidates, key=lambda r: r.get('captured_at', 0))]
    candidates.remove(chosen[0])
    def distance(a, b):
        x, y = a['tile_location']; xx, yy = b['tile_location']
        return ((x-xx)*math.cos(math.radians(lat0)))**2 + (y-yy)**2
    nearest = {r['id']: distance(r, chosen[0]) for r in candidates}
    while candidates and len(chosen) < cap:
        r = max(candidates, key=lambda r: nearest[r['id']])
        chosen.append(r); candidates.remove(r)
        for other in candidates:
            nearest[other['id']] = min(nearest[other['id']], distance(other, r))
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=160)
    p.add_argument('--reuse-coverage', action='store_true')
    p.add_argument('--download', action='store_true')
    args = p.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    token = read_token()
    job = json.loads(args.job.read_text())
    lat, lon = job['center']['lat'], job['center']['lon']
    dy = (job['contextRadiusM'] + 100)/111320; dx = dy/math.cos(math.radians(lat))
    bbox = [lon-dx, lat-dy, lon+dx, lat+dy]
    manifest = json.loads((out/'coverage.json').read_text()) if args.reuse_coverage else discover(bbox, token, out)
    chosen = select(manifest, cap=args.limit)
    fields = 'id,captured_at,computed_geometry,geometry,computed_altitude,altitude,computed_compass_angle,compass_angle,computed_rotation,camera_type,camera_parameters,width,height,is_pano,sequence,thumb_2048_url,thumb_original_url,sfm_cluster,mesh'
    def entity(row):
        dest = out/'entities'/f"{row['id']}.json"; dest.parent.mkdir(exist_ok=True)
        try:
            if dest.exists():
                rec = json.loads(dest.read_text())
            else:
                rec = fetch('https://graph.mapillary.com/' + row['id'] + '?' + urllib.parse.urlencode({'fields': fields}), token)
                dest.write_text(json.dumps(rec, separators=(',', ':')))
            if args.download:
                target = out/'images'/f"{row['id']}.jpg"; target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    # No API credential is sent to CDN download addresses.
                    url = rec.get('thumb_2048_url')
                    if url and urllib.parse.urlsplit(url).scheme == 'https':
                        with urllib.request.urlopen(url, timeout=50) as r:
                            target.write_bytes(r.read())
                rec = {**rec, 'local_image': str(target.resolve()) if target.exists() else None}
            return rec
        except Exception as e:
            return {'id': row['id'], 'error': str(e) if isinstance(e, RuntimeError) else 'Metadata or image download failure'}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rec in pool.map(entity, chosen):
            records.append(rec)
            if len(records)%20 == 0:
                print(json.dumps({'selected':len(chosen),'completed':len(records),'errors':sum('error' in r for r in records)}),flush=True)
    result = {'retrievedAt':datetime.now(timezone.utc).isoformat(), 'bbox':bbox, 'selection':'20m cells; most recent capture month then panorama; spatially distributed cap', 'calibrationValidated':False,'images':records}
    (out/'selected.json').write_text(json.dumps(result,separators=(',',':')))
    dates = [datetime.fromtimestamp(r['captured_at']/1000,timezone.utc).date().isoformat() for r in records if r.get('captured_at')]
    print(json.dumps({'discovered':len(manifest['images']),'selected':len(records),'downloaded':sum(bool(r.get('local_image')) for r in records),'errors':sum('error' in r for r in records),'oldest':min(dates) if dates else None,'newest':max(dates) if dates else None}),flush=True)


if __name__ == '__main__':
    main()
