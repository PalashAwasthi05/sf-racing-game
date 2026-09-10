"""Upgrade recent imagery when the source has more pixels than the cached preview."""
import argparse,json,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--imagery',type=Path,required=True);p.add_argument('--as-of',required=True);a=p.parse_args();now=datetime.fromisoformat(a.as_of).replace(tzinfo=timezone.utc);d=json.loads((a.imagery/'selected.json').read_text())
def upgrade(rec):
 if rec.get('error') or not rec.get('local_image'):return 'acquisition-failed'
 if (now-datetime.fromtimestamp(rec['captured_at']/1000,timezone.utc)).days>730:return 'historical-retained'
 path=Path(rec['local_image']);width=Image.open(path).width
 if width>=rec.get('width',width):return 'already-full-resolution'
 url=rec.get('thumb_original_url')
 if not url:return 'original-unavailable'
 try:
  with urllib.request.urlopen(url,timeout=60) as response:raw=response.read(40_000_001)
  if len(raw)>40_000_000:return 'image-exceeded-byte-budget'
  import io
  with Image.open(io.BytesIO(raw)) as im:
   im.verify()
  path.write_bytes(raw);return 'upgraded'
 except Exception:return 'upgrade-failed-preview-retained'
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(upgrade,d['images']))
from collections import Counter
report=dict(Counter(results));(a.imagery/'resolution-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
