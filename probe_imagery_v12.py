"""Collect area imagery using an in-memory credential read from non-echoed stdin."""
import sys,termios,json,urllib.request,urllib.parse,urllib.error,math,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
w=Path(__file__).resolve().parent;out=w/'imagery-v12';out.mkdir(exist_ok=True)
if sys.stdin.isatty():
 settings=termios.tcgetattr(sys.stdin);quiet=termios.tcgetattr(sys.stdin);quiet[3]&=~termios.ECHO;termios.tcsetattr(sys.stdin,termios.TCSANOW,quiet)
print('Ready for in-memory imagery credential',flush=True)
token=sys.stdin.readline().strip()
if sys.stdin.isatty():termios.tcsetattr(sys.stdin,termios.TCSANOW,settings)
job=json.loads((w/'area-source-job.json').read_text());lat=job['center']['lat'];lon=job['center']['lon'];radius=35;dy=radius/111320;dx=dy/math.cos(math.radians(lat));bbox=[lon-dx,lat-dy,lon+dx,lat+dy]
fields='id'
url='https://graph.mapillary.com/images?'+urllib.parse.urlencode({'bbox':','.join(map(str,bbox)),'fields':fields,'limit':1,'start_captured_at':'2024-01-01T00:00:00Z'});rows=[]
while url and len(rows)<1:
 parsed=urllib.parse.urlsplit(url)
 if parsed.scheme!='https' or parsed.hostname!='graph.mapillary.com':raise RuntimeError('Unexpected pagination host')
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'OAuth '+token}),timeout=60) as response:data=json.load(response)
 except urllib.error.HTTPError as exc:
  # Never print request URLs or headers; pagination can embed authentication.
  error=json.loads(exc.read()).get('error',{});print(json.dumps({'status':exc.code,'error_type':error.get('type'),'error_code':error.get('code'),'message':error.get('message','').replace(token,'[redacted]')}),flush=True);sys.exit(1)
 rows.extend(data.get('data',[]));url=data.get('paging',{}).get('next');print(json.dumps({'metadataRecords':len(rows)}),flush=True)
# Thumbnail URLs are source download addresses, not API access credentials.
(out/'metadata.json').write_text(json.dumps({'bbox':bbox,'images':rows,'retrievedAt':datetime.now(timezone.utc).isoformat(),'recordLimitReached':len(rows)>=6000},separators=(',',':')))
dates=[datetime.fromtimestamp(r['captured_at']/1000,tz=timezone.utc).date().isoformat() for r in rows if r.get('captured_at')]
print(json.dumps({'records':len(rows),'sequences':len({str(r.get('sequence')) for r in rows}),'oldest':min(dates) if dates else None,'newest':max(dates) if dates else None,'calibrationFieldsPresent':sum(bool(r.get('camera_parameters')) and bool(r.get('computed_rotation')) for r in rows)}),flush=True)
