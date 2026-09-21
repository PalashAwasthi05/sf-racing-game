"""Optional imagery adapter. Area and credentials are external inputs; no named targets."""
import os,json,argparse,hashlib,urllib.parse,urllib.request
from pathlib import Path
from datetime import datetime,timezone

def fetch_metadata(bbox,token,limit=2000):
 fields='id,captured_at,computed_geometry,compass_angle,camera_type,sequence,thumb_2048_url'
 url='https://graph.mapillary.com/images?'+urllib.parse.urlencode({'bbox':','.join(map(str,bbox)),'fields':fields,'limit':min(500,limit)})
 items=[]
 while url and len(items)<limit:
  request=urllib.request.Request(url,headers={'Authorization':'OAuth '+token})
  with urllib.request.urlopen(request,timeout=45) as response:data=json.load(response)
  items.extend(data.get('data',[]));url=data.get('paging',{}).get('next')
 return items[:limit]

def collect(bbox,output,limit=2000,download_limit=0):
 token=os.environ.get('MAPILLARY_ACCESS_TOKEN')
 if not token:raise RuntimeError('Imagery access is not configured; no provider request attempted')
 out=Path(output);out.mkdir(parents=True,exist_ok=True);records=fetch_metadata(bbox,token,limit);manifest=[]
 for i,r in enumerate(records):
  entry={'providerImageId':r['id'],'position':r.get('computed_geometry'),'heading':r.get('compass_angle'),'cameraType':r.get('camera_type'),'sequence':r.get('sequence'),'captureDate':datetime.fromtimestamp(r['captured_at']/1000,tz=timezone.utc).date().isoformat() if r.get('captured_at') else None,'calibrationStatus':'not-yet-validated'}
  if i<download_limit and r.get('thumb_2048_url'):
   # Image hosts receive no API credential. Store only source metadata, never pagination tokens.
   with urllib.request.urlopen(r['thumb_2048_url'],timeout=45) as response:raw=response.read()
   sha=hashlib.sha256(raw).hexdigest();name=sha+'.jpg';(out/name).write_bytes(raw);entry.update(imageFile=name,imageSha256=sha)
  manifest.append(entry)
 (out/'imagery.json').write_text(json.dumps({'images':manifest,'downloaded':sum('imageFile' in r for r in manifest),'calibrationComplete':False},indent=2));return manifest

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bbox',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=2000);p.add_argument('--download-limit',type=int,default=0);a=p.parse_args();collect([float(x) for x in a.bbox.split(',')],a.output,a.limit,a.download_limit)
