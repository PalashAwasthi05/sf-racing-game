"""Optional provider adapter. Requires an authorized MAPILLARY_ACCESS_TOKEN in the environment.
Retrieves metadata only; captures are selected against the wall queue before image transfer.
"""
import os,json,urllib.request,urllib.parse,pathlib,argparse
p=argparse.ArgumentParser();p.add_argument('--bbox',required=True,help='west,south,east,north');p.add_argument('--output',required=True);a=p.parse_args()
token=os.environ.get('MAPILLARY_ACCESS_TOKEN')
if not token:raise SystemExit('Mapillary access is not configured. Set MAPILLARY_ACCESS_TOKEN locally, then rerun. No image downloads were attempted.')
params={'bbox':a.bbox,'fields':'id,captured_at,computed_geometry,compass_angle,camera_type,sequence,thumb_2048_url','limit':500}
url='https://graph.mapillary.com/images?'+urllib.parse.urlencode(params);items=[]
while url:
 req=urllib.request.Request(url,headers={'Authorization':'OAuth '+token})
 with urllib.request.urlopen(req,timeout=60) as r:result=json.load(r)
 items.extend(result.get('data',[]));url=result.get('paging',{}).get('next')
pathlib.Path(a.output).write_text(json.dumps({'provider':'Mapillary','images':items},indent=2));print('Saved image metadata:',len(items))
