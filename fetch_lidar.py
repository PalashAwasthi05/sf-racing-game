import json,urllib.request,pathlib,concurrent.futures,math
import numpy as np,laspy
from pyproj import Transformer
ROOT=pathlib.Path(__file__).parent/'data';CACHE=ROOT/'lidar';CACHE.mkdir(exist_ok=True)
BASE='https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SanFrancisco_1_B23/'
ept=json.loads((ROOT/'ept.json').read_text());bounds=ept['bounds']
proj=Transformer.from_crs(4326,3857,always_xy=True)
cx,cy=proj.transform(-122.3912719,37.7786154)
cos=math.cos(math.radians(37.7786154)); r=290/cos
clip=[cx-r,cy-r,cx+r,cy+r];maxdepth=8
def overlaps(k):
 d,x,y,z=map(int,k.split('-'));s=(bounds[3]-bounds[0])/(2**d)
 lo=[bounds[0]+x*s,bounds[1]+y*s,bounds[2]+z*s]
 return lo[0]<=clip[2] and lo[0]+s>=clip[0] and lo[1]<=clip[3] and lo[1]+s>=clip[1] and lo[2]<200 and lo[2]+s>-100
def download(rel):
 f=CACHE/rel.replace('/','_')
 if not f.exists():
  req=urllib.request.Request(BASE+rel,headers={'User-Agent':'KingStreetResearchPilot/0.1'})
  raw=urllib.request.urlopen(req,timeout=45).read();f.write_bytes(raw)
 return f
pending=['0-0-0-0'];visited=set();keys=[]
while pending:
 k=pending.pop()
 if k in visited:continue
 visited.add(k)
 data=json.loads(download('ept-hierarchy/'+k+'.json').read_text())
 for node,count in data.items():
  if int(node.split('-')[0])>maxdepth or not overlaps(node):continue
  if count==-1:pending.append(node)
  elif count>0:keys.append(node)
print('Selected nodes',len(set(keys)),flush=True)
def process(key):
 f=download('ept-data/'+key+'.laz');las=laspy.read(f)
 x,y,z=np.asarray(las.x),np.asarray(las.y),np.asarray(las.z)
 mask=(x>=clip[0])&(x<=clip[2])&(y>=clip[1])&(y<=clip[3])&(z>-100)&(z<200)
 return np.stack([x[mask],y[mask],z[mask],np.asarray(las.classification)[mask],np.asarray(las.intensity)[mask]],axis=1)
out=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for i,arr in enumerate(pool.map(process,sorted(set(keys)))):
  out.append(arr)
  if i%10==0: print('Processed nodes',i+1,'points',sum(len(a) for a in out),flush=True)
pts=np.concatenate(out)
np.savez_compressed(ROOT/'points.npz',points=pts)
print('Complete',pts.shape,'classes',np.unique(pts[:,3],return_counts=True),'z',np.quantile(pts[:,2],[0,.1,.5,.9,1]),flush=True)
