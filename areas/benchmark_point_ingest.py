"""Measure complete source-cloud ingestion without building or rendering a scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from pyproj import Transformer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.project_point_cloud import project_point_cloud

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--area',required=True,type=Path)
p.add_argument('--target-crs',required=True)
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
center=json.loads((a.area/'config.json').read_text())['region']['center']
origin=Transformer.from_crs(4326,a.target_crs,always_xy=True).transform(center['lon'],center['lat'])
start=time.monotonic()
points=project_point_cloud(a.area/'data/points.npz',Transformer.from_crs(3857,a.target_crs,always_xy=True),origin)
h=hashlib.sha256();view=memoryview(points).cast('B')
for i in range(0,len(view),1024*1024):h.update(view[i:i+1024*1024])
report={'points':len(points),'columns':points.shape[1],'dtype':str(points.dtype),'arrayMiB':round(points.nbytes/1024**2,2),
        'seconds':round(time.monotonic()-start,3),'sha256':h.hexdigest(),'scope':'ingestion only; not a full scene build'}
a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
