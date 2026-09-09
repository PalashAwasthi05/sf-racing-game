"""Fit all input features identically; cache observations outside the heuristic."""
import argparse,json
from pathlib import Path
import numpy as np
from shapely.geometry import shape
from shapely import contains_xy
from general_world.roof_planes import fit_roof

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
path=Path(a.input);data=json.loads(path.read_text());cloud=np.load(path.parent/data['surfacePointsFile'])['points'];records=[]
for item in data['buildings']:
    g=shape(item['geometry']);lo=g.bounds[:2];hi=g.bounds[2:]
    points=cloud[((cloud[:,:2]>=lo)&(cloud[:,:2]<=hi)).all(axis=1)]
    model,report=fit_roof(g,points)
    records.append({'key':item['key'],'model':model,**report})
    print(json.dumps({'attempted':len(records),'accepted':sum(r['accepted'] for r in records),'sloped':sum(r.get('sloped',False) for r in records)}),flush=True)
Path(a.output).write_text(json.dumps({'records':records,'locationRules':0,'source':'normalized point cloud; no image validation'},indent=2))
