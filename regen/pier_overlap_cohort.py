import argparse,json
from pipeline.facade_grammar import openings,opening_outline
from pipeline.facade_lint import clean
from shapely.geometry import box
from pathlib import Path
ap=argparse.ArgumentParser(description='Find explicit facade piers overlapping observed openings, including registered window banks.')
ap.add_argument('profiles',type=Path)
ap.add_argument('output',type=Path)
args=ap.parse_args()
data=args.output
data.mkdir(parents=True,exist_ok=True)
rows=[]
for raw in json.loads(args.profiles.read_text())['profiles']:
 if not raw.get('piers') or not raw.get('frameU'):continue
 p,_=clean(raw)
 W=p['frameU'][1]-p['frameU'][0];H=p['heightRefM'];ops=openings(p,[],W,H)
 for q in p['piers']:
  x=q['x']*W;w=q.get('widthM',.5);a,b=q.get('y',[0,1]);s=box(x-w/2,a*H,x+w/2,b*H);h=[]
  for op in ops:
   try:ring=opening_outline(op,W,H)
   except ValueError:continue
   if ring.is_valid:
    inter=ring.intersection(s).area
    if inter>.001:h.append(round(inter/max(1e-8,ring.area),4))
  if h:rows.append(dict(profile=p['id'],feature=p['match']['featureId'],swatch=q.get('material',p['material']),count=len(h),maxOverlap=max(h),widthM=w))
ids=sorted({r['feature'] for r in rows})
(data/'affected.json').write_text(json.dumps(ids))
(data/'piers.json').write_text(json.dumps(rows))
print(len(ids),'features;',len(rows),'piers;',sum(x['count'] for x in rows),'intersections')
