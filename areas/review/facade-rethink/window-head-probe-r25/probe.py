import base64,json,sys,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from regen.jev import ask
out=Path('areas/review/facade-rethink/window-head-probe-r25')
cases=json.loads((out/'cases.json').read_text());assets={};photos={};questions={}
criteria={'flat':'Straight top edge of the wall aperture, even if a curved or triangular decorative hood sits above it.', 'segmental':'A shallow smoothly curved head of the wall aperture; rise clearly less than half the width.', 'round':'A rounded approximately semicircular head of the wall aperture, with rise about half its width.', 'pointed':'Two sides meet in a pointed arch at the top of the aperture.', 'unclear':'The aperture head is hidden, too blurred, absent, or another shape, so none of the four shapes can be established.'}
for i,c in enumerate(cases):
 key=str(uuid.uuid4());assets[key]={'type':'image','format':'png','data_base64':base64.b64encode(Path(c['image']).read_bytes()).decode()};photos[str(i)]=key
 questions[str(i)]={'type':'choice','instructions':{'task':f'Classify the head of the central wall opening in `photos.{i}`. It contains a window or entrance. Judge the aperture boundary where the solid wall stops, including the top of a glazed transom.','exclude':'Do not use the shape of a decorative hood, pediment or molding outside the aperture. Do not mistake a horizontal meeting rail for the top. Perspective can tilt a straight head. Do not invent a visible head when foliage obscures it.'},'criteria':criteria}
start=time.monotonic();ans=ask({'photos':photos},questions,tag='window-head-r25-benchmark',retries=1,assets=assets)
rows=[{**c,'answer':ans[str(i)],'correct':ans[str(i)]['choice']==c['expected']} for i,c in enumerate(cases)]
result={'experiment':'bounded classifier; never imported into generation','elapsedSeconds':time.monotonic()-start,'correct':sum(x['correct'] for x in rows),'total':len(rows),'cases':rows}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cases'}));print(json.dumps([{k:r[k] for k in ('id','expected','answer','correct')} for r in rows],indent=2))
