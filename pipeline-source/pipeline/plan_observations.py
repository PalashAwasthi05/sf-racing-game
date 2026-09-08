"""Prioritize missing wall evidence by expected visible error, not by address importance."""
import json,math,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];data=json.load(open(ROOT.parent/'outputs/king-street-evidence.json'));world=data['world'];jobs=[]
for wall in data['facadeWalls']:
 b=next(b for b in world['buildings'] if b['id']==wall['buildingId']);area=wall['lengthM']*min(b['height'],25);uncertainty=.45 if wall['status']=='visual-fit' else 1.
 x,y=wall['midpoint'];distance=math.hypot(x,y);importance=area/(1+distance/120);normal=wall['normal'];stable=hashlib.sha256((wall['buildingId']+str(wall['edge'])+str(wall['midpoint'])).encode()).hexdigest()[:16]
 jobs.append({'id':stable,'buildingId':wall['buildingId'],'wallEdge':wall['edge'],'priority':round(importance*uncertainty,2),'requestedCapturePositionLocal':[round(x+normal[0]*15,2),round(y+normal[1]*15,2)],'lookAtLocal':[round(x,2),round(y,2),6],'minimumViews':2,'fields':['capture date','wall corners','floor bands','opening rectangles','mullion topology','material','frame depth','canopies','sign text and position'],'existingSources':wall['sourceIds'],'status':'needs-independent-view' if wall['status']=='visual-fit' else 'needs-observation'})
jobs.sort(key=lambda x:x['priority'],reverse=True)
result={'schemaVersion':1,'strategy':'One capture can cover several jobs. Deduplicate by frustum, date and visibility before spending vision tokens. Persist compact descriptors and never re-read unchanged images.','jobs':jobs,'counts':{'total':len(jobs),'withoutDirectEvidence':sum(not x['existingSources'] for x in jobs)},'qualityGate':'FAIL: no independent held-out facade validation yet'}
(ROOT.parent/'outputs/king-street-observation-queue.json').write_text(json.dumps(result,indent=2));print(result['counts'])
