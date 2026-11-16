"""Photo capture, edit time and known change events are separate evidence clocks."""
from datetime import date

def evaluate(source,as_of,max_months=24):
 raw=source.get('captureDate',source.get('capturedAt'));today=date.fromisoformat(as_of)
 if not raw:return {'status':'capture-date-unknown','captureDate':None,'ageMonths':None}
 try:d=date.fromisoformat(raw[:10] if len(raw)>=10 else raw+'-01')
 except (ValueError,TypeError):return {'status':'capture-date-invalid','captureDate':raw,'ageMonths':None}
 age=(today.year-d.year)*12+today.month-d.month
 return {'status':'future-date-review' if age<0 else 'stale-review' if age>max_months else 'within-age-budget','captureDate':raw,'ageMonths':age}

def sign_allowed(sign):
 # A confirmed change event invalidates a former tenant sign. A new directory
 # label never becomes an observed sign without a literal visual observation.
 return not sign.get('supersededBy') and sign.get('textEvidence','image')=='image'

def ledger(profiles,sources,as_of,max_months=24):
 result=[]
 for p in profiles:
  evidence=[dict(sourceId=i,**evaluate(sources.get(i,{}),as_of,max_months)) for i in p.get('sourceIds',[])]
  result.append({'profileId':p['id'],'match':p['match'],'evidence':evidence,'hasDatedStreetEvidence':any(e['captureDate'] for e in evidence),'needsFreshCapture':not any(e['status']=='within-age-budget' for e in evidence),'currentAsOfValidated':False})
 return result
