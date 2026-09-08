from pathlib import Path
import json,copy
P=Path('work/pipeline')
def save(name,d):(P/name).write_text(json.dumps(d,indent=2))
def blank(rect,source,label='solid',note='Observed uninterrupted wall field'):
 return {'rect':rect,'label':label,'sourceIds':[source],'confidence':'visual','boundaryStatus':'image-estimated','note':note}
d=json.load(open(P/'facade-profiles.json'));loft=d['profiles'][0];loft['material']='brick_modular';loft['sourceIds'].append('188-broker')
loft['groups'][0]['region'][2]=.82;loft['groups'][0]['omit']=[[3,0],[4,0]]
loft['groups'][1]['region'][2]=.82
loft['groups'][2]['region']=[0,.31,.82,.84]
loft['groups'][3]['region']=[0,.87,.82,1]
loft['panels'][0]['rect'][2]=.82
loft['openings']=[{'rect':[.507,.006,.641,.12],'style':{'kind':'rollup'},'sourceIds':['king-2025-c']},{'rect':[.676,.005,.797,.128],'style':{'kind':'entry','recessM':.65,'canopy':True,'frameMaterial':'silver','mullionFractions':[.5],'transomFractions':[.82]},'sourceIds':['king-2025-c']}]
loft['surfaceEvidence']=[blank([.817,.13,.829,.84],'king-2025-c',note='Brick boundary between broad loft bank and independent small-window stack'),blank([.821,.84,.922,1],'188-broker',note='Upper strip beyond photo coverage; classification is unresolved')]
# Unknown is not negative evidence: never infer solid solely from cropped source.
loft['surfaceEvidence'][-1]['label']='unknown'
loft['groups'].append({'region':[.928,.13,1,.93],'columns':1,'rows':8,'style':{'openingRatio':.95,'openingHeightRatio':.86,'frameMaterial':'silver','mullionFractions':[.48],'transomFractions':[.77]}})
loft['unresolved']=['Main-bank facade width and absolute region heights are image estimates; not survey dimensions.','Seven small-window boundaries and the narrow entry are rectified from an undated broker photograph, checked for topology against Jan2025 street view.','Penthouse and far-right glazing repeats remain estimates.','Ground retail window films/artwork remain unresolved; historical TOTO campaign text excluded.','No independent street-photo accuracy evaluation has passed.']
# Direct edge annotations on the separately identified, coplanar narrow brick stack.
packet={'id':'loft-small-stack-boundaries','profileId':loft['id'],'sourceIds':['188-broker','king-2025-c'],'imageSize':[1280,853],'quadPx':[[752,0],[834,0],[863,728],[774,728]],'region':[.825,0,.92,.89],'planeStatus':'manually identified; top of photograph cropped; metric extent estimated','openings':[]}
for i,(xl,yt,xr,yb) in enumerate([(759,80,819,119),(760,147,821,185),(764,218,823,253),(767,287,826,326),(772,357,831,397),(775,432,834,475),(778,508,839,557),(783,600,847,713)]):
 packet['openings'].append({'id':'small-stack-'+str(i),'cornersPx':[[xl,yt],[xr,yt],[xr,yb],[xl,yb]],'style':{'kind':'entry' if i==7 else 'window','frameMaterial':'silver','recessM':.18,'mullionFractions':[.74],'transomFractions':[.79] if i==7 else []}})
save('observations/opening-boundaries.json',{'schemaVersion':1,'annotationMethod':'Manual edge picks; source pixels excluded from deliverable','packets':[packet]})
for p in d['profiles'][1:]:
 if p['material']=='brick':p['material']='brick_modular'
 if p['material']=='terracotta':p['material']='terracotta_panel'
 for g in p.get('groups',[]):
  if g.get('style',{}).get('material')=='zinc':g['style']['material']='panel_dark'
 if p['id']=='brick-and-projecting-window-strips':
  p['surfaceEvidence']=[blank([0,.235,1,.256],'king-2025-a','solid','Opaque sculptural belt; no window row')]
 if p['id'].startswith('brick-hotel'):
  # Separate panel appearance from glazing; stop slits at projecting-bay banks.
  p['material']='brick_dark'
  p['surfaceEvidence']=[blank([.096,.105,.985,.121],'hyatt-dusk',note='Opaque floor spandrel above shopfronts')]
save('facade-profiles.json',d)
e=json.load(open(P/'envelope-observations.json'))
for p in e['profiles']:
 def visit(n):
  if n.get('material')=='brick':n['material']='brick_modular'
  if n.get('material')=='terracotta':n['material']='terracotta_ribbed'
  if p['id'].startswith('stepped-corner') and n.get('material')=='limestone':n['material']='stucco'
  if p['id']=='stepped-corner-king' and n['id']=='shops':n['material']='stucco_red'
  if 'clock' in n['id'] and n.get('grids'):
   # The old giant window covered the tower's brick banner/clock field.
   n['surfaceEvidence']=[blank([.12,.36,.88,.94],'oracle-diversified',note='Brick clock/sign/banner field above the actual low opening')]
   n['openings']=[{'rect':[.31,.03,.69,.34],'style':{'widthRatio':1,'mullions':[.25,.5,.75],'transoms':[.24,.49,.74],'glassMaterial':'glass_dark','frameMaterial':'verdigris'},'sourceIds':['oracle-diversified']}]
  if p['id']=='mixed-height-loft-king' and n['id']=='corner-end-block':
   n.setdefault('children',[]).append({'id':'opaque-recessed-infill','rect':[.55,.04,.96,.38],'offsetM':-.48,'material':'panel_silver','surfaceEvidence':[blank([0,0,1,1],'glassworks-perry','opaque-panel','Opaque jointed metal infill below projecting loft bay')]})
  for c in n.get('children',[]):visit(c)
 for n in p['nodes']:visit(n)
 # Completion remains bounded by separate observed surfaces, not blanket facade coverage.
 p['unresolved']=[x for x in p.get('unresolved',[]) if 'lettering' not in x.lower()]+['Opening counts outside direct boundary annotations remain visual estimates; no independent validation.']
save('envelope-observations.json',e)
# Literal observed signage, attached through the same reusable facade frames.
S=[]
def sign(id,text,match,normal,u,z,w,h,source,**kw):
 S.append(dict(id=id,text=text,match=match,normal=normal,centerU=u,bottomM=z,widthM=w,heightM=h,sourceIds=source,**kw))
sign('loft-address','188',{'osmWayId':'149167949'},[.707,-.707],.736,4.7,2.0,.8,['king-2025-c','188-broker'],material='silver',font='Arial Bold.ttf',depthM=.45,captureDate='2025-01')
sign('glassworks-king-flagship','FLAGSHIP',{'osmWayId':'679290110'},[-.707,.707],.035,3.85,4.4,.62,['third-king-2025'],profileId='mixed-height-loft-king',depthM=.36,captureDate='2025-01',font='Arial.ttf')
sign('glassworks-third-flagship','FLAGSHIP',{'osmWayId':'679290110'},[.707,.707],.945,3.85,4.4,.62,['third-king-2025'],profileId='double-height-loft-third',depthM=.05,captureDate='2025-01',font='Arial.ttf')
sign('glassworks-bank','WELLS\nFARGO',{'osmWayId':'679290110'},[-.707,.707],.10,2.25,1.4,.9,['third-king-2025'],profileId='mixed-height-loft-king',depthM=-.06,captureDate='2025-01',font='Georgia Bold.ttf')
sign('hyatt-entry','HYATT PLACE',{'osmWayId':'573287814'},[-.707,-.707],.86,3.8,3.8,.43,['hyatt-dusk'],depthM=.5,captureDate=None,font='Arial.ttf',canopy={'projectionM':1.4})
sign('via-alley-entry','HOTEL VIA',{'osmWayId':'148547438'},[-.707,-.707],.85,4.4,5.,.85,['via-entry'],depthM=1.25,captureDate=None,font='Arial.ttf',canopy={'projectionM':1.4})
for pid,u in [('masonry-and-open-structure',.9725),('clock-tower-side',.5)]:
 sign('oracle-tower-'+pid,'ORACLE PARK',{'featureId':'sf_201006_0000003'},[-.707,-.707] if pid=='clock-tower-side' else [-.707,.707],u,28.7,7.3,1.4,['oracle-diversified','third-king-2025'],profileId=pid,depthM=.16,material='#c79e54',archM=.22,captureDate=None)
sign('oracle-number','24',{'featureId':'sf_201006_0000003'},[-.707,-.707],.5,22.1,1.35,1.05,['oracle-diversified'],profileId='clock-tower-side',depthM=.14,material='charcoal',captureDate=None,font='Arial Bold.ttf')
save('lettering-observations.json',{'schemaVersion':1,'method':'Literal photographed text, manual attachment estimate; reusable contour extrusion','signs':S,'excluded':['Historical TOTO retail campaign; Jan2025 artwork differs','Historical MISSION BAY MARKETING CENTER','BEACON brochure page typography','Unreadable current 170 frontage text'],'sources':json.load(open('work/facade-references/v6/signs-materials-blank-zones.json'))['sources']})
p=P/'config.json';c=json.load(open(p));c['version']='0.6';save('config.json',c)
