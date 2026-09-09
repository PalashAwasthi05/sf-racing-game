"""One-time transcription of inspected source packet to metric facade observations.
This is observation authoring, not the reusable inference/compiler stage.
All depths, spans and heights below are explicitly image-estimated.
"""
import json,copy
from pathlib import Path
root=Path('work/pipeline');E=json.load(open(root/'envelope-observations.json'));L=json.load(open(root/'lettering-observations.json'));packet=json.load(open('work/facade-references/v7/broader-frontage-observations.json'))
E['profiles']=[p for p in E['profiles'] if not p.get('observationBatch')=='v7'];L['signs']=[p for p in L['signs'] if p.get('observationBatch')!='v7']
SE=[.707,-.707];NW=[-.707,.707];SW=[-.707,-.707];NE=[.707,.707]
sources={k:{a:b for a,b in v.items() if a not in ['photoPath']} for k,v in packet['sources'].items()}
sources['180-townsend-google-2025']={'url':'https://www.google.com/maps/@37.779073,-122.39217,3a,95y,290h,105t/data=!3m1!1e1','captureDate':'2025-05','captureDateBasis':'Visible Street View label; camera 177 Townsend','inspection':'Five arched bays on each of three storeys. Dark ground, brick upper two, pale belts/dentils, observed 180 and andreessen horowitz.'}
E['sources']=[x for x in E['sources'] if x['id'] not in sources]+[dict(id=k,**v) for k,v in sources.items()];L['sources'].update(sources)
family={'brick_modular':('brick','default'),'brick_dark':('brick','dark'),'stucco':('render','default'),'stucco_red':('render','red'),'render_salmon':('render','salmon'),'render_gray':('render','gray'),'render_white':('render','white'),'stone':('pale-trim','default'),'panel_silver':('metal-panel','default'),'panel_dark':('metal-panel','dark'),'timber_dark':('timber','dark'),'timber_tan':('timber','tan')}
def node(id,span,z,mat,depth=0,grids=None,**kw):
 x={'id':id,'span':span,'z':z,'material':mat,'depthM':depth,**kw}
 if grids:x['grids']=grids
 return x
def grid(cols,rows,rect=[0,0,1,1],wr=.75,hr=.76,**style):return {'columns':cols,'rows':rows,'rect':rect,'style':{'widthRatio':wr,'heightRatio':hr,'frameMaterial':'charcoal','mullions':[.33,.67],'transoms':[.25,.75],**style}}
def op(rect,**style):return {'rect':rect,'style':{'frameMaterial':'charcoal',**style}}
def cornice(at=1,mat='stone',projection=.28):return {'op':'cornice','at':at,'material':mat,'section':[[0,.18,projection],[.15,.12,projection+.12]]}
def profile(id,fid,name,normal,height,nodes,src,**kw):
 p={'id':id,'observationBatch':'v7','match':{'featureId':fid},'displayName':name,'normal':normal,'replaceDepthM':3.0,'levels':{'roof':{'priorM':height,'searchRadiusM':1.2,'depthRangeM':[-4,.3]}},'nodes':nodes,'sourceIds':src,'unresolved':['Manually transcribed from photographs; metric bounds and depths are estimates.','Capture age is tracked separately from source publication and OSM edit date.','Unimaged regions and independent validation remain unresolved.'],**kw}
 for n in nodes:
  if n['material'] in family:
   f,finish=family[n['material']];n['materialEvidence']={'family':f,'finish':finish,'sourceIds':src,'status':'observed family; synthesized appearance'}
 E['profiles'].append(p);return p
def text(p,id,words,nodeid,u,z,width,height=.5,mat='white',**kw):
 src=p['sourceIds'];dates=[sources[s].get('captureDate') for s in src if s in sources and sources[s].get('captureDate')]
 L['signs'].append({'id':id,'observationBatch':'v7','text':words,'match':p['match'],'normal':p['normal'],'profileId':p['id'],'centerU':u,'bottomM':z,'widthM':width,'heightM':height,'material':mat,'sourceIds':src,'captureDate':max(dates) if dates else None,'attachment':{'kind':'envelope-surface','nodeId':nodeid},'carrierEvidence':{'type':'opaque-surface','sourceIds':src,'status':'visually observed carrier; pose estimated'},**kw})
# Bounded rows are restricted to each material patch; sign belts have no apertures.
p=profile('warehouse-painted-belt','sf_201006_0001817','128 King · Garcia & Maggini',SE,18.9,[
 node('retail-piers',[0,1],[0,6.2],'brick_modular',grids=[grid(7,1,[.17,0,1,1],.76,.86,recessM=.35)]),
 node('painted-belt',[0,1],[6.2,7.5],'brick_modular'),
 node('upper-brick',[0,1],[7.5,'roof'],'brick_modular',grids=[grid(7,2,[.17,0,1,.94],.72,.72)],components=[cornice(mat='brick_modular')])],['128-king-google-2025','128-king-noehill-2004'])
p['unresolved'].append('Only seven visible upper bays rendered in bounded right region; cropped left end has no claimed opening count.')
text(p,'warehouse-painted-name','GARCIA & MAGGINI CO','painted-belt',.56,6.55,14,.55);text(p,'warehouse-dried-fruits','DRIED FRUITS','painted-belt',.20,6.6,6,.45);text(p,'warehouse-nuts','NUTS HONEY','painted-belt',.83,6.6,5,.45);text(p,'warehouse-number','128','painted-belt',.96,6.5,1.4,.65)
p=profile('articulated-loft-corner','sf_201006_0002210','200 Townsend',SE,22.4,[
 node('ground-gray',[0,1],[0,4.8],'panel_silver',openings=[op([.04,.05,.25,.79],glassMaterial='panel_dark'),op([.40,.03,.51,.83]),op([.64,.05,.96,.78])]),
 node('salmon-wall',[0,.78],[4.8,'roof'],'render_salmon',grids=[grid(1,3,[.40,.03,.62,.97],.82,.75)],components=[cornice(mat='panel_dark')]),
 node('left-projecting-bays',[.035,.245],[5.1,21.4],'panel_dark',.42,grids=[grid(1,3,wr=.9,hr=.81)]),
 node('right-projecting-bays',[.56,.755],[5.1,21.4],'panel_dark',.42,grids=[grid(1,3,wr=.9,hr=.81)]),
 node('silver-corner',[.78,1],[4.8,'roof'],'panel_silver',.05,grids=[grid(1,3,wr=.84,hr=.8)],components=[{'op':'canopy','projectionM':1.7,'thicknessM':.16,'material':'charcoal'}])],['200-townsend-google-2025'])
# Projecting patches cut holes in the salmon parent rather than layering over it.
par=p['nodes'][1];par['children']=[]
for id in ['left-projecting-bays','right-projecting-bays']:
 n=next(n for n in p['nodes'] if n['id']==id);a,b=n['span'];n['rect']=[a/.78,(5.1-4.8)/(22.4-4.8),b/.78,(21.4-4.8)/(22.4-4.8)];n['offsetM']=.42;n.pop('span');n.pop('z');par['children'].append(n)
p['nodes']=[n for n in p['nodes'] if n['id'] not in ['left-projecting-bays','right-projecting-bays']]
text(p,'townsend-bank','FIRST REPUBLIC BANK','ground-gray',.80,4.0,8,.44);text(p,'townsend-200','200','ground-gray',.53,1.6,.9,.4)
p=profile('five-bay-decorated-front','sf_201006_0002834','224 Townsend · INSEAD',SE,9.2,[
 node('five-bay-wall',[0,1],[0,8.35],'render_white',grids=[grid(5,1,[0,.49,1,.83],.85,.9),grid(5,1,[0,0,1,.47],.85,.9)],surfaceEvidence=[{'label':'solid','rect':[0,.84,1,1]}],components=[{'op':'pilasters','count':6,'widthM':.3,'projectionM':.3,'material':'verdigris'},cornice(mat='verdigris')]),
 node('center-crown',[.37,.63],[8.35,10.1],'render_white',shape=[[0,0],[1,0],[1,.6],[.5,1],[0,.6]])],['224-townsend-google-2025','224-townsend-pq-2020'])
text(p,'insead-front','INSEAD','five-bay-wall',.5,7.6,2.8,.45,mat='verdigris')
p=profile('brick-six-bay-industrial','sf_201006_0002604','139 Townsend',NW,27.2,[
 node('ground-front',[0,1],[0,5.4],'brick_dark',grids=[grid(6,1,wr=.80,hr=.82)],components=[cornice(mat='verdigris'),*[{'op':'canopy','span':[a,b],'at':.8,'projectionM':1.05,'material':'verdigris'} for a,b in [[.21,.37],[.39,.64],[.66,.82]]]]),
 node('upper-front',[0,1],[5.4,'roof'],'brick_dark',grids=[grid(6,4,wr=.63,hr=.66,transoms=[.28],mullions=[.5])],components=[cornice(mat='brick_dark')])],['139-townsend-sks','139-townsend-transwestern'])
p=profile('paired-industrial-fire-escapes','sf_201006_0002920','133 / 135 Townsend',NW,26.5,[
 node('five-ground-bays',[0,1],[0,5.3],'render_gray',grids=[grid(5,1,wr=.78,hr=.76)],components=[cornice()]),
 node('paired-upper',[0,1],[5.3,'roof'],'render_white',grids=[grid(10,4,wr=.60,hr=.69,x='unused',mullions=[.25,.5,.75],transoms=[.2,.4,.6,.8])],jointsM=[1.2,.6],components=[cornice(),{'op':'fireEscape','span':[.025,.175],'levelsM':[6.2,11.3,16.4,21.4],'material':'charcoal'},{'op':'fireEscape','span':[.825,.975],'levelsM':[6.2,11.3,16.4,21.4],'material':'charcoal'}])],['135-townsend-google-2025'])
text(p,'sap-sign','SAP','five-ground-bays',.3,4.45,2.4,.6,mat='glass_blue');text(p,'entry-133','1\n3\n3','five-ground-bays',.075,1.1,.3,1.3)
p=profile('industrial-small-pane-front','sf_201006_0002572','101 Townsend',NW,16.3,[
 node('dark-shop-bases',[0,.7],[0,4.9],'panel_dark',openings=[op([.08,.08,.40,.86]),op([.58,.05,.68,.83]),op([.73,.08,.92,.83])]),
 node('entry-gray',[.7,1],[0,4.9],'render_gray',openings=[op([.14,.02,.85,.68])],components=[{'op':'canopy','at':.7,'span':[.05,.95],'projectionM':1.1,'braces':True,'material':'charcoal'}]),
 node('two-industrial-rows',[0,1],[4.9,'roof'],'render_gray',grids=[grid(4,2,[.3,0,1,.98],.8,.75,mullions=[.14,.28,.42,.57,.71,.85],transoms=[.2,.4,.6,.8],awning=True)],components=[cornice(mat='render_gray')])],['101-townsend-google-2025','101-townsend-dahlin'])
text(p,'cloudflare-name','CLOUDFLARE','entry-gray',.85,4.02,5,.45,mat='charcoal');text(p,'address-101','101 TOWNSEND','entry-gray',.85,3.55,3,.25)
# The street-facing end has observed systems; long waterfront counts stay unresolved.
p=profile('stacked-glass-addition','sf_201006_0000115','China Basin · Berry building',NW,26.2,[
 node('lower-three-levels',[0,.77],[0,15.8],'render_white',grids=[grid(6,3,wr=.80,hr=.68)],components=[cornice()]),
 node('arched-end',[.77,1],[0,15.8],'render_white',grids=[grid(1,1,[.05,.03,.95,.98],.84,.88,shapeKind='arch',springRatio=.62,transoms=[.33,.66],mullions=[.33,.67])]),
 node('louver-belt',[0,1],[15.8,17.2],'panel_dark',components=[{'op':'fins','count':8,'projectionM':.16,'material':'charcoal'}]),
 node('upper-addition',[0,1],[17.2,'roof'],'panel_silver',-.55,grids=[grid(9,2,wr=.94,hr=.85,glassMaterial='glass_blue')],components=[cornice(),{'op':'fins','count':2,'projectionM':.35,'material':'charcoal'}])],['china-basin-mccarthy-1','china-basin-mccarthy-2'])
p['unresolved'].append('Long facade bay counts estimated from visible module pitch; not fully counted. Courtyard CHINA BASIN words excluded until retaining-wall attachment is established.')
# Keep unobserved repetition explicit rather than claim those full-wall counts were seen.
for n in p['nodes']:
 for g in n.get('grids',[]):g['layoutStatus']='pitch-estimated on incomplete long view'
p=profile('wharfside-grid-end','osm_46264108','China Basin · Wharfside',NW,26.4,[
 node('promenade-base',[0,1],[0,4.2],'render_gray',grids=[grid(4,1,wr=.75,hr=.82)]),
 node('white-upper-frame',[0,1],[4.2,'roof'],'render_white',grids=[grid(4,5,wr=.84,hr=.73,mullions=[.33,.67],transoms=[.76])],components=[cornice()])],['china-basin-mccarthy-2','china-wharfside-mhp'])
p['unresolved'].append('Near-end counts are estimated; long waterfront wall retains priors.')
p=profile('brick-entry-sign-panel','sf_201006_0006801','25 Lusk',NE,11.8,[
 node('ground-brick',[0,1],[0,5.0],'brick_modular',openings=[op([.40,.03,.66,.86])],components=[{'op':'canopy','span':[.38,.84],'at':.92,'projectionM':1.25,'material':'white'}]),
 node('opaque-entry-sign',[.67,.84],[.2,4.45],'render_white',.05),
 node('upper-two-observed',[0,1],[5,'roof'],'brick_modular',grids=[grid(2,1,[.4,.10,.85,.85],.76,.74,mullions=[.166,.33,.5,.66,.83],transoms=[.25,.5,.75])],components=[cornice(mat='brick_modular'),{'op':'dentils','at':.92,'material':'brick_modular'}])],['25-lusk-psd','45-lusk-google-2023'])
# Nested opaque panel cuts away the carrier wall.
base=p['nodes'][0];sign=p['nodes'][1];sign['rect']=[.67,.04,.84,.89];sign['offsetM']=.05;sign.pop('span');sign.pop('z');base['children']=[sign];p['nodes'].remove(sign)
text(p,'25-lusk-panel','25\nLUSK','opaque-entry-sign',.755,1.05,2.4,2.3,mat='charcoal')
p=profile('low-timber-raised-entry','sf_201006_0005190','45 / 47 Lusk',NE,8.5,[
 node('timber-front',[0,1],[0,'roof'],'timber_dark',openings=[op([.04,.21,.28,.79],mullions=[.25,.5,.75],transoms=[.2,.4,.6,.8]),op([.35,.14,.49,.68]),op([.55,.21,.76,.79],mullions=[.25,.5,.75],transoms=[.2,.4,.6,.8])],components=[cornice(mat='timber_dark'),{'op':'entrySteps','span':[.35,.49],'topM':.7,'runM':1.4,'count':4,'handrail':True,'material':'stone'}])],['45-lusk-google-2023'])
p['unresolved'].append('Photographed 45 region spans estimated part of aggregated footprint. 47 tenant words excluded pending parcel correspondence.')
text(p,'lusk-45-number','45','timber-front',.53,1.6,.55,.45)
p=profile('independent-low-shop-zones','sf_201006_0007874','Third Street · low shop row',SW,7.2,[
 node('corner-storefront',[0,.34],[0,3.6],'charcoal',grids=[grid(3,1,wr=.9,hr=.95)]),
 node('corner-high-fascia',[0,.34],[3.6,6.4],'timber_tan'),
 node('sports-bar-front',[.34,.67],[0,5.9],'render_white',grids=[grid(3,1,[0,.10,1,.61],.69,.85)],components=[cornice(mat='charcoal')]),
 node('northern-low-shops',[.67,1],[0,'roof'],'render_white',openings=[op([.10,.05,.40,.59]),op([.56,.05,.91,.59])])],['third-king-google-2025','761-third-chronicle-2026','761-third-kron-2026'])
text(p,'sports-bar-name','CANDLESTICK PARK','sports-bar-front',.5,4.45,9,.40);text(p,'sports-bar-subtitle','SPORTS BAR','sports-bar-front',.5,4.03,4,.25)
L.setdefault('excluded',[]).append('BASEBALLISM March2025 observed sign superseded by January/February2026 tenant-change evidence; CardVault sign pixels not verified. No replacement text invented.')
p=profile('arched-three-storey-brick','sf_201006_0003782','180 Townsend',SE,16.1,[
 node('arched-ground',[0,1],[0,5.1],'panel_dark',grids=[grid(5,1,wr=.76,hr=.91,shapeKind='arch',springRatio=.65,mullions=[.33,.67],transoms=[.64])]),
 node('name-belt',[0,1],[5.1,6.2],'stone'),
 node('middle-arches',[0,1],[6.2,11.1],'brick_modular',grids=[grid(5,1,wr=.60,hr=.77,shapeKind='arch',springRatio=.74)],components=[cornice(),{'op':'dentils','at':1,'material':'stone'}]),
 node('top-arches',[0,1],[11.1,'roof'],'brick_modular',grids=[grid(5,1,wr=.50,hr=.71,shapeKind='arch',springRatio=.67)],components=[cornice()])],['180-townsend-google-2025'])
text(p,'a16z-physical-name','andreessen\nhorowitz','name-belt',.5,5.2,5,.9,mat='charcoal');text(p,'a16z-number','180','arched-ground',.5,3.1,.9,.45)
(root/'envelope-observations.json').write_text(json.dumps(E,indent=2));(root/'lettering-observations.json').write_text(json.dumps(L,indent=2))
# Portable facts retained with source dates; source pixels omitted.
packet['sources']=sources;(root/'broader-observations.json').write_text(json.dumps(packet,indent=2))
print('Added',sum(p.get('observationBatch')=='v7' for p in E['profiles']),'frontages;',len(L['signs']),'total sign placements')
