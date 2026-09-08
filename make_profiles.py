import json
from pathlib import Path
root=Path('work/pipeline');obs=json.load(open(root/'observations.json'))
obs['sources']['king-2025-c']={'url':'https://www.google.com/maps/@37.7785447,-122.3915291,3a,110y,285h,120t/data=!3m4!1e1!3m2!1sZEaDxnfTqji-aGZgIx9ffg!2e0','capturedAt':'2025-01','provider':'Google Street View','use':'Visual landmark annotation at 1280 x 720; pixels not embedded'}
obs['sources']['king-2025-d']={'url':'https://www.google.com/maps/@37.7786899,-122.3913452,3a,110y,320h,135t/data=!3m4!1e1!3m2!1sS4hs-Kyh3P14aRv9CWeTvg!2e0','capturedAt':'2025-01','provider':'Google Street View','use':'Visual grammar annotation; pixels not embedded; palm occlusion remains'}
obs['sources']['loft-architecture-report']={'url':'https://www.sfgate.com/entertainment/article/Belles-of-the-ballpark-neighborhood-two-2549100.php','capturedAt':'2006-10-19','provider':'Original architecture reporting','use':'Historical projection dimension (10 inches) and terrace/crown description; current silhouette cross-checked visually'}
obs['sources']['hyatt-dusk']={'url':'https://www.hyatt.com/hyatt-place/en-US/sfozs-hyatt-place-san-francisco-downtown/gallery','imageUrl':'https://assets.hyatt.com/content/dam/hyatt/hyattdam/images/2019/03/25/1010/Hyatt-Place-San-Francisco-Downtown-P012-Exterior-Dusk.jpg/Hyatt-Place-San-Francisco-Downtown-P012-Exterior-Dusk.4x3.jpg','capturedAt':None,'provider':'Hyatt operator photography','use':'Sparse image landmarks; pixels not embedded; date unknown, historical fit'}
obs['sources']['via-engineer']={'url':'https://imegcorp.com/project/new-12-story-hotel/','capturedAt':None,'provider':'Project structural engineer','use':'Historical exterior and completed entry photographs; no pixels embedded; current upper layout needs independent view'}
json.dump(obs,open(root/'observations.json','w'),indent=2)
profiles=[]
def group(region,cols,rows=1,**style):return {'region':region,'columns':cols,'rows':rows,'style':style}
profiles.append({'id':'deep-framed-loft','match':{'osmWayId':'149167949'},'normal':[.707,-.707],'sourceIds':['king-2025-c','king-2025-b','loft-architecture-report'],'material':'brick','panels':[{'rect':[0,.845,1,1],'material':'charcoal'}],'groups':[
 group([0,0,1,.13],5,kind='shopfront',openingRatio=.71,openingHeightRatio=.94,frameMaterial='silver',recessM=.6,canopy=True,mullionFractions=[.5],transomFractions=[.8]),
 group([0,.13,1,.30],5,openingRatio=.74,openingHeightRatio=.92,frameMaterial='silver',mullionFractions=[.25,.64],transomFractions=[.79]),
 {'fitId':'loft-projecting-frame-bank'},
 group([0,.87,1,1],10,openingRatio=.96,openingHeightRatio=.93,frameMaterial='silver',recessM=1.8,mullionFractions=[],transomFractions=[.28])],
 'bands':[{'height':.848,'thicknessM':.18,'depthM':.7,'material':'silver'},{'height':1,'thicknessM':.17,'depthM':.5,'material':'silver'}],
 'unresolved':['Facade region metric fractions and penthouse setback remain approximate','Garage and shopfront details only partially observed','Recess depth inferred; frame projection historically reported','Bay location errors from same-image fit are not ground-truth accuracy']})
profiles[0]['groups'][0]['omit']=[[3,0]];profiles[0]['openings']=[{'rect':[.63,.005,.77,.125],'style':{'kind':'rollup'}}]
# A second facade grammar encodes continuous glazing, rather than isolated repeated holes.
profiles.append({'id':'brick-and-projecting-window-strips','match':{'osmWayId':'149167937'},'normal':[.707,-.707],'sourceIds':['king-2025-a','king-2025-d'],'material':'brick','groups':[
 group([0,0,1,.13],5,kind='entry',openingRatio=.68,openingHeightRatio=.94,frameMaterial='silver',recessM=.4,mullionFractions=[.18,.82],transomFractions=[.76]),
 group([0,.13,1,.23],5,openingRatio=.71,openingHeightRatio=.86,frameMaterial='silver',mullionFractions=[.2,.8],transomFractions=[.85]),
 group([0,.265,1,.955],5,8,kind='window',openingRatio=.70,openingHeightRatio=.94,frameMaterial='silver',mullionFractions=[.24,.8],transomFractions=[.86]),
 group([0,.965,1,1],7,openingRatio=.84,openingHeightRatio=.92,frameMaterial='silver',mullionFractions=[],transomFractions=[])],
 'panels':[{'rect':[.225,.265,.375,.955],'material':'silver'},{'rect':[.625,.265,.775,.955],'material':'silver'}],
 'openings':[{'rect':[.45,.005,.55,.126],'style':{'kind':'entry','canopy':True,'canopyDepthM':2.1,'frameMaterial':'silver','mullionFractions':[.5]}}],
 'bands':[{'height':.125,'thicknessM':.65,'depthM':.3,'material':'stone'},{'height':.245,'thicknessM':.45,'depthM':.48,'material':'stone'}],
 'unresolved':['Palm-occluded window strips completed from adjacent visible repeats','Eight upper rows and five bays are visual grammar estimates, not numerical landmark fits','Figurative sculptural frieze not reconstructed; continuous base band only','No material scan tied to this exact building']})
# Fit the observed Third Street bay stack; other wall uses same family with explicit transfer status.
for normal,ident,fitgroup in [([-.707,-.707],'brick-hotel-third',{'fitId':'hotel-bay-bank'}),([-.707,.707],'brick-hotel-townsend',group([.16,.17,.86,.75],3,7,kind='projected_bay',material='zinc',frameMaterial='charcoal',openingRatio=.74,openingHeightRatio=.66,projectionM=.5,recessM=.13,mullionFractions=[.22,.78],transomFractions=[.7]))]:
 profiles.append({'id':ident,'match':{'osmWayId':'573287814'},'normal':normal,'sourceIds':['hyatt-dusk'],'material':'brick','groups':[
 group([0,0,1,.095],9,kind='shopfront',openingRatio=.74,openingHeightRatio=.93,frameMaterial='charcoal',mullionFractions=[.5],transomFractions=[.77]),fitgroup,
 group([.10,.125,.98,.79],4,9,openingRatio=.29,openingHeightRatio=.58,frameMaterial='charcoal',mullionFractions=[.4],transomFractions=[]),
 group([0,.805,1,1],14,3,openingRatio=.95,openingHeightRatio=.92,frameMaterial='silver',mullionFractions=[],transomFractions=[.32,.82]),
 group([0,.10,.095,.80],1,10,openingRatio=.97,openingHeightRatio=.96,frameMaterial='silver',mullionFractions=[.5],transomFractions=[.28,.8])],
 'panels':[{'rect':[0,.79,1,1],'material':'silver'},{'rect':[0,.10,.095,.80],'material':'silver'}],
 'bands':[{'height':.10,'thicknessM':.13,'material':'stone'},{'height':.79,'thicknessM':.17,'material':'stone'},{'height':.858,'thicknessM':.045,'depthM':.45,'material':'silver'},{'height':.925,'thicknessM':.045,'depthM':.45,'material':'silver'}],
 'unresolved':['Unknown capture date','Depth estimated from oblique imagery','Townsend pattern transferred from same photographed building; not independently fitted','Entrance portal and lower wing silhouette need further fitting']})
profiles.append({'id':'terracotta-hotel-with-shades','match':{'osmWayId':'148547438'},'normal':[.707,-.707],'sourceIds':['via-engineer'],'material':'terracotta','panels':[{'rect':[.26,.095,.59,.94],'material':'stucco'},{'rect':[.66,.095,.95,.94],'material':'stucco'}],'groups':[
 group([0,0,1,.09],5,kind='shopfront',openingRatio=.90,openingHeightRatio=.96,mullionFractions=[.5],transomFractions=[.75],recessM=.25),
 group([0,.11,.22,.94],1,10,openingRatio=.92,openingHeightRatio=.75,frameMaterial='charcoal',mullionFractions=[.67],transomFractions=[],shade=True),
 group([.27,.11,.58,.94],2,10,openingRatio=.78,openingHeightRatio=.63,frameMaterial='charcoal',mullionFractions=[],transomFractions=[],shade=True),
 group([.66,.11,.94,.94],2,10,openingRatio=.78,openingHeightRatio=.63,frameMaterial='charcoal',mullionFractions=[],transomFractions=[],shade=True),
 group([0,.955,1,1],6,openingRatio=.90,openingHeightRatio=.92,frameMaterial='charcoal',mullionFractions=[],transomFractions=[])],
 'bands':[{'height':.97,'thicknessM':.25,'depthM':1.7,'material':'charcoal','projectionM':.6}],
 'unresolved':['Historical construction photograph establishes upper facade only','Ten repeated rows are visual count estimate','Ground storefront pattern inferred from completed alley entry photograph; frontage details unresolved','Exact tile size and shade depth inferred']})
json.dump({'schemaVersion':1,'description':'Reusable grammar data derived from observations; manually annotated grammar parameters are distinct from numerical image fits. Do not transfer a profile to an unseen facade silently.','profiles':profiles},open(root/'facade-profiles.json','w'),indent=2)
