"""Compile wall observations and semantic priors into reusable architectural components."""
import math
import numpy as np
from shapely.geometry import Point,LineString
from . import facade_grammar

DEFAULTS={
 'masonry':dict(material='brick_modular',archetype='masonry',floorHeightM=3.65,groundHeightM=4.3,bayWidthM=3.4,openingRatio=.56,openingHeightRatio=.66,frameProjectionM=.03,frameWidthM=.065,mullions=1,transomRatio=.66,ground='shopfront'),
 'residential':dict(material='stucco',archetype='residential',floorHeightM=3.15,groundHeightM=4.0,bayWidthM=3.4,openingRatio=.62,openingHeightRatio=.7,frameProjectionM=.05,frameWidthM=.07,mullions=1,transomRatio=.72,ground='entry'),
 'commercial':dict(material='stucco',archetype='commercial',floorHeightM=3.8,groundHeightM=4.5,bayWidthM=3.0,openingRatio=.81,openingHeightRatio=.76,frameProjectionM=.035,frameWidthM=.065,mullions=2,transomRatio=.73,ground='shopfront'),
 'venue':dict(material='brick_modular',archetype='venue',floorHeightM=4.8,groundHeightM=4.5,bayWidthM=6.,openingRatio=.65,openingHeightRatio=.66,frameProjectionM=.04,frameWidthM=.1,mullions=3,transomRatio=.72,ground='entry')}

def prior(tags,height,area):
 if tags.get('leisure')=='stadium' or area>18000:return DEFAULTS['venue'].copy()
 if tags.get('building') in ['apartments','residential','hotel']:return DEFAULTS['residential'].copy()
 if tags.get('building') in ['commercial','office','retail'] or height>23:return DEFAULTS['commercial'].copy()
 return DEFAULTS['masonry'].copy()

def compile(ctx,c,z0,main_height,height,tags,observations):
 default_height=main_height;g=c['g'].simplify(.18,preserve_topology=True);fid=c['id'];cube=ctx['cube'];cylinder=ctx['cylinder'];records=[];components=0
 profiles=ctx.get('PROFILES',[]);fits=ctx.get('FITS',[])
 matches=[o for o in observations if str(o['match'].get('osmWayId'))==str(c['osm']['id'] if c['osm'] else '')]
 for poly in ctx['polys'](g):
  for ring in [poly.exterior]+list(poly.interiors):
   coords=list(ring.coords);segments=[];cut=ctx.get('envelopeMask')
   for aa,bb in zip(coords,coords[1:]):
    line=LineString([aa,bb]);remaining=line.difference(cut.buffer(.015)) if cut is not None and not cut.is_empty else line
    for part in ctx['lines'](remaining):segments.extend(zip(part.coords,list(part.coords)[1:]))
   for edge,(aa,bb) in enumerate(segments):
    a,b=np.asarray(aa),np.asarray(bb);u=b-a;length=np.linalg.norm(u)
    if length<.3:continue
    u=u/length;n=np.array([u[1],-u[0]]);mid=(a+b)/2
    if g.contains(Point(*(mid+n*.2))):n=-n
    main_height=default_height
    mask=ctx.get('envelopeMask')
    if mask is not None and not mask.is_empty and mask.buffer(.12).contains(Point(*mid)):continue
    if c.get('elevationOffset',0)>0 and c.get('outerBoundary') is not None and c['outerBoundary'].distance(Point(*mid))<.18:continue
    samples=ctx.get('roofSamples')
    if samples is not None and len(samples)>30 and not c.get('roofBand'):
     delta=samples[:,:2]-a;along=delta@u;away=np.abs(delta@n);near=samples[(along>=0)&(along<=length)&(away<2.0)]
     if len(near)>20:main_height=max(default_height,min(height,float(np.quantile(near[:,2],.96)-z0)))
    angle=math.atan2(u[1],u[0]);style=prior(tags,height,c['full_area'])
    profile=next((p for p in profiles if c['osm'] and str(p['match'].get('osmWayId'))==str(c['osm']['id']) and np.dot(n,p['normal'])>.95),None)
    if profile and c.get('elevationOffset',0)<.5:
     profile_height=main_height
     if samples is not None and len(samples)>30:
      delta=samples[:,:2]-a;along=delta@u;away=np.abs(delta@n);near=samples[(along>=-1)&(along<=length+1)&(away<3.5)&(samples[:,2]>z0+height*.65)]
      if len(near)>15:
       bins=np.arange(z0+height*.65,z0+height+1,.5);hist,edges=np.histogram(near[:,2],bins);peak=int(np.argmax(hist));cluster=near[(near[:,2]>=edges[peak]-.5)&(near[:,2]<edges[peak]+1)]
       profile_height=min(height,float(np.quantile(cluster[:,2],.85)-z0+.12)) if len(cluster) else height
      else:profile_height=height
     fitted=facade_grammar.compile_wall(ctx,c,a,b,z0,profile_height,profile,fits)
     if fitted:
      records.append(fitted['record']);components+=fitted['components'];continue
    obs=None if profile else next((o for o in matches if np.dot(n,o['normal'])>.82),None)
    if obs:style.update(obs)
    visible=length>=3.5 and not ctx['all_buildings'].contains(Point(*(mid+n*.8))) and ctx['boundary'].boundary.distance(Point(*mid))>.5
    mat=style['material'];layer='buildings__'+fid;detail='facades__'+fid
    def block(s,z,w,h,d,material,depth=0,key=detail):
     if w<=.001 or h<=.001:return
     pos=a+u*s+n*depth;cube([pos[0],pos[1],z0+z],[w,d,h],key,material,angle)
    if c.get('roofBand'):
     # Roof height bands are opaque envelope risers, never invented window floors.
     block(length/2,main_height/2,length,main_height,.24,mat,-.12,layer);continue
    if not visible:
     block(length/2,main_height/2,length,main_height,.38,mat,-.19,layer);continue
    cols=max(1,round(length/style['bayWidthM']));bay=length/cols;ground_level=c.get('elevationOffset',0)<.5;gh=min(style['groundHeightM'],main_height*.42) if ground_level else 0
    rows=max(1,round((main_height-gh)/style['floorHeightM']));fh=(main_height-gh)/rows
    wid=bay*style['openingRatio'];opening_h=fh*style['openingHeightRatio'];fw=style['frameWidthM'];projection=style['frameProjectionM'];recess=.18
    records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':rows+1,'columns':cols,'archetype':style['archetype'],'material':mat,'sourceIds':obs.get('sourceIds',[]) if obs else [],'status':'visual-fit' if obs else 'inferred','unresolved':obs.get('unresolved',[]) if obs else ['complete facade has no direct image fit; material and openings remain unverified priors']})
    for row in range(0 if ground_level else 1,rows+1):
     low=0 if row==0 else gh+(row-1)*fh;rh=gh if row==0 else fh
     ow=bay*(.74 if row==0 else style['openingRatio']);oh=rh*(.77 if row==0 else style['openingHeightRatio']);bottom=low+(.15 if row==0 else (rh-oh)*.48);top=bottom+oh
     for col in range(cols):
      s=(col+.5)*bay;left=s-ow/2;right=s+ow/2;fullprojection=projection if row>1 or not ground_level else .035
      # Real openings: four wall pieces leave a recess. No solid box behind the glazing.
      block((col*bay+left)/2,low+rh/2,left-col*bay,rh,.38,mat,-.19,layer)
      block((right+(col+1)*bay)/2,low+rh/2,(col+1)*bay-right,rh,.38,mat,-.19,layer)
      block(s,(low+bottom)/2,ow,bottom-low,.38,mat,-.19,layer)
      block(s,(top+low+rh)/2,ow,low+rh-top,.38,mat,-.19,layer)
      variant=(col+row*3)%7;glass=['glass','glass_dark','glass_light'][variant%3]
      block(s,(bottom+top)/2,ow,oh,.04,glass,-recess)
      # Four metal jambs, projecting frame, sill and operable-panel divisions.
      depth=.18+fullprojection
      for x in [left+fw/2,right-fw/2]:block(x,(bottom+top)/2,fw,oh,depth,'aluminum',fullprojection/2)
      for zz in [bottom+fw/2,top-fw/2]:block(s,zz,ow,fw,depth,'aluminum',fullprojection/2)
      for k in range(1,style['mullions']+1):block(left+ow*k/(style['mullions']+1),(bottom+top)/2,.042,oh,.065,'aluminum',-.07)
      block(s,bottom+oh*style['transomRatio'],ow,.046,.065,'aluminum',-.07)
      block(s,bottom-.055,ow+.15,.11,.48,'concrete',.06)
      # Interior depth cues are synthetic, stable by grid cell; they do not assert interior truth.
      if row>0 and variant in [1,3,5]:
       ch=oh*(.18+.08*(variant%3));block(s,top-ch/2-fw,ow-fw*2,ch,.02,'curtain',-.16)
       for zz in np.arange(top-ch,top-.05,.07):block(s,zz,ow-fw*2,.016,.022,'recess',-.139)
      if row==0:
       if col==cols//2:
        block(s,bottom+.015,ow,.03,1.,'concrete',.1)
        for dx in [-.16,.16]:block(s+dx,bottom+oh*.43,.024,.45,.05,'steel',-.07)
        if style['ground']=='canopy_entry':
         block(s,top+.12,ow+.45,.12,1.8,'aluminum',.65)
         block(s,top+.20,ow+.2,.045,1.65,'glass_light',.68)
         for dx in [-ow*.4,ow*.4]:
          p=a+u*(s+dx);cylinder([*(p+n*.03),z0+top+.8],[*(p+n*1.35),z0+top+.2],.023,detail,'steel',6)
      components+=1
    # Continuous base, cornice and coping; stone bands only where observed.
    for zz,hh,dd in [(0.22,.44,.10),(main_height-.22,.15,.15),(main_height+.10,.18,.48)]:block(length/2,zz,length,hh,dd,'concrete',.05)
    if style.get('stoneBands'):
     for zz in [gh-.1,gh+fh*1.2]:block(length/2,zz,length,.32,.20,'concrete',.12)
 return records,components
