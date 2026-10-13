"""Compile wall observations and semantic priors into reusable architectural components."""
import math
import numpy as np
from shapely.geometry import Point,LineString
from shapely.ops import unary_union
from . import facade_grammar
from .surface_evidence import resolve_material

PART_HEIGHT_SHARE=.3


def belongs(p,c,height=None):
 """Whether a photographed reading describes this building part. It must be of the same feature (or OSM way). An
 automatic reading of this very part (match.partId) always belongs; one of another part only when the two wall heights
 (its frame heightRefM and this part's) are within 30 %: a part that much lower or taller is another mass. A hand
 reading covers its feature's parts as before: excluding it from parts much taller than its frame is right for a low
 building before a tall one, but the one such case in this area is a corner whose tall part's outline wrongly carries
 slivers into the low building's frontage, and there it made the slivers stand out pale (see NEXT_AGENT.md)."""
 m=p['match']
 if not (m.get('featureId')==c.get('sourceId',c['id']) or (c.get('osm') and m.get('osmWayId') is not None and str(m.get('osmWayId'))==str(c['osm']['id']))):return False
 part=m.get('partId')
 if part is not None and part==c['id']:return True
 ref=p.get('heightRefM')
 if height is None or not ref:return part is None
 ref,height=float(ref),float(height)
 if part is None:return True
 return abs(ref-height)<=PART_HEIGHT_SHARE*max(ref,height)


def observed_material(profiles,c,height=None):
 """Unobserved walls continue the material of the building's observed frontages.

 A building is usually clad in one material all round, so the frontage material is the best prior for its other
 walls. Explicit whole-envelope evidence settles it outright. Otherwise the frontages' own materials (not their
 panels, which are local) must agree: when two thirds or more of the observed frontages share one material it
 continues; frontages that disagree leave the prior. Removing this transfer turned most unobserved walls to
 the white fallback prior.
 """
 # A reading that saw under 40 % of its wall (behind trees, cars or another building) colours that wall, not the
 # whole building: a colour-only reading of a quarter of a wall would otherwise set every unseen wall of the part.
 # And the material every unseen wall of a part takes comes from readings of that mass: a hand reading drawn in a
 # frame 30 % lower than this part (a low building's frontage) still draws the walls it covers, but does not set
 # the taller part's material (a one-storey brick frontage's reading turned a glass-and-metal block brick).
 def of_this_mass(p):
  if p['match'].get('partId') is not None or height is None or not p.get('heightRefM'):return True
  return float(height)<=float(p['heightRefM'])*(1+PART_HEIGHT_SHARE)
 # A reading kept only because no better one exists (reader v34 marks it unsupported) draws its own wall and nothing more.
 matches=[p for p in profiles if belongs(p,c,height) and of_this_mass(p) and float((p.get('reading') or {}).get('visibleShare',1.))>=.4
          and not ((p.get('reading') or {}).get('consensus') or {}).get('unsupported')]
 if not matches:return None
 whole=[p for p in matches if p.get('materialEvidence',{}).get('scope')=='whole-envelope']
 if whole:
  materials={resolve_material(p) for p in whole}
  for p in whole:materials.update(resolve_material(q) for q in p.get('panels',[]) if q.get('material') or q.get('materialEvidence'))
  return next(iter(materials)) if len(materials)==1 else None
 if VOTE_BY_LIKENESS:return _vote_by_likeness(matches)
 counts={}
 for p in matches:
  m=resolve_material(p)
  if m:counts[m]=counts.get(m,0)+1
 if not counts:return None
 best,n=max(counts.items(),key=lambda kv:(kv[1],kv[0]))
 return best if n>=2/3*sum(counts.values()) else None


# Colours voting by likeness (readings within dE 15 in Lab are one cluster; the largest continues when it holds two
# thirds of the readings, in its family's recipe and median colour) fixes near-identical bricks splitting an exact-name
# vote three ways. It is off: a building whose frontage readings are all of one occluded photograph (a neighbour's base
# read as its own wall) then spreads that reading to every unseen wall, and it flipped a flagged building's walls from
# render to brick (2026-09-25). Enable after readings carry per-part evidence identity and an occlusion check.
VOTE_BY_LIKENESS=True


def _vote_by_likeness(matches):
 """The largest set of the frontage readings whose colours can all be one material under different light
 (pipeline/colour_agreement.same_material) continues when it holds two thirds of the readings, in its family's recipe
 and median colour."""
 from itertools import combinations
 from . import material_recipes
 from .colour_agreement import same_material
 mats=[m for m in (resolve_material(p) for p in matches) if m]
 if not mats:return None
 def rgb(m):
  base,_,hx=m.partition('#')
  return [int(hx[i:i+2],16) for i in (0,2,4)] if len(hx)==6 else list(material_recipes.RECIPES.get(base,{}).get('color',[180,180,180]))
 cols=[rgb(m) for m in mats];best=[0]
 for size in range(len(mats),1,-1):
  g=next((g for g in combinations(range(len(mats)),size) if all(same_material(cols[a],cols[b]) for a,b in combinations(g,2))),None)
  if g:best=list(g);break
 if len(best)<2/3*len(mats):return None
 fams=[mats[k].split('#')[0] for k in best];fam=max(set(fams),key=fams.count)
 med=np.median([cols[k] for k in best],0).astype(int)
 return fam if len(set(mats[k] for k in best))==1 and '#' not in mats[best[0]] else f"{fam}#{med[0]:02x}{med[1]:02x}{med[2]:02x}"

def observed_blank_level(profiles,c,height=None):
 """Height of a building's photographed frontage that shows no openings at all, or None.

 A wall this building was photographed with is blank (a planter or garden wall, a service wall): its unobserved
 pieces of the same height are blank walls too, in the observed material. The prior's shopfront and window row
 describe a storey; put on a 4 m garden wall round the corner they made a glazed base with pilasters.
 """
 for p in profiles:
  if belongs(p,c,height):
   if p.get('materialOnly'):continue   # an automatic reading without a clear grid says nothing about openings
   # Only a low blank wall (a garden or service wall) speaks for the building's other pieces; a tall blank reading is
   # one side (a party or flank wall) and says nothing about the windows of its street front.
   if not p.get('groups') and not p.get('openings') and p.get('heightRefM') and float(p['heightRefM'])<=6.:return float(p['heightRefM'])
 return None

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

def area_donor(profiles,fid,height):
 """A measured facade of this area to lend its character to a building no photograph covers: among the
 automatic readings with a measured grid, the five closest in wall height, one chosen by the building's own id
 (deterministic, so neighbours differ). Its values come from photographs of this area, never a fixed palette."""
 import hashlib
 pool=[p for p in profiles if str(p.get('status','')).startswith('image-measured-auto') and p.get('reading',{}).get('floorPitchM') and p.get('reading',{}).get('bayPitchM') and p.get('heightRefM')]
 if not pool:return None
 pool.sort(key=lambda p:(abs(math.log(max(float(p['heightRefM']),1.)/max(float(height),1.))),p['id']))
 near=pool[:5];return near[int(hashlib.md5(fid.encode()).hexdigest(),16)%len(near)]

def lend(style,donor):
 """The prior archetype with a donor's measured material, storey and bay pitch, opening proportions and ground."""
 r=donor['reading'];s=dict(style)
 s.update(material=resolve_material(donor),floorHeightM=float(np.clip(r['floorPitchM'],2.8,5.5)),bayWidthM=float(np.clip(r['bayPitchM'],1.4,8.)),
          openingRatio=float(r['openingRatio']),openingHeightRatio=float(r['openingHeightRatio']),ground='shopfront' if r.get('shopfront') else s.get('ground','entry'),
          mullions=2 if r['bayPitchM']*r['openingRatio']>2.6 else 1 if r['bayPitchM']*r['openingRatio']>1.4 else 0,donorId=donor['id'])
 return s

BIN_M=2.5   # along-wall roof reading resolution
STEP_M=1.5  # a roof height change larger than this between neighbouring bins is a step in the frontage

def wall_profile(along,z,length,bin_m=BIN_M):
 """Roof top (96th percentile) per bin along a wall, despiked; NaN where a bin has too few returns."""
 bins=max(2,int(round(length/bin_m)));index=np.clip((along/length*bins).astype(int),0,bins-1);tops=np.full(bins,np.nan)
 for b in range(bins):
  sel=index==b
  if sel.sum()>=4:tops[b]=np.quantile(z[sel],.96)
 return despike(tops)

def despike(tops,maximum_run=2,minimum_rise_m=3.0):
 """A run of at most `maximum_run` measured bins standing more than `minimum_rise_m` above the measured bins on
 both sides (on the one side that exists at the profile's end) is not the wall: a tree crown over a corner, a
 pole, a vent a bin or two wide. A parapet or false front runs the length of its wall and a tower is wider than
 two bins, so both stay. The run takes the height of its neighbours. Read raw, such a crown made the 2.6 m end
 piece of a frontage 4 m taller than the wall it belongs to, a dark block standing above the roof line."""
 tops=np.array(tops,float);idx=np.where(np.isfinite(tops))[0]
 if len(idx)<3:return tops
 vals=tops[idx].copy();n=len(vals);i=1
 while i<n:
  left=vals[i-1]
  if vals[i]<=left+minimum_rise_m:i+=1;continue
  j=i
  while j+1<n and vals[j+1]>left+minimum_rise_m:j+=1
  right=vals[j+1] if j+1<n else None
  if j-i+1<=maximum_run and (right is None or vals[i:j+1].min()>right+minimum_rise_m):vals[i:j+1]=max(v for v in (left,right) if v is not None)
  i=j+1
 for k in range(min(maximum_run,n-1)):   # a spike at the very start has only a right neighbour
  if vals[:k+1].min()>vals[k+1]+minimum_rise_m:vals[:k+1]=vals[k+1];break
 tops[idx]=vals;return tops

def is_sloped(tops,length):
 """The bin tops follow one trend along the wall: a sloped roof, not steps between flat parts."""
 ok=np.isfinite(tops)
 if ok.sum()<3 or np.nanmax(tops)-np.nanmin(tops)<=1.2:return False
 x=(np.where(ok)[0]+.5)/len(tops)*length;y=tops[ok];slope,icpt=np.polyfit(x,y,1);residual=y-(slope*x+icpt)
 return abs(slope)>.05 and np.abs(residual).max()<max(.4,.25*(y.max()-y.min()))

def wall_level(along,z,length):
 """Height a wall piece is raised to, from the roof returns along it.

 A level counts only where most of the piece reaches it (30th percentile of the bin tops), so a parapet
 running its whole length raises it while a stray higher return does not. Where the tops follow a trend
 the roof is sloped: the piece is raised only to its eave, the lowest bin, and the roof above it is a
 plain surface. Windows never climb a roof slope.
 """
 tops=wall_profile(along,z,length);valid=tops[np.isfinite(tops)]
 if len(valid)<2:return float(np.quantile(z,.96))
 if is_sloped(tops,length):return float(valid.min())
 return float(np.quantile(valid,.3))

def split_at_roof_steps(a,b,along,z,minimum_piece_m=2.5):
 """Cut one footprint edge where the roof height along it steps.

 A frontage shared by a low wing and a taller block is two walls, each as tall as its own roof; before,
 the whole edge rose to the tallest return near it and carried windows above the low roof. A sloped
 profile is not split (the eave rule applies to the whole edge).
 """
 a=np.asarray(a,float);b=np.asarray(b,float);u=b-a;length=float(np.linalg.norm(u))
 if length<2*minimum_piece_m:return [(tuple(a),tuple(b))]
 u=u/length;tops=wall_profile(along,z,length);bins=len(tops)
 if is_sloped(tops,length):return [(tuple(a),tuple(b))]
 cuts=[];previous=None
 for i in range(bins):
  if not np.isfinite(tops[i]):continue
  if previous is not None and abs(tops[i]-tops[previous])>STEP_M:cuts.append(i/bins*length)
  previous=i
 stations=[0.]+[s for s in cuts if minimum_piece_m<=s<=length-minimum_piece_m]+[length]
 merged=[stations[0]]
 for s in stations[1:]:
  if s-merged[-1]<minimum_piece_m:merged[-1]=s if s==length else merged[-1]
  else:merged.append(s)
 if merged[-1]!=length:merged.append(length)
 return [(tuple(a+u*s0),tuple(a+u*s1)) for s0,s1 in zip(merged,merged[1:]) if s1-s0>.02]

def frame_overlap(p,a,b):
 """Metres of the segment a-b inside a photographed profile's wall frame (its `frameU` along its own tangent)."""
 pn=np.asarray(p['normal'],float);pn/=np.linalg.norm(pn);pu=np.array([-pn[1],pn[0]]);f0,f1=map(float,p['frameU'])
 s0,s1=sorted([float(np.asarray(a)@pu),float(np.asarray(b)@pu)]);return max(0.,min(s1,f1)-max(s0,f0))

def split_at_frames(segments,frames):
 """Segments cut where a photographed frame begins or ends, so the photographed stretch and the rest of a long wall
 are drawn separately (the rest takes the generic facade, not the reading's bare material)."""
 out=[]
 for aa,bb in segments:
  a,b=np.asarray(aa,float),np.asarray(bb,float);cuts=[0.,1.]
  for p in frames:
   pn=np.asarray(p['normal'],float);pn/=np.linalg.norm(pn);u=b-a;L=np.linalg.norm(u)
   if L<2 or abs(np.dot(u/L,np.array([-pn[1],pn[0]])))<.96:continue
   pu=np.array([-pn[1],pn[0]]);sa,sb=float(a@pu),float(b@pu)
   for f in map(float,p['frameU']):
    t=(f-sa)/(sb-sa) if abs(sb-sa)>1e-6 else -1
    if .02<t<.98 and min(t,1-t)*L>1.:cuts.append(t)
  cuts=sorted(set(round(t,4) for t in cuts))
  out.extend((tuple(a+(b-a)*t0),tuple(a+(b-a)*t1)) for t0,t1 in zip(cuts,cuts[1:]))
 return out

def wall_exposure(ctx,c,mid,n,z0,main_height):
 """What a wall faces, from the building outlines and the mapped streets (no place-specific facts):
 'party'      another building touches it: blank, including whatever rises above the neighbour's roof;
 'own-podium' a lower part of the same building touches it and the wall rises a storey above (a tower over its
              podium keeps its facade);
 'hidden'     a part of the same building stands as high in front of it;
 'street'     the outward view reaches a mapped street or pedestrian way within 30 m before any other building;
 'gap'        another building stands within 3 m in front (a lot-line wall: blank);
 'secondary'  anything else (a yard, a lot, a courtyard): a lesser facade."""
 own=c.get('sourceId',c['id']);index=ctx.get('building_index')
 def who(p):
  pt=Point(*p)
  if index is None:return 'other' if ctx['all_buildings'].contains(pt) else None
  for i in index.query(pt):
   if ctx['candidates'][i]['g'].contains(pt):return ctx['building_sources'][i]
  return None
 at=who(mid+n*.8)
 if at is not None:
  if at==own and ctx.get('surfaceTop') is not None:
   neighbour=min(ctx['surfaceTop'](*(mid+n*d)) for d in (1.6,2.6,3.6))
   return 'own-podium' if neighbour<=z0+main_height-3. else 'hidden'
  return 'party'
 corridor=ctx.get('street_corridor')
 for d in (1.5,3.,6.,10.,15.,20.,25.,30.):
  p=mid+n*d;b=who(p)
  if b is not None and b!=own:return 'gap' if d<=3. else 'secondary'
  if corridor is not None and corridor.contains(Point(*p)):return 'street'
 return 'secondary'

def compile(ctx,c,z0,main_height,height,tags,observations):
 default_height=main_height;g=c['g'].simplify(.18,preserve_topology=True);fid=c['id'];cube=ctx['cube'];cylinder=ctx['cylinder'];records=[];components=0
 profiles=ctx.get('PROFILES',[]);fits=ctx.get('FITS',[])
 matches=[o for o in observations if str(o['match'].get('osmWayId'))==str(c['osm']['id'] if c['osm'] else '')]
 spans=ctx.get('envelopeSpans') or []
 own=[p for p in profiles if belongs(p,c,main_height)]
 donor=None if own or tags.get('leisure')=='stadium' or c['full_area']>18000 else area_donor(profiles,c.get('sourceId',c['id']),height)
 # A building photographed on one side lends what was measured there to its unseen walls: one building, one character.
 own_measured=next((p for p in own if p.get('reading',{}).get('floorPitchM') and p.get('reading',{}).get('bayPitchM') and not (p.get('reading',{}).get('consensus') or {}).get('unsupported')),None)
 def base_style():
  st=prior(tags,height,c['full_area'])
  if own_measured:return lend(st,own_measured)
  return lend(st,donor) if donor else st
 host_style=base_style()
 continuity=observed_material(profiles,c,main_height);blank_level=observed_blank_level(profiles,c,main_height)
 if continuity:host_style['material']=continuity
 def host_bands(aa,bb):
  """Host wall removed under a replacement slot is kept wherever the emitted box does not stand:
  below its floor and above its top return. The cut follows the emitted skin, not the frame."""
  a,b=np.asarray(aa),np.asarray(bb);u=b-a;length=np.linalg.norm(u)
  if length<.05:return
  u=u/length;n=np.array([u[1],-u[0]]);mid=(a+b)/2
  if g.contains(Point(*(mid+n*.2))):n=-n
  covered=sorted((max(s['zlo']-z0,0.),s['zhi']-z0) for s in spans if s['slot'].buffer(.03).contains(Point(*mid)))
  if not covered:return
  gaps=[];z=0.
  for lo,hi in covered:
   if lo>z+.05 and z<default_height:gaps.append((z,min(lo,default_height)))
   z=max(z,hi)
  if z<default_height-.05:gaps.append((z,default_height))
  angle=math.atan2(u[1],u[0]);thick,off=(.24,-.12) if c.get('roofBand') else (.38,-.19)
  for lo,hi in gaps:
   if hi-lo<.05:continue
   pos=mid+n*off;cube([pos[0],pos[1],z0+(lo+hi)/2],[length,thick,hi-lo],'buildings__'+fid,host_style['material'],angle)
 for poly in ctx['polys'](g):
  for ring in [poly.exterior]+list(poly.interiors):
   coords=list(ring.coords);segments=[];cut=ctx.get('envelopeMask')
   for aa,bb in zip(coords,coords[1:]):
    line=LineString([aa,bb])
    if cut is not None and not cut.is_empty:
     zone=cut.buffer(.015);remaining=line.difference(zone);removed=line.intersection(zone)
     # Adjacent slots replace different heights: split the removed run at slot boundaries first.
     if spans:removed=removed.difference(unary_union([s['slot'].boundary for s in spans]).buffer(.001))
     for part in ctx['lines'](removed):
      for pa,pb in zip(part.coords,list(part.coords)[1:]):host_bands(pa,pb)
    else:remaining=line
    for part in ctx['lines'](remaining):segments.extend(zip(part.coords,list(part.coords)[1:]))
   samples=ctx.get('roofSamples');levels=None
   if samples is not None and len(samples)>30 and not c.get('roofBand'):
    pieces=[];levels=[]
    for aa,bb in segments:
     a,b=np.asarray(aa),np.asarray(bb);u=b-a;length=np.linalg.norm(u)
     if length<5.:parts=[(aa,bb)]
     else:
      u=u/length;n=np.array([u[1],-u[0]]);delta=samples[:,:2]-a;along=delta@u;away=np.abs(delta@n);sel=(along>=0)&(along<=length)&(away<2.0)
      parts=split_at_roof_steps(a,b,along[sel],samples[sel,2]) if sel.sum()>20 else [(aa,bb)]
     # Each piece is raised to the level its own roof returns support; a piece with too few measured bins takes
     # the level of the nearest supported piece of the same edge rather than a percentile of stray returns.
     part_levels=[]
     for pa,pb in parts:
      pa_,pb_=np.asarray(pa),np.asarray(pb);pu=pb_-pa_;plen=np.linalg.norm(pu)
      if plen<.02:part_levels.append(None);continue
      pu=pu/plen;pn=np.array([pu[1],-pu[0]]);d=samples[:,:2]-pa_;al=d@pu;aw=np.abs(d@pn);msk=(al>=0)&(al<=plen)&(aw<2.0)
      part_levels.append(wall_level(al[msk],samples[msk,2],plen) if msk.sum()>20 else None)
     supported=[(i,l) for i,l in enumerate(part_levels) if l is not None]
     for i,l in enumerate(part_levels):
      if l is None and supported:part_levels[i]=min(supported,key=lambda s:abs(s[0]-i))[1]
     pieces.extend(parts);levels.extend(part_levels)
    segments=pieces
    # A recess: a piece shorter than a bay whose level lies more than 3 m below both neighbours' while the roof
    # 2-4 m behind it reaches their level is a shallow recess or an entrance canopy under the same wall, not a
    # lower wall (an outline drawn round a hotel's entrance recess gave its 2.6 m jambs the canopy's 4.9 m in a
    # 33 m frontage). Its wall continues at the neighbours' level.
    for i,(aa,bb) in enumerate(segments):
     if levels[i] is None or len(segments)<3:continue
     a_,b_=np.asarray(aa),np.asarray(bb);plen=float(np.linalg.norm(b_-a_))
     if plen<.02 or plen>=6.:continue
     nb=[levels[(i-1)%len(segments)],levels[(i+1)%len(segments)]]
     if any(v is None for v in nb):continue
     target=min(nb)
     if levels[i]>=target-3.:continue
     pu=(b_-a_)/plen;pn=np.array([pu[1],-pu[0]])
     if g.contains(Point(*((a_+b_)/2+pn*.2))):pn=-pn
     d=samples[:,:2]-a_;al=d@pu;aw=-(d@pn);msk=(al>=0)&(al<=plen)&(aw>=2.0)&(aw<4.0)
     if msk.sum()>=8 and float(np.quantile(samples[msk,2],.9))>=target-1.5:levels[i]=target
   frames=[p for p in profiles if p.get('frameU') and not p.get('materialOnly') and (belongs(p,c,main_height))]
   if frames:
    cut=split_at_frames(segments,frames)
    if levels is not None and len(cut)!=len(segments):
     # each cut piece keeps the level of the segment it came from
     def source(ca,cb):
      mid=Point(*((np.asarray(ca)+np.asarray(cb))/2))
      return min(range(len(segments)),key=lambda i:LineString(segments[i]).distance(mid))
     levels=[levels[source(ca,cb)] for ca,cb in cut]
    segments=cut
   for edge,(aa,bb) in enumerate(segments):
    a,b=np.asarray(aa),np.asarray(bb);u=b-a;length=np.linalg.norm(u)
    if length<.02:continue
    u=u/length;n=np.array([u[1],-u[0]]);mid=(a+b)/2
    if g.contains(Point(*(mid+n*.2))):n=-n
    main_height=default_height
    if c.get('elevationOffset',0)>0 and c.get('outerBoundary') is not None and c['outerBoundary'].distance(Point(*mid))<.18:continue
    samples=ctx.get('roofSamples')
    if levels is not None:
     level=levels[edge]
     if level is not None:main_height=max(.5 if ctx.get('roofSamplesOwned') else default_height,min(height,float(level-z0)))
    # A wall never rises above the roof envelope actually built plus a parapet: LiDAR returns of a dropped tier or of
    # plant near the edge would otherwise stand a wall open above the roof.
    if ctx.get('wallTopCapZ') is not None:main_height=min(main_height,max(default_height,float(ctx['wallTopCapZ']-z0)))
    angle=math.atan2(u[1],u[0]);style=base_style()
    # A colour-only reading colours every wall of its facing when it saw most of its wall; one that saw under 40 %
    # colours only the wall its frame covers (a quarter-seen corner wall's brick had spread along a whole facade).
    facing=next((p for p in profiles if p.get('materialOnly') and (belongs(p,c,main_height)) and np.dot(n,p['normal'])>.95
                 and (float((p.get('reading') or {}).get('visibleShare',1.))>=.4 or (p.get('frameU') and frame_overlap(p,a,b)>=.5*length
                      and (p.get('planeV') is None or abs(float(mid@(n/np.linalg.norm(n)))-float(p['planeV']))<=1.5)))),None)
    if facing:style['material']=resolve_material(facing)   # this facing's measured material, the prior's openings
    matching=[p for p in profiles if not p.get('materialOnly') and (belongs(p,c,main_height)) and np.dot(n,p['normal'])>.95]
    # A hand reading (no frame) covers its whole facing and takes precedence. Otherwise a photographed reading draws
    # the wall it covers: of several, the one covering most of this piece, and only if it covers at least half of it.
    framed=[(frame_overlap(p,a,b),p) for p in matching if p.get('frameU')]
    unframed=[p for p in matching if not p.get('frameU')]
    best=max(framed,key=lambda x:x[0],default=(0.,None))
    profile=unframed[0] if unframed else (best[1] if best[1] is not None and best[0]>=.5*length else None)
    if profile and c.get('elevationOffset',0)<.5:
     # The wall a profile is drawn on is this piece's LiDAR wall level (main_height): the height the roof returns
     # along the wall reach, raised only by a photographed parapet (observedWallTopM, at most 3 m). The tallest
     # return cluster near the wall, or the building maximum, is never this wall's height: a taller block behind
     # a low frontage stands on its own roof tier, and a wall drawn to its height is a wall the LiDAR never saw
     # (a 10 m podium came out as a 37 m blank wall, a 4 m planter wall as a 31 m block with strip windows).
     # A measured profile carries the metre frame its fractions refer to (heightRefM). A hand-annotated profile's
     # fractions were authored against the cluster height, so that height becomes its frame: its features land
     # at the metres the author meant, and whatever rises above the wall is clipped away with the wall.
     frame=float(profile['heightRefM']) if profile.get('heightRefM') else None
     if frame is None:
      frame=float(height)
      if samples is not None and len(samples)>30:
       delta=samples[:,:2]-a;along=delta@u;away=np.abs(delta@n);near=samples[(along>=-1)&(along<=length+1)&(away<3.5)&(samples[:,2]>z0+height*.65)]
       if len(near)>15:
        bins=np.arange(z0+height*.65,z0+height+1,.5);hist,edges=np.histogram(near[:,2],bins);peak=int(np.argmax(hist));cluster=near[(near[:,2]>=edges[peak]-.5)&(near[:,2]<edges[peak]+1)]
        if len(cluster):frame=min(float(height),float(np.quantile(cluster[:,2],.85)-z0+.12))
     wall_top=float(main_height)
     if profile.get('observedWallTopM'):
      # A parapet is a thin wall the roof-return statistics do not see; a photograph shows where the frontage
      # ends. The observed top may raise the wall above the LiDAR level, by at most 3 m; it never lowers it.
      wall_top=min(max(wall_top,float(profile['observedWallTopM'])),wall_top+3.)
     fitted=facade_grammar.compile_wall(ctx,c,a,b,z0,wall_top,{**profile,'heightRefM':round(frame,2)},fits)
     if fitted:
      fitted['record'].update(wallLevelM=round(float(main_height),2),wallTopM=round(wall_top,2),frameHeightM=round(frame,2))
      records.append(fitted['record']);components+=fitted['components'];continue
    obs=None if profile else next((o for o in matches if np.dot(n,o['normal'])>.82),None)
    if obs:style.update(obs)
    elif continuity and not facing:style['material']=continuity
    # A wall is hidden only where another building stands in front of it up to its top: a wall rising above a lower
    # neighbour is seen. The neighbour's height is the measured surface 1.6-3.6 m out (the lowest of three samples,
    # so the wall's own returns do not count).
    # What the wall faces decides whether it carries windows: a wall another building touches is a party wall and
    # stays blank even where it rises above a lower neighbour; a wall over a narrow gap is a lot-line wall; a wall to a
    # yard or lot is a lesser facade; a street wall, or a tower face over its own podium, is a full one.
    exposure=wall_exposure(ctx,c,mid,n,z0,main_height)
    # A facing a street camera photographed is seen from the street, whatever the mapped streets say (plazas and wide
    # set-backs put the corridor beyond reach); a touching neighbour still makes a party wall.
    if exposure in ('secondary','gap') and any(np.dot(n,p['normal'])>.95 and (belongs(p,c,main_height)) for p in profiles):exposure='street'
    visible=length>=3.5 and exposure in ('street','own-podium','secondary') and ctx['boundary'].boundary.distance(Point(*mid))>.5
    if exposure=='secondary':style=dict(style,bayWidthM=style['bayWidthM']*2,openingRatio=style['openingRatio']*.6)
    if ctx.get('venueField') is not None and ctx['venueField'].distance(Point(*mid))<6.:
     # The edge of a venue's footprint that faces its field is the seating bowl's front: a low plain wall.
     main_height=min(main_height,2.6);visible=False
    mat=style['material'];layer='buildings__'+fid;detail='facades__'+fid
    def block(s,z,w,h,d,material,depth=0,key=detail):
     if w<=.001 or h<=.001:return
     pos=a+u*s+n*depth;cube([pos[0],pos[1],z0+z],[w,d,h],key,material,angle)
    if blank_level is not None and not c.get('roofBand') and main_height<=blank_level+1.0:
     # The photographed frontage at this height is a blank wall: so is this unobserved piece of the same height.
     block(length/2,main_height/2,length,main_height,.38,mat,-.19,layer)
     records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':0,'columns':0,'archetype':style['archetype'],'material':mat,'sourceIds':[],'status':'inferred','unresolved':['blank wall by continuity with the photographed blank frontage of this building at this height']});continue
    if c.get('roofBand') and main_height<3.5:
     # A roof band lower than a storey is an opaque riser, never an invented window floor. A band a storey or more
     # high is a volume with walls of its own and takes the same facade rules as any wall; blank storey-high bands
     # were the windowless towers.
     block(length/2,main_height/2,length,main_height,.24,mat,-.12,layer);continue
    if not visible:
     block(length/2,main_height/2,length,main_height,.38,mat,-.19,layer)
     records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':0,'columns':0,'archetype':style['archetype'],'material':mat,'sourceIds':[],'status':'blank','exposure':exposure,'unresolved':[]})
     continue
    cols=max(1,round(length/style['bayWidthM']));bay=length/cols;ground_level=c.get('elevationOffset',0)<.5;gh=min(style['groundHeightM'],main_height*.42) if ground_level else 0
    rows=max(1,round((main_height-gh)/style['floorHeightM']));fh=(main_height-gh)/rows
    wid=bay*style['openingRatio'];opening_h=fh*style['openingHeightRatio'];fw=style['frameWidthM'];projection=style['frameProjectionM'];recess=.18
    records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':rows+1,'columns':cols,'archetype':style['archetype'],'material':mat,'sourceIds':obs.get('sourceIds',[]) if obs else [],'exposure':exposure,'status':'visual-fit' if obs else 'inferred','unresolved':obs.get('unresolved',[]) if obs else ['complete facade has no direct image fit; material and openings remain unverified priors']})
    for row in range(0 if ground_level else 1,rows+1):
     low=0 if row==0 else gh+(row-1)*fh;rh=gh if row==0 else fh
     if row==0 and exposure=='secondary':
      # A lesser facade has no shopfront: its ground storey is wall (service doors are not invented).
      block(length/2,rh/2,length,rh,.38,mat,-.19,layer);continue
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
