"""Compile wall observations and semantic priors into reusable architectural components."""
import math
import numpy as np
from shapely.geometry import Point,LineString
from shapely.ops import unary_union
from . import facade_grammar
from .surface_evidence import resolve_material

PART_HEIGHT_SHARE=.3


def belongs(p,c,height=None,mass=None):
 """Whether a photographed reading describes this building part. It must be of the same feature (or OSM way). An
 automatic reading of this very part (match.partId) always belongs; one of another part only when the two wall heights
 (its frame heightRefM and this part's) are within 30 %: a part that much lower or taller is another mass. A hand
 reading names no part: on a split outline it belongs to the parts whose main roof level (`mass`, the height the part's
 outer walls stand to; not its tallest point, and not a tier's short wall) is within 30 % of its frame. Covering every part, a low building's
 hand-read brick front was drawn on the tall block standing behind it as a blank brick box that hid the low building's
 gable; that was kept only while the tall part's outline carried slivers into the low frontage, which it no longer does."""
 m=p['match']
 if not (m.get('featureId')==c.get('sourceId',c['id']) or (c.get('osm') and m.get('osmWayId') is not None and str(m.get('osmWayId'))==str(c['osm']['id']))):return False
 part=m.get('partId')
 if part is not None and part==c['id']:return frame_on_part(p,c)
 ref=p.get('heightRefM')
 if part is None:
  if not c.get('partOf') or mass is None or not ref:return True
  return abs(float(ref)-float(mass))<=PART_HEIGHT_SHARE*max(float(ref),float(mass))
 if height is None or not ref:return False
 ref,height=float(ref),float(height)
 return abs(ref-height)<=PART_HEIGHT_SHARE*max(ref,height)


def frame_cover(g,p):
 """Metres of a photographed reading's frame (frameU along its wall, within 1.5 m of its plane planeV when it has one)
 that lie along outline g's walls facing the reading's way."""
 from shapely.geometry.polygon import orient
 pn=np.asarray(p['normal'],float)[:2];pn=pn/np.linalg.norm(pn);pu=np.array([-pn[1],pn[0]]);f0,f1=sorted(map(float,p['frameU']));spans=[]
 for poly in ([g] if g.geom_type=='Polygon' else list(getattr(g,'geoms',[]))):
  xy=np.asarray(orient(poly,1).exterior.coords)
  for a,b in zip(xy,xy[1:]):
   v=b-a;L=float(np.linalg.norm(v))
   if L<.3 or float(np.array([v[1],-v[0]])@pn)/L<.95:continue
   if p.get('planeV') is not None and abs(float(((a+b)/2)@pn)-float(p['planeV']))>1.5:continue
   s0,s1=sorted([float(a@pu),float(b@pu)]);lo,hi=max(s0,f0),min(s1,f1)
   if hi>lo:spans.append((lo,hi))
 total=0.;cur=None
 for lo,hi in sorted(spans):
  if cur is None or lo>cur[1]:
   total+=(cur[1]-cur[0]) if cur else 0.;cur=[lo,hi]
  else:cur[1]=max(cur[1],hi)
 return total+((cur[1]-cur[0]) if cur else 0.)

def frame_on_part(p,c,share=.5):
 """Whether a reading filed under this part still lies on it. The reader names the part it read a wall against; when
 the outline is split differently later, that name can go stale: a frame read mostly on a low building's side was
 filed under the tall block behind it, and the 1.7 m of it that falls on the tall block's wall stood out as a thin
 wall in the photograph's colour. The part owns the reading when it holds at least half of the frame the whole
 outline holds."""
 whole=c.get('sourceG')
 if whole is None or not p.get('frameU') or p.get('normal') is None:return True
 held=frame_cover(whole,p)
 return held<=0 or frame_cover(c['g'],p)>=share*held

def observed_material(profiles,c,height=None,mass=None):
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
 matches=[p for p in profiles if belongs(p,c,height,mass) and of_this_mass(p) and float((p.get('reading') or {}).get('visibleShare',1.))>=.4
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

def observed_blank_level(profiles,c,height=None,mass=None):
 """Height of a building's photographed frontage that shows no openings at all, or None.

 A wall this building was photographed with is blank (a planter or garden wall, a service wall): its unobserved
 pieces of the same height are blank walls too, in the observed material. The prior's shopfront and window row
 describe a storey; put on a 4 m garden wall round the corner they made a glazed base with pilasters.
 """
 for p in profiles:
  if belongs(p,c,height,mass):
   if p.get('materialOnly'):continue   # an automatic reading without a clear grid says nothing about openings
   # Only a low blank wall (a garden or service wall) speaks for the building's other pieces; a tall blank reading is
   # one side (a party or flank wall) and says nothing about the windows of its street front.
   if not p.get('groups') and not p.get('openings') and p.get('heightRefM') and float(p['heightRefM'])<=6.:return float(p['heightRefM'])
 return None

def framed_blank_level(profiles,c,g,segments,levels,z0,height=None,mass=None):
 """(height, reading) of a photographed blank wall whose reading was framed on a taller level, else (None, None).

 A hand reading of a low planter wall standing in front of a set-back block carries the block's height as its frame
 (21 m); it still describes the low wall it was drawn on, and the LiDAR gives that wall's height along the pieces of its
 own facing. Tested on the frame (6 m or less), the reading never counted, and the podium's other walls took the prior's
 shopfronts and dark garage openings where the photographs show one blank tiled wall."""
 for p in profiles:
  if not belongs(p,c,height,mass) or p.get('materialOnly') or p.get('groups') or p.get('openings') or p.get('frameU'):continue
  if not p.get('heightRefM') or float(p['heightRefM'])<=6.:continue
  pn=np.asarray(p['normal'],float)[:2];pn=pn/np.linalg.norm(pn);hs=[];ws=[]
  for (aa,bb),lv in zip(segments,levels):
   if lv is None:continue
   a,b=np.asarray(aa,float),np.asarray(bb,float);u=b-a;L=float(np.linalg.norm(u))
   if L<1.:continue
   u=u/L;n=np.array([u[1],-u[0]])
   if g.contains(Point(*((a+b)/2+n*.2))):n=-n
   if float(n@pn)>.95:hs.append(float(lv-z0));ws.append(L)
  if hs:
   order=np.argsort(hs);cum=np.cumsum(np.asarray(ws)[order]);h=float(np.asarray(hs)[order][int(np.searchsorted(cum,cum[-1]/2))])
   if h<=6.:return h,p
 return None,None

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

def measured_grid(p):
 """A reading's measured storey and bay grid: its reading's pitches, or else those of a pitch-given bank (a reading whose
 windows were laid out from the photograph's periodic edges carries them only there). None when neither exists."""
 r=p.get('reading') or {}
 if r.get('floorPitchM') and r.get('bayPitchM'):return r
 for g in p.get('groups',[]):
  st=g.get('style') or {}
  if g.get('rowPitchM') and g.get('columnPitchM') and st.get('kind','window')=='window' and st.get('openingRatio') and st.get('openingHeightRatio'):
   return {'floorPitchM':float(g['rowPitchM']),'bayPitchM':float(g['columnPitchM']),'openingRatio':float(st['openingRatio']),
           'openingHeightRatio':float(st['openingHeightRatio']),'shopfront':bool(r.get('shopfront')),**({'ribbon':bool(st['ribbon'])} if 'ribbon' in st else {})}
 return None

def lend(style,donor):
 """The prior archetype with a donor's measured material, storey and bay pitch, opening proportions and ground."""
 r=measured_grid(donor) or donor['reading'];s=dict(style)
 s.update(material=resolve_material(donor),floorHeightM=float(np.clip(r['floorPitchM'],2.8,5.5)),bayWidthM=float(np.clip(r['bayPitchM'],1.4,8.)),
          openingRatio=float(r['openingRatio']),openingHeightRatio=float(r['openingHeightRatio']),ground='shopfront' if r.get('shopfront') else s.get('ground','entry'),
          mullions=2 if r['bayPitchM']*r['openingRatio']>2.6 else 1 if r['bayPitchM']*r['openingRatio']>1.4 else 0,donorId=donor['id'],
          ribbon=bool(r['ribbon']) if 'ribbon' in r else r['bayPitchM']*(1-r['openingRatio'])<=facade_grammar.RIBBON_PIER_M)
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

def gable_eave(tops,length):
 """The eave height of an end wall under a pitched roof, or None. The bin tops rise to one peak and fall again, each
 side on its own trend; the eave is where the two rake lines meet the wall's ends (a bin's top reads the high end of
 its stretch of slope, 0.7 m above the eave on an 18-degree gable). Such a wall stops at its eaves and the triangle
 above is the roof's end: a flat top at a percentile of the rake stood above the roof slope at both corners and cut
 the ridge off."""
 ok=np.isfinite(tops)
 if ok.sum()<5 or np.nanmax(tops)-np.nanmin(tops)<=1.2:return None
 x=(np.where(ok)[0]+.5)/len(tops)*length;y=tops[ok];tol=max(.4,.25*(y.max()-y.min()))
 for k in range(2,len(y)-2):
  s1,i1=np.polyfit(x[:k+1],y[:k+1],1);s2,i2=np.polyfit(x[k:],y[k:],1)
  if s1>.05 and s2<-.05 and np.abs(y[:k+1]-(s1*x[:k+1]+i1)).max()<tol and np.abs(y[k:]-(s2*x[k:]+i2)).max()<tol:
   w=length/len(tops)   # a bin's top is its high end, half a bin up the slope from its centre
   return float(max(min(i1-s1*w/2,s2*length+i2+s2*w/2),y.min()-1.5))
 return None

def is_gable(tops,length):
 return gable_eave(tops,length) is not None

def wall_level(along,z,length,pitched=False):
 """Height a wall piece is raised to, from the roof returns along it.

 A level counts only where most of the piece reaches it (30th percentile of the bin tops), so a parapet
 running its whole length raises it while a stray higher return does not. Where the tops follow a trend
 the roof is sloped: the piece is raised only to its eave, the lowest bin, and the roof above it is a
 plain surface. Windows never climb a roof slope. Under a pitched roof the built model has (`pitched`), an end wall
 whose tops rise to a ridge and fall again is raised to its eaves the same way, and the roof solid closes the gable.
 """
 tops=wall_profile(along,z,length);valid=tops[np.isfinite(tops)]
 if len(valid)<2:return float(np.quantile(z,.96))
 if pitched:
  eave=gable_eave(tops,length)
  if eave is not None:return eave
 if is_sloped(tops,length):return float(valid.min())
 return float(np.quantile(valid,.3))

def split_at_roof_steps(a,b,along,z,minimum_piece_m=2.5,pitched=False):
 """Cut one footprint edge where the roof height along it steps.

 A frontage shared by a low wing and a taller block is two walls, each as tall as its own roof; before,
 the whole edge rose to the tallest return near it and carried windows above the low roof. A sloped
 profile is not split (the eave rule applies to the whole edge).
 """
 a=np.asarray(a,float);b=np.asarray(b,float);u=b-a;length=float(np.linalg.norm(u))
 if length<2*minimum_piece_m:return [(tuple(a),tuple(b))]
 u=u/length;tops=wall_profile(along,z,length);bins=len(tops)
 if is_sloped(tops,length) or (pitched and is_gable(tops,length)):return [(tuple(a),tuple(b))]
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

def tier_runs(c,g,segments,step=.5):
 """Pieces of a tier's wall segments over which the same things stand at either side: whether a higher tier stands flush
 over the wall (reaching within 0.3 m inside it) and which lower roof stands in front (0.3 m outside), sampled every
 `step` metres. Judged once at a wall's midpoint, a 21 m face in front of which a small glazed vault stood at its middle
 took the vault's roof as its foot along its whole length and stood on a false ledge 7 m up."""
 above=c.get('tierAbove');below=c.get('tiersBelow',());base=float(c['tierBase']);out=[]
 def state(p,n):
  flush=above is not None and above.contains(Point(*(p-n*.3)))
  zs=base
  for geo,ztop in below:
   if geo.contains(Point(*(p+n*.3))):zs=max(zs,float(ztop))
  return flush,round(zs,3)
 for aa,bb in segments:
  a,b=np.asarray(aa,float),np.asarray(bb,float);u=b-a;L=float(np.linalg.norm(u))
  if L<.02:continue
  u=u/L;n=np.array([u[1],-u[0]])
  if g.contains(Point(*((a+b)/2+n*.2))):n=-n
  k=max(1,int(np.ceil(L/step)));ts=(np.arange(k)+.5)/k*L;st=[state(a+u*t,n) for t in ts]
  i0=0
  for i in range(1,k+1):
   if i==k or st[i]!=st[i0]:
    if not st[i0][0]:out.append((tuple(a+u*(i0/k*L)),tuple(a+u*(i/k*L))))
    i0=i
 return out

def compile(ctx,c,z0,main_height,height,tags,observations):
 default_height=main_height;g=c['g'].simplify(.18,preserve_topology=True);fid=c['id'];cube=ctx['cube'];cylinder=ctx['cylinder'];records=[];components=0
 # A hand reading's frame is compared with the part's main roof level, not its tallest point (a tower corner made a
 # 36 m part of a 22 m frontage and its own hand reading stopped counting) and not a tier's short wall.
 mass=c.get('massHeightM',height)
 profiles=ctx.get('PROFILES',[]);fits=ctx.get('FITS',[])
 matches=[o for o in observations if str(o['match'].get('osmWayId'))==str(c['osm']['id'] if c['osm'] else '')]
 spans=ctx.get('envelopeSpans') or []
 own=[p for p in profiles if belongs(p,c,main_height,mass)]
 donor=None if own or tags.get('leisure')=='stadium' or c['full_area']>18000 else area_donor(profiles,c.get('sourceId',c['id']),height)
 # A building photographed on one side lends what was measured there to its unseen walls: one building, one character.
 own_measured=next((p for p in own if measured_grid(p) and not p.get('materialOnly') and not (p.get('reading',{}).get('consensus') or {}).get('unsupported')),None)
 def base_style():
  st=prior(tags,height,c['full_area'])
  if own_measured:return lend(st,own_measured)
  return lend(st,donor) if donor else st
 host_style=base_style()
 continuity=observed_material(profiles,c,main_height,mass);blank_level=observed_blank_level(profiles,c,main_height,mass)
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
   if c.get('roofBand') and c.get('tierBase') is not None:segments=tier_runs(c,g,segments)
   samples=ctx.get('roofSamples');levels=None;gables=None;pitched_roof=ctx.get('pitchedRoof')
   def under_pitch(p,q):
    # the piece runs under a pitched roof the model builds (most of its length within 1 m of an accepted plane)
    if pitched_roof is None or pitched_roof.is_empty:return False
    line=LineString([tuple(p),tuple(q)]);return line.length>0 and line.intersection(pitched_roof.buffer(1.)).length>=.6*line.length
   if samples is not None and len(samples)>30 and not c.get('roofBand'):
    pieces=[];levels=[];gables=[]
    for aa,bb in segments:
     a,b=np.asarray(aa),np.asarray(bb);u=b-a;length=np.linalg.norm(u)
     if length<5.:parts=[(aa,bb)]
     else:
      u=u/length;n=np.array([u[1],-u[0]]);delta=samples[:,:2]-a;along=delta@u;away=np.abs(delta@n);sel=(along>=0)&(along<=length)&(away<2.0)
      parts=split_at_roof_steps(a,b,along[sel],samples[sel,2],pitched=under_pitch(a,b)) if sel.sum()>20 else [(aa,bb)]
     # Each piece is raised to the level its own roof returns support; a piece with too few measured bins takes
     # the level of the nearest supported piece of the same edge rather than a percentile of stray returns.
     part_levels=[]
     for pa,pb in parts:
      pa_,pb_=np.asarray(pa),np.asarray(pb);pu=pb_-pa_;plen=np.linalg.norm(pu)
      if plen<.02:part_levels.append(None);gables.append(False);continue
      pu=pu/plen;pn=np.array([pu[1],-pu[0]]);d=samples[:,:2]-pa_;al=d@pu;aw=np.abs(d@pn);msk=(al>=0)&(al<=plen)&(aw<2.0)
      pitched=under_pitch(pa_,pb_)
      part_levels.append(wall_level(al[msk],samples[msk,2],plen,pitched=pitched) if msk.sum()>20 else None)
      gables.append(bool(msk.sum()>20 and pitched and is_gable(wall_profile(al[msk],samples[msk,2],plen),plen)))
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
   frames=[p for p in profiles if p.get('frameU') and not p.get('materialOnly') and (belongs(p,c,main_height,mass))]
   if frames:
    cut=split_at_frames(segments,frames)
    if levels is not None and len(cut)!=len(segments):
     # each cut piece keeps the level of the segment it came from
     def source(ca,cb):
      mid=Point(*((np.asarray(ca)+np.asarray(cb))/2))
      return min(range(len(segments)),key=lambda i:LineString(segments[i]).distance(mid))
     levels=[levels[source(ca,cb)] for ca,cb in cut];gables=[gables[source(ca,cb)] for ca,cb in cut]
    segments=cut
   # Pieces of a rounded corner (short segments turning a little each, a quarter turn or so together) are one curved wall:
   # each piece is a bay of the facade it turns between, not a stub too short to carry a window.
   arc_ids=set();run=[]
   dirs=[(np.asarray(bb,float)-np.asarray(aa,float)) for aa,bb in segments];lens=[float(np.linalg.norm(d)) for d in dirs]
   for i in range(len(segments)):
    if not 0.<lens[i]<3.5:                     # the outline is simplified to 18 cm: a larger radius comes out in longer facets
     if len(run)>=3:arc_ids.update(run)
     run=[];continue
    if run:
     ang=math.degrees(math.acos(float(np.clip((dirs[run[-1]]/lens[run[-1]])@(dirs[i]/lens[i]),-1.,1.))))
     if not 3.<ang<35.:
      if len(run)>=3:arc_ids.update(run)
      run=[i];continue
    run.append(i)
   if len(run)>=3:arc_ids.update(run)
   ring_blank,blank_src=blank_level,None
   if ring_blank is None and levels is not None and not c.get('roofBand'):ring_blank,blank_src=framed_blank_level(profiles,c,g,segments,levels,z0,main_height,mass)
   gable_mark=None
   def end_gable():
    # records of a gable end drawn to its ridge: the roof solid's end face under it is not drawn (the wall is)
    for r in records[gable_mark:]:r['gableEnd']=True
    if ctx.get('set_clip'):ctx['set_clip'](None)
   for edge,(aa,bb) in enumerate(segments):
    if gable_mark is not None:end_gable();gable_mark=None
    a,b=np.asarray(aa),np.asarray(bb);u=b-a;length=np.linalg.norm(u)
    if length<.02:continue
    u=u/length;n=np.array([u[1],-u[0]]);mid=(a+b)/2
    if g.contains(Point(*(mid+n*.2))):n=-n
    main_height=default_height
    if c.get('elevationOffset',0)>0 and c.get('outerBoundary') is not None and c['outerBoundary'].distance(Point(*mid))<.18:continue
    zs=z0
    if c.get('roofBand') and c.get('tierBase') is not None:
     # Walls of stacked tiers standing in one plane are one wall. A tier wall with a higher tier flush over it (reaching
     # within 0.3 m inside it) is not drawn; the higher tier's wall comes down to the roof actually standing in front of
     # it: the top of the highest lower tier there, else the base roof. Drawn per tier, each with its own base band and
     # cornice, one straight block stood as three stacked boxes with a ledge at every join.
     if c.get('tierAbove') is not None and c['tierAbove'].contains(Point(*(mid-n*.3))):continue
     zs=float(c['tierBase']);outside=Point(*(mid+n*.3))
     for geo,ztop in c.get('tiersBelow',()):
      if geo.contains(outside):zs=max(zs,float(ztop))
     main_height=float(z0+default_height-zs)
    samples=ctx.get('roofSamples')
    if levels is not None:
     level=levels[edge]
     if level is not None:main_height=max(.5 if ctx.get('roofSamplesOwned') else default_height,min(height,float(level-z0)))
    # A wall never rises above the roof envelope actually built plus a parapet: LiDAR returns of a dropped tier or of
    # plant near the edge would otherwise stand a wall open above the roof.
    if ctx.get('wallTopCapZ') is not None:main_height=min(main_height,max(default_height,float(ctx['wallTopCapZ']-z0)))
    gable_planes=None
    if gables is not None and gables[edge] and ctx.get('pitchedModels') and ctx.get('set_clip'):
     # A gable end is drawn to its ridge with whatever its reading shows, and everything on it is cut by the roof's own
     # planes (10 cm proud of them): the wall's top follows the rake and a band or window in the gable stays. Stopped
     # at its eaves, a stepped gable lost its hand-read band and peak window and took a cornice along its eaves.
     from general_world.roof_planes import evaluate as roof_evaluate
     # Only under a roof patch that runs the wall's whole length, and only as high as the LiDAR along the wall goes: a
     # plane evaluated past the end of its patch kept rising, and the wall stood a triangle above the real roof.
     seg=LineString([tuple(a),tuple(b)]);line=np.array([a+u*t for t in np.linspace(0,length,max(3,int(length/.5)))])
     near=samples[(np.abs((samples[:,:2]-a)@n)<2.)&(((samples[:,:2]-a)@u)>=0)&(((samples[:,:2]-a)@u)<=length)] if samples is not None else None
     lidar_top=float(np.quantile(near[:,2],.98)) if near is not None and len(near)>20 else None
     for gg,mm in ctx['pitchedModels']:
      if seg.intersection(gg.buffer(1.)).length<.9*length:continue
      top=float(roof_evaluate(mm,line).max())
      if lidar_top is None or top>lidar_top+.75:continue
      main_height=max(main_height,top-z0);gable_planes=[((-float(q[0]),-float(q[1]),1.),float(q[2])+.10) for q in mm['planes']]
      gable_mark=len(records);ctx['set_clip'](gable_planes);break
    angle=math.atan2(u[1],u[0]);style=base_style()
    def plane_gap(p):
     # a reading's planeV is its wall's offset along the reading's own normal: measure this wall along that same normal
     # (a reading taken on an edge 6 degrees off this wall, 100 m from the origin, was 0.6 m off its plane but 10 m off
     # along this wall's normal, and a blank flank lost its photographed reading)
     pn=np.asarray(p['normal'],float)[:2];return abs(float(mid@(pn/np.linalg.norm(pn)))-float(p['planeV']))
    # A colour-only reading colours every wall of its facing when it saw most of its wall; one that saw under 40 %
    # colours only the wall its frame covers (a quarter-seen corner wall's brick had spread along a whole facade).
    facing=next((p for p in profiles if p.get('materialOnly') and (belongs(p,c,main_height,mass)) and np.dot(n,p['normal'])>.95
                 and (float((p.get('reading') or {}).get('visibleShare',1.))>=.4 or (p.get('frameU') and frame_overlap(p,a,b)>=.5*length
                      and (p.get('planeV') is None or plane_gap(p)<=1.5)))),None)
    if facing:style['material']=resolve_material(facing)   # this facing's measured material, the prior's openings
    # A reading measured on one plane draws only walls standing in it (the grammar's own 2.5 m frontage tolerance): the
    # face of a block set back behind its podium is read on its own plane and must not land on the podium wall below,
    # nor the podium's reading on the block.
    matching=[p for p in profiles if not p.get('materialOnly') and (belongs(p,c,main_height,mass)) and np.dot(n,p['normal'])>.95
              and (p.get('planeV') is None or plane_gap(p)<=2.5)]
    # A hand reading (no frame) covers its whole facing and takes precedence. Otherwise a photographed reading draws
    # the wall it covers: of several, the one covering most of this piece, and only if it covers at least half of it.
    framed=[(frame_overlap(p,a,b),p) for p in matching if p.get('frameU')]
    unframed=[p for p in matching if not p.get('frameU')]
    best=max(framed,key=lambda x:x[0],default=(0.,None))
    profile=unframed[0] if unframed else (best[1] if best[1] is not None and best[0]>=.5*length else None)
    # A set-back tier wall takes a reading only when that reading was measured on this very plane from this wall's base
    # (the reader rectifies tier walls from their own foot, zBaseM); a ground-level reading never climbs onto a tier.
    tier_reading=bool(profile and c.get('roofBand') and profile.get('planeV') is not None and profile.get('zBaseM') is not None
                      and abs(float(profile['zBaseM'])-(zs-(z0-c.get('elevationOffset',0.))))<=1.5)
    if profile and (c.get('elevationOffset',0)<.5 or tier_reading):
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
     # a gable end stops at its eaves whatever the photograph's frame: the triangle above is the roof's end
     if profile.get('observedWallTopM') and not (gables is not None and gables[edge]):
      # A parapet is a thin wall the roof-return statistics do not see; a photograph shows where the frontage
      # ends. The observed top may raise the wall above the LiDAR level, by at most 3 m; it never lowers it.
      wall_top=min(max(wall_top,float(profile['observedWallTopM'])),wall_top+3.)
     fitted=facade_grammar.compile_wall(ctx,c,a,b,zs,wall_top,{**profile,'heightRefM':round(frame,2),'gableEnd':gable_planes is not None},fits)
     if fitted:
      fitted['record'].update(wallLevelM=round(float(main_height),2),wallTopM=round(wall_top,2),frameHeightM=round(frame,2))
      if c.get('roofBand'):fitted['record']['zBaseM']=round(zs-(z0-c.get('elevationOffset',0.)),2)
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
    exposure=wall_exposure(ctx,c,mid,n,zs,main_height)
    # A facing a street camera photographed is seen from the street, whatever the mapped streets say (plazas and wide
    # set-backs put the corridor beyond reach); a touching neighbour still makes a party wall.
    if exposure in ('secondary','gap') and any(np.dot(n,p['normal'])>.95 and (belongs(p,c,main_height,mass)) for p in profiles):exposure='street'
    visible=(length>=3.5 or edge in arc_ids) and exposure in ('street','own-podium','secondary') and ctx['boundary'].boundary.distance(Point(*mid))>.5
    if exposure=='secondary':style=dict(style,bayWidthM=style['bayWidthM']*2,openingRatio=style['openingRatio']*.6)
    if ctx.get('venueField') is not None and ctx['venueField'].distance(Point(*mid))<6.:
     # The edge of a venue's footprint that faces its field is the seating bowl's front: a low plain wall.
     main_height=min(main_height,2.6);visible=False
    mat=style['material'];layer='buildings__'+fid;detail='facades__'+fid
    def block(s,z,w,h,d,material,depth=0,key=detail):
     if w<=.001 or h<=.001:return
     pos=a+u*s+n*depth;cube([pos[0],pos[1],zs+z],[w,d,h],key,material,angle)
    if ring_blank is not None and not c.get('roofBand') and main_height<=ring_blank+1.0:
     # The photographed frontage at this height is a blank wall: so is this unobserved piece of the same height, and
     # the photographed wall's trims (a dentil course, a cap) run round it at the heights they were photographed at.
     if blank_src is not None and (blank_src.get('bands') or blank_src.get('ornaments')):
      tp={'id':blank_src['id']+'-continued','match':blank_src['match'],'normal':n.tolist(),'sourceIds':blank_src.get('sourceIds',[]),'heightRefM':blank_src['heightRefM'],
          'bands':[{k:v for k,v in q.items() if k!='xRange'} for q in blank_src.get('bands',[])],
          'ornaments':[{k:v for k,v in q.items() if k not in ('xRange','xs')} for q in blank_src.get('ornaments',[])]}
      if blank_src.get('materialEvidence'):tp['materialEvidence']=blank_src['materialEvidence']
      else:tp['material']=mat
      trimmed=facade_grammar.compile_wall(ctx,c,a,b,z0,main_height,tp,fits)
      if trimmed:
       rec=trimmed['record'];rec.update(status='inferred',edge=edge,grammarId=None,archetype=style['archetype'],sourceIds=[],unresolved=['blank wall by continuity with the photographed blank frontage of this building at this height; its trims continued'])
       records.append(rec);continue
     block(length/2,main_height/2,length,main_height,.38,mat,-.19,layer)
     records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':0,'columns':0,'archetype':style['archetype'],'material':mat,'sourceIds':[],'status':'inferred','unresolved':['blank wall by continuity with the photographed blank frontage of this building at this height']});continue
    if c.get('roofBand') and main_height<3.5:
     # A roof band lower than a storey is an opaque riser, never an invented window floor. A band a storey or more
     # high is a volume with walls of its own and takes the same facade rules as any wall; blank storey-high bands
     # were the windowless towers.
     block(length/2,main_height/2,length,main_height,.24,mat,-.12,layer);continue
    if not visible:
     block(length/2,main_height/2,length,main_height,.38,mat,-.19,layer)
     records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':0,'columns':0,'archetype':style['archetype'],'material':mat,'sourceIds':[],'status':'blank','exposure':exposure,'unresolved':[],**({'zBaseM':round(zs-(z0-c.get('elevationOffset',0.)),2)} if c.get('roofBand') else {})})
     continue
    cols=1 if edge in arc_ids else max(1,round(length/style['bayWidthM']));bay=length/cols;ground_level=c.get('elevationOffset',0)<.5;gh=min(style['groundHeightM'],main_height*.42) if ground_level else 0
    rows=max(1,round((main_height-gh)/style['floorHeightM']));fh=(main_height-gh)/rows
    wid=bay*style['openingRatio'];opening_h=fh*style['openingHeightRatio'];fw=style['frameWidthM'];projection=style['frameProjectionM'];recess=.18
    records.append({'buildingId':fid,'edge':edge,'midpoint':mid.tolist(),'normal':n.tolist(),'lengthM':round(length,2),'heightM':round(main_height,2),'floors':rows+1,'columns':cols,'archetype':style['archetype'],'material':mat,'sourceIds':obs.get('sourceIds',[]) if obs else [],'exposure':exposure,'status':'visual-fit' if obs else 'inferred','unresolved':obs.get('unresolved',[]) if obs else ['complete facade has no direct image fit; material and openings remain unverified priors'],**({'zBaseM':round(zs-(z0-c.get('elevationOffset',0.)),2)} if c.get('roofBand') else {})})
    for row in range(0 if ground_level else 1,rows+1):
     low=0 if row==0 else gh+(row-1)*fh;rh=gh if row==0 else fh
     if row==0 and exposure=='secondary':
      # A lesser facade has no shopfront: its ground storey is wall (service doors are not invented).
      block(length/2,rh/2,length,rh,.38,mat,-.19,layer);continue
     ow=bay*(.74 if row==0 else style['openingRatio']);oh=rh*(.77 if row==0 else style['openingHeightRatio']);bottom=low+(.15 if row==0 else (rh-oh)*.48);top=bottom+oh
     if row>0 and style.get('ribbon'):
      # Ribbon glazing: one band of glass along the storey between solid spandrels, its lights divided by mullions at
      # the bay joints; wall only at the band's ends. The spandrel's lip is the sill, in the wall's own material.
      e=(bay-ow)/2;span=length-2*e;fullprojection=projection;depth=.18+fullprojection;jw=float(np.clip(bay-ow,.05,.12))
      for x,w_ in [(e/2,e),(length-e/2,e)]:block(x,low+rh/2,w_,rh,.38,mat,-.19,layer)
      block(length/2,(low+bottom)/2,span,bottom-low,.38,mat,-.19,layer);block(length/2,(top+low+rh)/2,span,low+rh-top,.38,mat,-.19,layer)
      for col in range(cols):
       x0_=max(e,col*bay);x1_=min(length-e,(col+1)*bay);variant=(col+row*3)%7
       block((x0_+x1_)/2,(bottom+top)/2,x1_-x0_,oh,.04,['glass','glass_dark','glass_light'][variant%3],-recess)
       for k in range(1,style['mullions']+1):block((col+.5)*bay-ow/2+ow*k/(style['mullions']+1),(bottom+top)/2,.042,oh,.065,'aluminum',-.07)
       if col:block(col*bay,(bottom+top)/2,jw,oh,depth,'aluminum',fullprojection/2)
       if variant in [1,3,5]:ch=oh*(.18+.08*(variant%3));block((x0_+x1_)/2,top-ch/2-fw,x1_-x0_-fw*2,ch,.02,'curtain',-.16)
      for x in [e+fw/2,length-e-fw/2]:block(x,(bottom+top)/2,fw,oh,depth,'aluminum',fullprojection/2)
      for zz in [bottom+fw/2,top-fw/2]:block(length/2,zz,span,fw,depth,'aluminum',fullprojection/2)
      block(length/2,bottom+oh*style['transomRatio'],span,.046,.065,'aluminum',-.07)
      block(length/2,bottom-.05,span+.1,.1,.3,mat,.02,layer)
      components+=cols;continue
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
          p=a+u*(s+dx);cylinder([*(p+n*.03),zs+top+.8],[*(p+n*1.35),zs+top+.2],.023,detail,'steel',6)
      components+=1
    # Continuous base, cornice and coping; stone bands only where observed. A gable end has no cornice at its top.
    if c.get('roofBand'):
     # A tier's wall stands on the roof below and is the same mass as the wall under it: no plinth on that roof and no
     # cornice at each step, only a slim cap of its own material. A concrete base, cornice and coping at every tier made
     # a terraced tile block a stack of boxes with white cornices.
     block(length/2,main_height+.05,length,.1,.44,mat,.03,layer)
    else:
     for zz,hh,dd in ([(0.22,.44,.10)] if gable_planes is not None else [(0.22,.44,.10),(main_height-.22,.15,.15),(main_height+.10,.18,.48)]):block(length/2,zz,length,hh,dd,'concrete',.05)
    if style.get('stoneBands'):
     for zz in [gh-.1,gh+fh*1.2]:block(length/2,zz,length,.32,.20,'concrete',.12)
   if gable_mark is not None:end_gable();gable_mark=None
 return records,components
