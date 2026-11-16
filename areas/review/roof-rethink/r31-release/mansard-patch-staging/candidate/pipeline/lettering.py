"""Observed literal text -> reusable raised-letter geometry on a facade plane.
No source-image pixels, invented tenants or per-address rendering branches.
Fonts approximate the photographed letterforms; placement remains an image estimate.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from .envelopes import matches,frame,_polys
from .freshness import sign_allowed

def attachment(a,ctx,c,resolved):
 """Resolve text from its supporting component, using the compiler's dimensions."""
 anchor=a.get('attachment')
 if not anchor:raise ValueError('Missing semantic lettering attachment: '+a['id'])
 z=a['bottomM'];center=a['centerU'];depth=a.get('depthM',.1);kind=anchor['kind']
 carrier=a.get('carrierEvidence')
 if carrier:
  if not carrier.get('sourceIds'):raise ValueError('Sign carrier evidence requires a source')
  kind={'opaque-surface':'envelope-surface','glass':'glazing','entrance-canopy':'opening-canopy','freestanding-canopy':'generated-canopy'}[carrier['type']]
 if kind=='opening-canopy':
  profile=next(p for p in ctx['PROFILES'] if p['id']==anchor['profileId'])
  op=next(o for o in profile['openings'] if o.get('id')==anchor['openingId'])
  rec=next(r for r in ctx['facade_records'] if r['buildingId']==c['id'] and r.get('grammarId')==profile['id'])
  # This is the same canopy slab center/thickness used by compile_wall.
  z=op['rect'][3]*rec['heightM']+.19;center=(op['rect'][0]+op['rect'][2])/2
 elif kind in ['envelope-surface','glazing']:
  r=next(r for r in resolved if r['profile']['id']==a['profileId'])
  node=next(n for n in r['nodes'] if n['id']==anchor['nodeId']);depth=node['depthM']
  if kind=='glazing':depth-=node['grids'][anchor.get('gridIndex',0)]['style'].get('recessM',.16)
  depth+=.025
 elif kind=='generated-canopy':
  if not a.get('canopy'):raise ValueError('Missing support canopy')
  # Top of canopy is tied to this baseline by construction.
 else:raise ValueError('Unsupported sign attachment '+kind)
 return z,center,depth,{'rule':'sign-attachment','kind':kind,'beforeBottomM':a['bottomM'],'resolvedBottomM':z,'beforeCenterU':a['centerU'],'resolvedCenterU':center,'beforeDepthM':a.get('depthM',.1),'resolvedDepthM':depth,'changed':abs(z-a['bottomM'])+abs(center-a['centerU'])+abs(depth-a.get('depthM',.1))>.001}

@lru_cache(maxsize=128)
def contours(text,font='Arial Narrow Bold.ttf'):
 path=Path('/System/Library/Fonts/Supplemental')/font
 f=ImageFont.truetype(str(path),128) if path.exists() else ImageFont.load_default(size=128)
 bb=f.getbbox(text);w=bb[2]-bb[0];h=bb[3]-bb[1]
 im=Image.new('L',(w+4,h+4));ImageDraw.Draw(im).text((2-bb[0],2-bb[1]),text,font=f,fill=255)
 mask=np.asarray(im)>127;parts=[shape(g).simplify(.45,preserve_topology=True) for g,v in shapes(mask.astype('uint8'),mask=mask)]
 return unary_union(parts),w+4,h+4

def compile(ctx,c,z0,observations,resolved):
 records=[]
 for a in observations:
  if not matches(c,a) or not sign_allowed(a):continue
  if not a.get('sourceIds') or not a.get('text'):raise ValueError('Lettering requires literal observed text and source IDs')
  r=next((r for r in resolved if r['profile']['id']==a.get('profileId')),None)
  f=r['frame'] if r else frame(c,a);u,n=f['tangent'],f['normal'];key='facades__'+c['id']+'__observed_lettering'
  # a part of a split outline draws only the lettering on its own stretch of the frontage
  from .envelopes import span_clip
  z,center_u,depth,check=attachment(a,ctx,c,resolved)
  center=f['left']+f['width']*center_u;width=a['widthM'];height=a['heightM'];thick=a.get('thicknessM',.035)
  if f.get('span') and not (f['span'][0]-.5<=center<=f['span'][1]+.5):continue   # the sign is on another part's stretch
  clip=span_clip(f)
  if clip and ctx.get('set_clip'):ctx['set_clip'](clip)
  def point(x,y,d):return np.r_[u*(center+x)+n*(f['plane']+d),z0+z+y]
  textlines=a['text'].split('\n');lineheight=height/len(textlines)
  for i,line in enumerate(textlines):
   geometry,ww,hh=contours(line,a.get('font','Arial Narrow Bold.ttf'))
   # Preserve letter proportions unless an observed width explicitly requests fit.
   scale=min(width/ww,lineheight*.88/hh);gw=ww*scale;gh=hh*scale
   for poly in _polys(geometry):
    mesh=trimesh.creation.extrude_polygon(poly,height=thick,engine='earcut');v=mesh.vertices.copy();xx=(v[:,0]-ww/2)*scale;yy=(hh-v[:,1])*scale+(len(textlines)-1-i)*lineheight
    if a.get('archM'):yy+=a['archM']*(1-(xx/(gw/2))**2)
    mesh.vertices=np.array([point(x,y,depth+d) for x,y,d in zip(xx,yy,v[:,2])]);mesh.fix_normals();ctx['put'](mesh,key,a.get('material','white'))
  if a.get('canopy'):
   ca=a['canopy'];p=u*center+n*(f['plane']+ca.get('projectionM',1.4)/2)
   ctx['cube']([*p,z0+z-.12],[width+.9,ca.get('projectionM',1.4),.18],key,ca.get('material','panel_dark'),np.arctan2(u[1],u[0]))
  if clip and ctx.get('set_clip'):ctx['set_clip'](None)
  records.append({**a,'buildingId':c['id'],'attachmentCheck':check,'placementStatus':'image-estimated; support-constrained','fontStatus':'approximate letterforms','worldCenter':point(0,height/2,depth).tolist(),'normal':n.tolist(),'independentValidation':False})
 return records
