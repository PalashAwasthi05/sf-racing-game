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
  if not matches(c,a):continue
  if not a.get('sourceIds') or not a.get('text'):raise ValueError('Lettering requires literal observed text and source IDs')
  r=next((r for r in resolved if r['profile']['id']==a.get('profileId')),None)
  f=r['frame'] if r else frame(c,a);u,n=f['tangent'],f['normal'];depth=a.get('depthM',.1);key='facades__'+c['id']+'__observed_lettering'
  center=f['left']+f['width']*a['centerU'];z=a['bottomM'];width=a['widthM'];height=a['heightM'];thick=a.get('thicknessM',.035)
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
  records.append({**a,'buildingId':c['id'],'placementStatus':'image-estimated','fontStatus':'approximate letterforms','worldCenter':point(0,height/2,depth).tolist(),'normal':n.tolist(),'independentValidation':False})
 return records
