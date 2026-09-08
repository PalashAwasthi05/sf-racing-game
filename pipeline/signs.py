"""Deterministic, readable sign decals from inventory legends; no generated text."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import re,math
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

def texture(family,legend):
 im=Image.new('RGB',(256,384),'#e5e7df');d=ImageDraw.Draw(im)
 fontpath='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
 def font(size):
  try:return ImageFont.truetype(fontpath,size)
  except:return ImageFont.load_default(size=size)
 d.rounded_rectangle((6,6,249,377),radius=13,outline='#444a46',width=5)
 if family=='STOP':
  d.rectangle((0,0,256,384),fill='#a52825');lines=['STOP'];col='white';size=62
 elif family=='ONE WAY':lines=['ONE','WAY','→'];col='#182121';size=54
 elif family=='DO NOT ENTER':
  d.ellipse((22,32,234,244),fill='#ae2b26');d.rectangle((44,125,212,155),fill='white');lines=['DO NOT','ENTER'];col='#182121';size=34
 elif family=='SPEED LIMIT':
  number=re.search(r'\b(\d{2})\b',legend);lines=['SPEED','LIMIT',number[1] if number else ''];col='#182121';size=46
 elif family=='NO STOPPING':lines=['NO','STOPPING','ANY','TIME'];col='#a72d28';size=34
 else:lines=['NO','PARKING'];col='#a72d28';size=44
 y=255 if family=='DO NOT ENTER' else max(30,(384-len(lines)*(size+14))/2)
 for line in lines:
  d.text((128,y),line,font=font(size),anchor='mt',fill=col);y+=size+14
 return im

def install(ctx):
 def make(pos,z,angle,family,legend,fid):
  key='signtexture_'+family.replace(' ','_')+'_'+re.sub('[^a-zA-Z0-9]','',legend)[:60]
  if key not in ctx['material_bank'].cache:ctx['material_bank'].cache[key]=PBRMaterial(name=key,baseColorTexture=texture(family,legend),metallicFactor=.12,roughnessFactor=.48,doubleSided=False)
  u=np.array([math.cos(angle),math.sin(angle)]);n=np.array([math.sin(angle),-math.cos(angle)]);w=.5;h=.75
  ctx['cube']([*pos,z],[w,.035,h],'signs','aluminum',angle)
  v=np.array([[*(pos-u*w/2+n*.022),z-h/2],[*(pos+u*w/2+n*.022),z-h/2],[*(pos+u*w/2+n*.022),z+h/2],[*(pos-u*w/2+n*.022),z+h/2]])
  ctx['put'](trimesh.Trimesh(v,[[0,1,2],[0,2,3]],process=False),'signs',key,uv=np.array([[0,0],[1,0],[1,1],[0,1]]))
 return make
