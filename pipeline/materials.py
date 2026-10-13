"""Reusable PBR materials; physical UV scale, no image-specific facade assumptions."""
import json
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial
from pathlib import Path
from . import material_recipes

class Materials:
 def __init__(self,root,palette):
  self.root=Path(root);self.palette=palette;self.cache={};self.scales={}
  self.manifest=json.load(open(self.root/'manifest.json'))
  self.scans={k:v for k,v in zip(['brick','asphalt','concrete'],self.manifest['materials'])}
 def get(self,key):
  aliases={'brick2':'brick','sidewalk':'concrete','frame':'aluminum','rail':'steel','metal':'steel','dark':'recess','paint':'white','roof':'roofing'}
  key=aliases.get(key,key)
  if key in self.cache:return self.cache[key],self.scales.get(key,(2.,2.))
  if '#' in key and key not in material_recipes.RECIPES:
   # 'recipe#rrggbb': the recipe's structure (joints, relief, roughness) with an observed colour.
   base,hexcol=key.split('#',1)
   if base in material_recipes.RECIPES and len(hexcol)==6:material_recipes.RECIPES[key]={**material_recipes.RECIPES[base],'color':[int(hexcol[i:i+2],16) for i in (0,2,4)]}
  if key in material_recipes.RECIPES:
   m,scale=material_recipes.create(key);self.scales[key]=scale
  elif key in self.scans:
   a=self.scans[key];color=Image.open(self.root/a['maps']['color']['path']);normal=Image.open(self.root/a['maps']['normal_gl']['path']);r=Image.open(self.root/a['maps']['roughness']['path']).convert('L');rr=np.asarray(r);mr=np.zeros((*rr.shape,3),np.uint8);mr[:,:,1]=rr
   m=PBRMaterial(name=key,baseColorTexture=color,normalTexture=normal,metallicRoughnessTexture=Image.fromarray(mr),metallicFactor=0.,roughnessFactor=.62 if key=='asphalt' else 1.,baseColorFactor=[255,255,255,255]);self.scales[key]=a['tile_size_m']   # asphalt keeps a sheen: a wearing course is smoother than its roughness scan alone suggests
  else:
   colors={'padding':'#1f392b','leather':'#6c4529','bottle':'#3f6a55','screen':'#0d1216','limestone':'#c3c1b5','copper':'#73452f','verdigris':'#294342','lamp':'#c5cbc5','glass_blue':'#23313a','glass_clear':'#2c3538','silver':'#aab1b1','charcoal':'#2d3335','zinc':'#4c5359','stucco':'#c2c4bf','terracotta':'#b36f49','interior':'#111719','aluminum':'#6f777a','steel':'#3f4548','recess':'#171d20','glass':'#1e2528','glass_dark':'#141a1d','glass_light':'#3a4448','curtain':'#aaa698','roofing':'#696d6a','white':'#dfdfd5','joint':'#686967','soil':'#494536','redpaint':'#983d36','greenpaint':'#477d6b','bark':'#807163','frond':'#4a6a34','thatch':'#5b4a35','leaf1':'#547b38','leaf2':'#366024','leaf3':'#729249'}
   c=colors.get(key,self.palette.get(key,key));c=c.lstrip('#');rgba=[int(c[i:i+2],16) for i in (0,2,4)]+[255]
   # Finish by the kind of surface the class names (not by place): glazing is smooth and reflective, metals are
   # metallic with a light satin, paint has a sheen, masonry, render and interiors stay matte.
   finish={'glass':(.06,0.),'aluminum':(.5,.35),'steel':(.55,.3),'silver':(.45,.45),'zinc':(.42,.8),'copper':(.4,.75),'charcoal':(.62,.05),'curtain':(.35,.25),'verdigris':(.7,.15),'lamp':(.3,.5),'white':(.55,0.),'redpaint':(.55,0.),'greenpaint':(.55,0.),'yellow':(.55,0.),'terracotta':(.6,0.),'water':(.05,0.)}   # roofing keeps its default: a v13 observation identifies that source material by fingerprint
   rough,metal=finish.get('glass' if key.startswith('glass') else key,(.9,0.))
   rgb=np.array(rgba[:3],float)/255;linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);rgba=np.r_[linear,1.]
   m=PBRMaterial(name=key,baseColorFactor=rgba,metallicFactor=metal,roughnessFactor=rough,doubleSided=key.startswith('leaf'))
  self.cache[key]=m;return m,self.scales.get(key,(2.,2.))
