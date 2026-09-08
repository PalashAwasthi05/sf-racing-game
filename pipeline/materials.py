"""Reusable PBR materials; physical UV scale, no image-specific facade assumptions."""
import json
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial
from pathlib import Path

class Materials:
 def __init__(self,root,palette):
  self.root=Path(root);self.palette=palette;self.cache={};self.scales={}
  self.manifest=json.load(open(self.root/'manifest.json'))
  self.scans={k:v for k,v in zip(['brick','asphalt','concrete'],self.manifest['materials'])}
 def get(self,key):
  aliases={'brick2':'brick','stone':'concrete','sidewalk':'concrete','frame':'aluminum','rail':'steel','metal':'steel','dark':'recess','paint':'white','roof':'roofing'}
  key=aliases.get(key,key)
  if key in self.cache:return self.cache[key],self.scales.get(key,(2.,2.))
  if key in self.scans:
   a=self.scans[key];color=Image.open(self.root/a['maps']['color']['path']);normal=Image.open(self.root/a['maps']['normal_gl']['path']);r=Image.open(self.root/a['maps']['roughness']['path']).convert('L');rr=np.asarray(r);mr=np.zeros((*rr.shape,3),np.uint8);mr[:,:,1]=rr
   m=PBRMaterial(name=key,baseColorTexture=color,normalTexture=normal,metallicRoughnessTexture=Image.fromarray(mr),metallicFactor=0.,roughnessFactor=1.,baseColorFactor=[255,255,255,255]);self.scales[key]=a['tile_size_m']
  else:
   colors={'aluminum':'#b9c0c3','steel':'#51595c','recess':'#171d20','glass':'#526975','glass_dark':'#263e49','glass_light':'#75858b','curtain':'#aaa698','roofing':'#696d6a','white':'#dfdfd5','joint':'#686967','soil':'#494536','redpaint':'#983d36','greenpaint':'#477d6b','bark':'#807163','leaf1':'#547b38','leaf2':'#366024','leaf3':'#729249'}
   c=colors.get(key,self.palette.get(key,key));c=c.lstrip('#');rgba=[int(c[i:i+2],16) for i in (0,2,4)]+[255]
   rough=.9;metal=0.
   if key.startswith('glass'):rough=.16;metal=.22
   if key in ['aluminum','steel']:rough=.34;metal=.75
   m=PBRMaterial(name=key,baseColorFactor=rgba,metallicFactor=metal,roughnessFactor=rough,doubleSided=key.startswith('leaf'))
  self.cache[key]=m;return m,self.scales.get(key,(2.,2.))
