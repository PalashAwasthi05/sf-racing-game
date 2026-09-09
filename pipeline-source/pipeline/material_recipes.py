"""Deterministic metric appearance recipes, not scans of a particular building.
Small joints and shallow relief belong to a material family; no baked lighting.
"""
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial

RECIPES={
 'render_salmon':{'scale':[2,2],'color':[182,123,107],'rough':.82,'noise':.8},
 'render_gray':{'scale':[2,2],'color':[163,166,161],'rough':.87,'noise':.8},
 'render_white':{'scale':[2,2],'color':[218,216,204],'rough':.87,'noise':.8},
 'timber_dark':{'scale':[2.4,1.2],'cell':[2.4,.15],'joint':.003,'color':[60,69,75],'mortar':[38,44,47],'relief':.001,'rough':.84,'variation':2},
 'timber_tan':{'scale':[2.4,1.2],'cell':[.10,1.2],'joint':.003,'color':[151,128,96],'mortar':[94,78,56],'relief':.001,'rough':.85,'variation':2},
 'brick_modular':{'scale':[1.8,.6],'cell':[.225,.075],'joint':.006,'color':[145, 70,49],'mortar':[139,128,108],'relief':.0018,'variation':5,'rough':.82},
 'brick_dark':{'scale':[1.8,.6],'cell':[.225,.075],'joint':.006,'color':[112, 65,48],'mortar':[120,112,98],'relief':.0018,'variation':4,'rough':.85},
 'terracotta_panel':{'scale':[2.4,1.2],'cell':[.6,.3],'joint':.004,'color':[174,92, 61],'mortar':[ 90,66, 50],'relief':.001,'variation':2,'rough':.67},
 'terracotta_ribbed':{'scale':[2.4,1.2],'cell':[.6,.3],'joint':.004,'color':[170,80, 50],'mortar':[ 90,61,45],'relief':.001,'variation':2,'rough':.72,'ribs':.025},
 'stone':{'scale':[2.4,1.2],'color':[199,193,177],'rough':.8,'noise':.9},
 'limestone':{'scale':[2.4,1.2],'color':[199,193,177],'rough':.8,'noise':.9},
 'stucco':{'scale':[2.,2.],'color':[201,198,182],'rough':.9,'noise':1.3},
 'stucco_red':{'scale':[2.,2.],'color':[142,80,65],'rough':.9,'noise':1.3},
 'panel_dark':{'scale':[2.4,1.2],'cell':[1.2,.3],'joint':.003,'color':[55,62,64],'mortar':[36,41,43],'relief':.0006,'variation':1,'rough':.74},
 'panel_silver':{'scale':[2.4,1.2],'cell':[1.2,.6],'joint':.003,'color':[176,181,177],'mortar':[117,123,121],'relief':.0005,'variation':1,'rough':.37,'metal':.55}
}

def create(key):
 a=RECIPES[key];rng=np.random.default_rng(710);n=512;w,h=a['scale'];yy,xx=np.mgrid[0:n,0:n];x=xx*w/n;y=yy*h/n
 color=np.broadcast_to(np.array(a['color'],float),(n,n,3)).copy();relief=np.zeros((n,n));noise=rng.normal(0,a.get('noise',1.1),(n,n));color+=noise[:,:,None]
 if 'cell' in a:
  cw,ch=a['cell'];row=np.floor(y/ch).astype(int);offset=(row%2)*cw/2 if key.startswith('brick') else 0
  dx=np.mod(x+offset,cw);dy=np.mod(y,ch);j=a['joint'];seam=(dx<j)|(dy<j)
  nr=int(round(h/ch));nc=int(round(w/cw));variation=rng.normal(0,a.get('variation',2),(nr+1,nc+2))
  color+=variation[np.minimum(row,nr),np.floor((x+offset)/cw).astype(int)][:,:,None]
  color[seam]=a['mortar'];relief[~seam]=a['relief']
 if 'ribs' in a:relief+=.00035*np.sin(y/a['ribs']*2*np.pi)
 gy,gx=np.gradient(relief,h/n,w/n);normal=np.stack([-gx,-gy,np.ones_like(gx)],axis=2);normal/=np.linalg.norm(normal,axis=2,keepdims=True)
 mr=np.zeros((n,n,3),np.uint8);mr[:,:,1]=np.clip(a['rough']*255+noise,0,255);mr[:,:,2]=int(a.get('metal',0)*255)
 mat=PBRMaterial(name=key,baseColorTexture=Image.fromarray(np.clip(color,0,255).astype('uint8')),normalTexture=Image.fromarray(np.clip((normal*.5+.5)*255,0,255).astype('uint8')),metallicRoughnessTexture=Image.fromarray(mr),baseColorFactor=[255,255,255,255],roughnessFactor=1.,metallicFactor=1.)
 return mat,a['scale']
