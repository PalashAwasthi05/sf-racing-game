"""Shared dense-leaf finish for source-identified shrubs and clipped hedges.

The profile supplies the foliage colour and occupied volume. The small leaf
structure is a reusable appearance prior, not an observation of the species.
"""
from functools import lru_cache
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial


def leaf_key(colour):
    if isinstance(colour,str) and len(colour)==7 and colour.startswith('#'):
        try:int(colour[1:],16)
        except ValueError:return colour
        return 'groundcover'+colour.lower()
    return colour


@lru_cache(maxsize=1)
def maps():
    # A toroidal 0.72m tile of overlapping leaves. Reuse the same colour and
    # tangent-space maps across all observed tints (one image of each per GLB).
    rng=np.random.default_rng(209);n=256
    light=np.full((n,n),.42);relief=np.zeros((n,n))
    for layer in range(4):
        for _ in range(38):
            cx,cy=rng.uniform(0,n,2);base=rng.uniform(0,2*np.pi)
            for i in range(7):
                a=base+(1 if i%2 else -1)*rng.uniform(.4,1.3);s=(i//2)*4.;x=cx+s*np.cos(base)+3.5*np.cos(a);y=cy+s*np.sin(base)+3.5*np.sin(a)
                length=rng.uniform(4.4,10.);width=length*rng.uniform(.29,.54);rr=int(np.ceil(length+1));iy,ix=np.mgrid[-rr:rr+1,-rr:rr+1];X=ix+np.floor(x)-x;Y=iy+np.floor(y)-y
                u=(X*np.cos(a)+Y*np.sin(a))/length;v=(-X*np.sin(a)+Y*np.cos(a))/width
                mask=u*u+np.abs(v)**1.6<1;yy=(int(np.floor(y))+iy[mask])%n;xx=(int(np.floor(x))+ix[mask])%n
                arch=np.maximum(0.,1-u[mask]**2-np.abs(v[mask]));tone=rng.uniform(.68,1.)*(.72+layer*.08)
                light[yy,xx]=np.clip(tone+.08*arch-.07*v[mask],.35,1);relief[yy,xx]=(.18*layer+.15*arch+.07*v[mask])
    nx=(np.roll(relief,1,1)-np.roll(relief,-1,1))*.45;ny=(np.roll(relief,-1,0)-np.roll(relief,1,0))*.45
    norm=np.stack([nx,ny,np.ones_like(nx)],axis=2);norm/=np.linalg.norm(norm,axis=2,keepdims=True)
    albedo=Image.fromarray(np.uint8(np.rint(light*255))).convert('RGB');normal=Image.fromarray(np.uint8(np.rint((norm*.5+.5)*255)))
    return albedo,normal


def material(key):
    rgb=np.frombuffer(bytes.fromhex(key.split('#',1)[1]),dtype=np.uint8).astype(float)/255
    linear=np.where(rgb<.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    c,n=maps()
    return PBRMaterial(name=key,baseColorTexture=c,normalTexture=n,baseColorFactor=np.r_[np.minimum(1.,linear/mean_light()),1.],metallicFactor=0.,roughnessFactor=.84),(.72,.72)


@lru_cache(maxsize=1)
def mean_light():
    v=np.asarray(maps()[0],float)[:,:,0]/255.
    return float(np.mean(np.where(v<.04045,v/12.92,((v+.055)/1.055)**2.4)))


def emit(ctx,mesh,colour):
    # Smooth only tagged foliage. Hard edges and physically scaled architectural
    # UVs on the adjoining building are unaffected. UVs use the original mesh
    # position, not separately rotated projection axes on every triangle.
    mesh.metadata['smoothLeafSurface']=True
    ctx['put'](mesh,ctx['detail'],leaf_key(colour))
