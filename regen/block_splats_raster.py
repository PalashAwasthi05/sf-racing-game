"""Small CPU tangent Gaussian rasterizer and robust coordinate-descent appearance fit."""
import ctypes
import hashlib
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
from block_splats_pose import basis, perspective


# A real front-to-back elliptical Gaussian rasterizer. Accumulate rendered residuals
# through the very same per-pixel transmittance weights to solve for RGB coefficients.
_SOURCE=r"""
#include <algorithm>
#include <cmath>
#include <vector>
struct Disc { int idx; float depth,x,y,a,b,c,opacity; int x0,x1,y0,y1; };
extern "C" void raster(int n,int width,int height,float f,const float* eye,const float* view,
                       const float* points,const float* rgb,const float* target,float* predicted,
                       float* gradient,float* denom,int fit) {
    std::vector<Disc> discs;
    for(int i=0;i<n;i++) {
        const float* p=points+i*10;
        float cen[3]={},u[3]={},v[3]={};
        for(int d=0;d<3;d++) for(int k=0;k<3;k++){
            cen[d]+=view[d*3+k]*(p[k]-eye[k]);
            u[d]+=view[d*3+k]*p[3+k]; v[d]+=view[d*3+k]*p[6+k];
        }
        if(cen[2]<.7f||cen[2]>85.f)continue;
        float inv=1.f/cen[2],xx=cen[0]*inv,yy=cen[1]*inv;
        float x=(width-1)*.5f+f*xx,y=(height-1)*.5f+f*yy;
        float ux=f*inv*(u[0]-xx*u[2]),uy=f*inv*(u[1]-yy*u[2]);
        float vx=f*inv*(v[0]-xx*v[2]),vy=f*inv*(v[1]-yy*v[2]);
        float a=ux*ux+vx*vx+.09f,b=ux*uy+vx*vy,c=uy*uy+vy*vy+.09f,det=a*c-b*b;
        float rx=std::min(60.f,3.f*std::sqrt(a)),ry=std::min(60.f,3.f*std::sqrt(c));
        if(x+rx<0||x-rx>=width||y+ry<0||y-ry>=height||det<1e-6f)continue;
        discs.push_back({i,cen[2],x,y,c/det,-b/det,a/det,p[9],std::max(0,int(std::floor(x-rx))),
                         std::min(width-1,int(std::ceil(x+rx))),std::max(0,int(std::floor(y-ry))),
                         std::min(height-1,int(std::ceil(y+ry)))});
    }
    std::stable_sort(discs.begin(),discs.end(),[](const Disc& a,const Disc& b){return a.depth<b.depth;});
    std::vector<float> trans(width*height,1.f);
    for(int pass=0;pass<(fit?2:1);pass++){
        std::fill(trans.begin(),trans.end(),1.f);
        for(const auto& s:discs) for(int y=s.y0;y<=s.y1;y++) for(int x=s.x0;x<=s.x1;x++){
            int px=y*width+x; float t=trans[px];
            if(t<.004f)continue;
            float dx=x-s.x,dy=y-s.y;
            float r2=s.a*dx*dx+2*s.b*dx*dy+s.c*dy*dy;
            if(r2>9.f)continue;
            float a=s.opacity*std::exp(-.5f*r2),weight=t*a;
            trans[px]=t*(1.f-a);
            if(!pass){
                for(int d=0;d<3;d++) predicted[4*px+d]+=weight*rgb[3*s.idx+d];
                predicted[4*px+3]+=weight;
            }else {
                float cover=predicted[4*px+3];
                if(cover<.72f)continue;
                float error[3],mean=0.f;
                for(int d=0;d<3;d++){
                    error[d]=target[3*px+d]-predicted[4*px+d]/cover;
                    mean+=std::abs(error[d])/3.f;
                }
                float robust=std::min(1.f,.09f/std::max(.0001f,mean));
                float coeff=weight/cover*robust;
                for(int d=0;d<3;d++) gradient[3*s.idx+d]+=coeff*error[d];
                denom[s.idx]+=coeff;
            }
        }
    }
}
"""


def native():
    temp=Path(tempfile.gettempdir())/('block-splats-raster-'+hashlib.sha256(_SOURCE.encode()).hexdigest()[:16])
    temp.mkdir(exist_ok=True)
    lib=temp/'raster.dylib'
    if not lib.exists():
        source=temp/'raster.cpp'
        source.write_text(_SOURCE)
        subprocess.run(['clang++','-std=c++17','-O3','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    fn=ctypes.CDLL(str(lib)).raster
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ptr]*8+[ctypes.c_int]
    fn.restype=None
    return fn


def images(cameras,along):
    width,height,focal=466,320,205.
    n=np.array([-along[1],along[0],0.])
    records=[]
    for c in cameras:
        for side in (-1,1):
            view=basis(n*side).astype('f')
            rgb=perspective(c,view,width,height,focal)[...,::-1].astype('f')/255.
            records.append((c['id'],side,np.array(c['eye'],'f'),view,focal,np.ascontiguousarray(rgb)))
    return records


def run_views(fn,surf,rgb,records,fit=False,preview=False):
    gradient=np.zeros_like(rgb);denom=np.zeros(len(rgb),'f')
    n=0;total=0;sse=0.;absolute=0.;panels=[];by_camera=[]
    for camera,side,eye,view,focal,target in records:
        h,w=target.shape[:2]
        pred=np.zeros((h,w,4),'f')
        fn(len(rgb),w,h,focal,eye,view,surf,rgb,target,pred,gradient,denom,int(fit))
        mask=pred[...,3]>=.72
        norm=pred[...,:3]/np.maximum(pred[...,3:],1.e-6)
        err=target[mask]-norm[mask]
        sse+=float(np.sum(err.astype(float)**2));absolute+=float(np.sum(abs(err.astype(float))))
        n+=int(mask.sum());total+=w*h
        by_camera.append({'id':camera,'side':side,'coverage':round(float(mask.mean()),4),
                          'maeSrgb255':round(float(abs(err).mean()*255),2) if len(err) else None})
        if preview and len(panels)<50:
            rendered=np.where(mask[...,None],norm,.105)
            panels.append((int(mask.sum()),camera,side,np.uint8(np.clip(np.hstack([target,rendered]),0,1)*255)))
    metrics={'pixels':n,'totalPixels':total,'coverage':n/max(total,1),'alphaMinimum':.72,
             'maeSrgb255':absolute/max(n*3,1)*255,'psnrDb':-10*np.log10(max(sse/max(n*3,1),1.e-12)),'perView':by_camera}
    return metrics,gradient/np.maximum(denom[:,None],1.e-6),panels


def train(models,cameras,along,heldout_fn,out):
    xyz,u,v,rgb,opacity=models
    surf=np.ascontiguousarray(np.column_stack([xyz,u,v,opacity]),dtype='f')
    fn=native()
    records=images(cameras,along)
    history=[]
    for epoch in range(5):
        metrics,grad,_=run_views(fn,surf,rgb,records,fit=epoch<4)
        history.append({k:v for k,v in metrics.items() if k!='perView'})
        print('RGB fit epoch',epoch,'MAE',metrics['maeSrgb255'],'cover',metrics['coverage'],flush=True)
        if epoch<4:
            rgb=np.ascontiguousarray(np.clip(rgb+.86*grad,0,1),dtype='f')
    # The fitted model is frozen before any independent-drive image is decoded.
    del records
    metrics['history']=history
    hold_records=images(heldout_fn(),along)
    held,_,panels=run_views(fn,surf,rgb,hold_records,preview=True)
    for side in (-1,1):
        candidates=[p for p in panels if p[2]==side]
        if candidates:
            _,identity,_,im=max(candidates,key=lambda p:p[0])
            cv2.imwrite(str(out/f'splat-heldout-side-{side}.jpg'),im[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,91])
    return rgb,metrics,held
