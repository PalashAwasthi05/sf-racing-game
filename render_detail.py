"""CPU inspection of exported geometry and texture UVs. Not a browser screenshot."""
import numpy as np,trimesh,json,time
from PIL import Image
from pathlib import Path
ROOT=Path(__file__).parent.parent
s=trimesh.load(ROOT/'outputs/king-street-pilot.glb',force='scene');W,H=1200,1000
camera=np.array([45.,35.,65.]);target=np.array([0.,14.,0.]);forward=(target-camera);forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward)
pixels=np.full((H,W,3),[178,198,212],np.uint8);depth=np.full((H,W),np.inf);scale=15.;total=0
light=np.array([-.5,.8,.4]);light/=np.linalg.norm(light)
for node in s.graph.nodes_geometry:
 T,name=s.graph[node];m=s.geometry[name];v=trimesh.transform_points(m.vertices,T);n=m.face_normals@T[:3,:3].T;faces=m.faces;p=np.c_[(v-target)@right,(v-target)@up,(v-camera)@forward];pos=np.c_[p[:,0]*scale+W/2,H/2-p[:,1]*scale];tri=pos[faces];front=n@(-forward)>.001;inside=(tri[:,:,0].max(axis=1)>0)&(tri[:,:,0].min(axis=1)<W)&(tri[:,:,1].max(axis=1)>0)&(tri[:,:,1].min(axis=1)<H);indices=np.flatnonzero(front&inside)
 if not len(indices):continue
 mat=m.visual.material;factor=np.array(mat.baseColorFactor if mat.baseColorFactor is not None else [255]*4)[:3]/255.;tex=np.array(mat.baseColorTexture.convert('RGB')) if mat.baseColorTexture is not None else None;uv=m.visual.uv
 for i in indices:
  a,b,c=tri[i];xmin=max(0,int(np.floor(min(a[0],b[0],c[0]))));xmax=min(W-1,int(np.ceil(max(a[0],b[0],c[0]))));ymin=max(0,int(np.floor(min(a[1],b[1],c[1]))));ymax=min(H-1,int(np.ceil(max(a[1],b[1],c[1]))));den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
  if abs(den)<.002 or xmax<xmin or ymax<ymin:continue
  xx,yy=np.meshgrid(np.arange(xmin,xmax+1)+.5,np.arange(ymin,ymax+1)+.5);u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den;vv=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;ww=1-u-vv;z=p[faces[i],2];zz=u*z[0]+vv*z[1]+ww*z[2];window=depth[ymin:ymax+1,xmin:xmax+1];take=(u>=0)&(vv>=0)&(ww>=0)&(zz<window)
  if not take.any():continue
  shade=.56+.44*max(0,n[i]@light)
  if tex is not None:
   tuv=uv[faces[i]];uu=(u[take]*tuv[0,0]+vv[take]*tuv[1,0]+ww[take]*tuv[2,0])%1;tv=(u[take]*tuv[0,1]+vv[take]*tuv[1,1]+ww[take]*tuv[2,1])%1;colors=tex[((1-tv)*(len(tex)-1)).astype(int),(uu*(tex.shape[1]-1)).astype(int)]*factor*shade
  else:colors=255*factor*shade
  pixels[ymin:ymax+1,xmin:xmax+1][take]=np.clip(colors,0,255).astype(np.uint8);window[take]=zz[take];total+=1
Image.fromarray(pixels).save(ROOT/'work/detail-inspection.png');print('Rasterized triangles',total)
