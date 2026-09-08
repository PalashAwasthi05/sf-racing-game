"""Artifact geometry inspection; orthographic CPU preview, not a browser screenshot."""
from pathlib import Path
import numpy as np, trimesh, json
from PIL import Image, ImageDraw
root=Path(__file__).parent.parent
s=trimesh.load(root/'outputs/king-street-pilot.glb',force='scene')
meshes=[]
for node in s.graph.nodes_geometry:
 T,name=s.graph[node];g=s.geometry[name].copy();g.apply_transform(T);meshes.append(g)
m=trimesh.util.concatenate(meshes)
assert np.isfinite(m.vertices).all()
assert m.faces.min()>=0 and m.faces.max()<len(m.vertices)
f=m.faces;v=m.vertices
look=np.array([-.58,-.56,-.59]);look/=np.linalg.norm(look)
right=np.cross(look,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,look)
p=np.c_[v@right,v@up,v@look]
tri=p[f];norm=m.face_normals;visible=norm@(-look)>.001
idx=np.flatnonzero(visible);idx=idx[np.argsort(tri[idx,:,2].mean(axis=1))[::-1]]
colors=m.visual.vertex_colors[f][:,:,:3].mean(axis=1)
light=np.array([-.45,.8,.4]);light/=np.linalg.norm(light)
shade=.57+.43*np.maximum(norm@light,0);colors=np.clip(colors*shade[:,None],0,255).astype('uint8')
W,H=1440,1100;scale=min((W-100)/np.ptp(p[:,0]),(H-100)/np.ptp(p[:,1]));pos=p[:,:2].copy();pos[:,0]=(pos[:,0]-(p[:,0].min()+p[:,0].max())/2)*scale+W/2;pos[:,1]=H/2-(pos[:,1]-(p[:,1].min()+p[:,1].max())/2)*scale
pixels=np.full((H,W,3),[189,207,214],dtype=np.uint8);depth=np.full((H,W),np.inf)
for i in idx:
 t=pos[f[i]];z=tri[i,:,2]
 xmin=max(0,int(np.floor(t[:,0].min())));xmax=min(W-1,int(np.ceil(t[:,0].max())));ymin=max(0,int(np.floor(t[:,1].min())));ymax=min(H-1,int(np.ceil(t[:,1].max())))
 if xmax<xmin or ymax<ymin:continue
 a,b,c=t;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
 if abs(den)<.001:continue
 xx,yy=np.meshgrid(np.arange(xmin,xmax+1)+.5,np.arange(ymin,ymax+1)+.5)
 u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den;vv=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den;ww=1-u-vv
 zz=u*z[0]+vv*z[1]+ww*z[2];window=depth[ymin:ymax+1,xmin:xmax+1];take=(u>=0)&(vv>=0)&(ww>=0)&(zz<window)
 window[take]=zz[take];pixels[ymin:ymax+1,xmin:xmax+1][take]=colors[i]
Image.fromarray(pixels).save(root/'work/model-inspection.png')
summary={'meshes':len(s.geometry),'vertices':len(v),'triangles':len(f),'bounds':s.bounds.tolist(),'finiteGeometry':True,'validIndices':True,'modelBytes':(root/'outputs/king-street-pilot.glb').stat().st_size}
(root/'outputs/king-street-validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
