import trimesh,numpy as np
s=trimesh.load('work/coverage-section-final-v12/candidate.glb',force='scene',process=False)
o=np.array([-45.,-170,85]);f=np.array([0.,21,-70]);f/=np.linalg.norm(f);right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,f)
for x,y in [(274,380),(258,397),(245,410),(232,424),(219,439),(208,451)]:
 d=f+right*(2*x/530-1)*(530/773)*np.tan(np.radians(29))+up*(1-2*y/773)*np.tan(np.radians(29));d/=np.linalg.norm(d);hits=[]
 for name,m in s.geometry.items():
  tri=m.triangles;e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h);valid=np.abs(det)>1e-8;inv=np.zeros_like(det);inv[valid]=1/det[valid];ss=o-tri[:,0];u=inv*np.einsum('ij,ij->i',ss,h);q=np.cross(ss,e1);v=inv*(q@d);t=inv*np.einsum('ij,ij->i',e2,q);ids=np.where(valid&(u>=0)&(v>=0)&(u+v<=1)&(t>0))[0]
  for i in ids: hits.append((t[i],name,int(i),m.face_normals[i].tolist()))
 print((x,y),[(round(t,3),name,i,np.round(o+t*d,3).tolist(),np.round(n,3).tolist()) for t,name,i,n in sorted(hits)[:3]])
