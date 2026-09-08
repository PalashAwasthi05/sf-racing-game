"""Image landmarks -> rectified, fitted repetition grammar. No per-building code.
A vision pass supplies wall-region corners and sparse opening centers, once.
The numeric fit chooses repetitions, completes masked cells and reports holdouts.
"""
import json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).parent

def homography(src,dst):
 A=[];b=[]
 for (x,y),(u,v) in zip(src,dst):
  A.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]]);b.extend([u,v])
 h=np.linalg.solve(np.asarray(A,float),np.asarray(b,float));return np.r_[h,1].reshape(3,3)

def project(h,points):
 p=np.c_[np.asarray(points,float),np.ones(len(points))]@h.T
 return p[:,:2]/p[:,2:]

def fit_axis(values,maximum=20):
 values=np.asarray(values);candidates=[]
 for n in range(1,maximum+1):
  def residual(theta):
   centers=theta[0]+np.arange(n)*theta[1]
   return (values[:,None]-centers[None,:])[np.arange(len(values)),np.argmin(abs(values[:,None]-centers[None,:]),axis=1)]
  p=1/n
  fit=least_squares(residual,[p/2,p],bounds=([p*.1,p*.78],[p*.9,p*1.22]),loss='soft_l1',f_scale=.015)
  centers=fit.x[0]+np.arange(n)*fit.x[1]
  if centers[-1]>1.02:continue
  errors=residual(fit.x);rms=float(np.sqrt(np.mean(errors**2)));score=float(np.mean(np.minimum((errors/.04)**2,9)))+.25*n
  candidates.append(dict(count=n,centers=centers.tolist(),rmse=rms,score=score,spacing=float(fit.x[1])))
 return min(candidates,key=lambda c:c['score'])

def infer(packet):
 start=time.perf_counter();H=homography(packet['quadPx'],[[0,1],[1,1],[1,0],[0,0]])
 points=project(H,packet['openingCentersPx']);fx=fit_axis(points[:,0],packet.get('maxColumns',16));fy=fit_axis(points[:,1],packet.get('maxRows',16))
 centers=np.array([[x,y] for x in fx['centers'] for y in fy['centers']]);predicted=project(np.linalg.inv(H),centers)
 holdout=np.asarray(packet.get('holdoutCentersPx',[]));holdout_errors=[]
 if len(holdout):holdout_errors=np.linalg.norm(holdout[:,None,:]-predicted[None,:,:],axis=2).min(axis=1).tolist()
 assignments=np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2).argmin(axis=1)
 observed=set(assignments.tolist());cells=[dict(x=float(x),y=float(y),evidence='observed-center' if i in observed else 'inferred-repeat') for i,(x,y) in enumerate(centers)]
 return dict(id=packet['id'],match=packet['match'],normal=packet['normal'],region=packet['region'],style=packet['style'],columns=fx,rows=fy,cells=cells,sourceIds=packet['sourceIds'],inputSha256=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),method='sparse image landmarks, projective rectification, robust lattice search',status='image-layout-fit',directCenterSamples=len(points),completedCells=len(cells)-len(observed),holdoutPixelErrors=holdout_errors,independentValidation=False,fitSeconds=round(time.perf_counter()-start,4),uncertainty=packet.get('uncertainty',[]))

def main():
 src=json.load(open(ROOT/'observations/image-landmarks.json'));results=[infer(p) for p in src['packets']]
 (ROOT/'facade-layouts.json').write_text(json.dumps({'schemaVersion':1,'fits':results},indent=2))
 for f in results:print(f['id'],f['columns']['count'],f['rows']['count'],f['columns']['rmse'],f['rows']['rmse'],f['holdoutPixelErrors'])
if __name__=='__main__':main()
