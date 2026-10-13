"""Rectify annotated opening edges, not a guessed size around a center lattice.
Vision labels/plane correspondences are inputs. This is not automatic segmentation.
No completion is performed; unobserved windows require separate evidence.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from .infer_facades import homography,project

def infer(packet):
 h=homography(packet['quadPx'],[[0,1],[1,1],[1,0],[0,0]]);out=[];errors=[]
 a,b,c,d=packet['region']
 for op in packet['openings']:
  uv=project(h,op['cornersPx']);lo=uv.min(axis=0);hi=uv.max(axis=0)
  # Opposite boundary observations estimate a rectangle on this single plane.
  left=np.mean(uv[[0,3],0]);right=np.mean(uv[[1,2],0]);top=np.mean(uv[[0,1],1]);bottom=np.mean(uv[[2,3],1])
  snapped=np.array([[left,top],[right,top],[right,bottom],[left,bottom]])
  error=np.linalg.norm(project(np.linalg.inv(h),snapped)-op['cornersPx'],axis=1)
  errors.extend(error.tolist());out.append({'id':op['id'],'rect':[a+left*(c-a),b+bottom*(d-b),a+right*(c-a),b+top*(d-b)],'style':op['style'],'sourceIds':packet['sourceIds'],'evidence':'observed-boundary','cornerResidualPx':error.tolist()})
 return {'id':packet['id'],'profileId':packet['profileId'],'openings':out,'method':'four-corner plane rectification, opposing-edge fit; no repetition completion','inputSha256':hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),'meanCornerResidualPx':float(np.mean(errors)),'independentValidation':False}

def main():
 root=Path(__file__).parent;d=json.load(open(root/'observations/opening-boundaries.json'));fits=[infer(p) for p in d['packets']]
 (root/'boundary-layouts.json').write_text(json.dumps({'schemaVersion':1,'fits':fits},indent=2))
 for f in fits:print(f['id'],len(f['openings']),f['meanCornerResidualPx'])
if __name__=='__main__':main()
