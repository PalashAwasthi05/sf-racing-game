"""Scene objects require observed identity, pose and a physical support graph."""
import numpy as np

def admit_object(record):
 required=['observedKind','position','orientation','supportKey','sourceIds']
 absent=[k for k in required if not record.get(k)]
 if record.get('semanticRole')=='control-point':absent.append('physical-object-observation')
 if record.get('positionUncertaintyM',float('inf'))>.3:absent.append('precise-position')
 if record.get('observedKind')=='sign' and not (record.get('artwork') or record.get('agreedOcr')):absent.append('observed-sign-content')
 return {'accepted':not absent,'missing':absent}

def same_physical_object(a,b,distance_m=.25,angle_degrees=8):
 if a.get('sourceObjectId') and a.get('sourceObjectId')==b.get('sourceObjectId'):return True
 if a.get('observedKind')!=b.get('observedKind') or a.get('supportKey')!=b.get('supportKey'):return False
 if np.linalg.norm(np.array(a['position'])-b['position'])>distance_m:return False
 aa=np.asarray(a['orientation']);bb=np.asarray(b['orientation']);dot=np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb));return np.degrees(np.arccos(np.clip(dot,-1,1)))<=angle_degrees

def leaf_is_supported(base,branch_start,branch_end,terrain,obstacles):
 p,a,b=np.asarray(base),np.asarray(branch_start),np.asarray(branch_end);v=b-a;t=np.clip(np.dot(p-a,v)/max(np.dot(v,v),1e-9),0,1)
 if np.linalg.norm(p-(a+t*v))>.02:return False
 if p[2]<terrain(p[0],p[1]):return False
 return not any(o(p) for o in obstacles)
