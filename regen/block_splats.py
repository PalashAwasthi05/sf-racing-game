"""Build independently stereo-reconstructed and appearance-fitted oriented Gaussian discs on a captured block."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from block_splats_pose import unit, load_cameras, candidate_pairs, camera_refinement


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def calibrated_inputs(path, output):
    data = json.loads(path.read_text())
    cameras = load_cameras(data['train'],path)
    if ({c['id'] for c in cameras}&{c['id'] for c in data['heldout']} or
        {c['driveId'] for c in cameras}&{c['driveId'] for c in data['heldout']}):
        raise ValueError('train and holdout must use separate frames and drives')
    route = np.array(data['route']['enu'])[:,:2]
    along = unit(route[-1]-route[0])
    pose_pairs = candidate_pairs(cameras,along)
    stamp = hashlib.sha256((json.dumps(data['train'],sort_keys=True)+Path(__file__).with_name('block_splats_pose.py').read_text()).encode()).hexdigest()
    f = output/'splat-train-poses.json'
    pose = json.loads(f.read_text()) if f.exists() else {}
    if pose.get('inputsSha256') != stamp:
        print('Fit relative train-camera poses',len(cameras),len(pose_pairs),flush=True)
        pose = camera_refinement(cameras,along,pose_pairs)
        pose['inputsSha256'] = stamp
        atomic_json(f,pose)
    else:
        for c in cameras:
            if pose.get('used'):
                p=pose['cameras'][c['id']]
                c['eye']=np.array(p['eyeEnu']);c['R']=np.array(p['rotationWorldToCamera'])
    print('Pose',str({k:v for k,v in pose.items() if k!='cameras'}),flush=True)
    return data, cameras, route, along, pose


def write_model(path,model):
    xyz,u,v,rgb,opacity=model
    packed=np.empty(len(xyz),dtype=[('center','<f4',(3,)),('major','<f4',(3,)),
                                    ('minor','<f4',(3,)),('rgba','u1',(4,))])
    for k,col in [('center',xyz),('major',u),('minor',v)]:
        packed[k]=col[:,[0,2,1]]*np.array([1,1,-1])
    packed['rgba']=np.clip(np.rint(np.column_stack([rgb,opacity])*255),0,255).astype('u1')
    if packed.itemsize!=40 or not len(packed) or not all(np.isfinite(packed[k]).all() for k in ['center','major','minor']):
        raise ValueError('invalid Gaussian asset')
    tmp=path.with_suffix('.tmp')
    packed.tofile(tmp)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(path,pose_only=False):
    from block_splats_stereo import triangulate_pair,fuse
    from block_splats_raster import train
    out=path.parent
    start=time.monotonic()
    data,cameras,route,along,pose=calibrated_inputs(path,out)
    if pose_only:
        return
    if not all((out/f'surface-side-{side}.npz').is_file() for side in (-1,1)):
        raise FileNotFoundError('both block facade support charts are required for static support gating')
    segments=[]
    pair_stats=[]
    for i,j in candidate_pairs(cameras,along):
        for a,b in [(i,j),(j,i)]:
            xyz,u,v,rgb,score,stat=triangulate_pair(cameras[a],cameras[b],route,along,stride=3)
            pair_stats.append({**stat,'left':cameras[a]['id'],'right':cameras[b]['id']})
            segments.append((xyz,u,v,rgb,score))
        print('Stereo',i,j,pair_stats[-2]['validStrideSamples'],pair_stats[-1]['validStrideSamples'],flush=True)
    model,counts=fuse(segments,route,along,out)
    del segments
    print('Fusion',counts,flush=True)
    atomic_json(out/'splat-fusion.json',{'counts':counts,'pairs':pair_stats})
    def holdout_cameras():
        return load_cameras(data['heldout'],path)
    rgb,tr,ev=train(model,cameras,along,holdout_cameras,out)
    xyz,u,v,_,opacity=model
    output_hash=write_model(out/'block-splats.bin',(xyz,u,v,rgb,opacity))
    info={'schema':'block-surface-gaussians-v1','method':'Left/right-consistent spherical stereo, independent pair fusion, tangent differential SVD and iterated robust alpha-composited RGB fitting.',
          'coordinateSystem':'viewer (east,up,-north) metres','strideBytes':40,'count':len(xyz),
          'attributes':['center f32[3]','tangentSigmaMajor f32[3]','tangentSigmaMinor f32[3]','sRGB+peakOpacity u8[4]'],
          'sha256':output_hash,'bytes':int((out/'block-splats.bin').stat().st_size),
          'trainIds':[c['id'] for c in cameras],'heldoutIds':[c['id'] for c in data['heldout']],
          'trainDrive':sorted({c['driveId'] for c in cameras}),'heldoutDrive':sorted({c['driveId'] for c in data['heldout']}),
          'reconstruction':counts,'stereoPairs':len(pair_stats)//2,'directionalStereoReconstructions':len(pair_stats),
          'poseRefinement':{k:v for k,v in pose.items() if k not in ['cameras','inputsSha256']},'training':tr,'independentDrive':ev,
          'axesMetres':{k:np.percentile(np.linalg.norm(arr,axis=1),[5,50,95]).tolist() for k,arr in [('majorP05P50P95',u),('minorP05P50P95',v)]},
          'peakOpacityP05P50P95':np.percentile(opacity,[5,50,95]).tolist(),'compute':'CPU OpenCV SGBM, SciPy relative-pose optimization and native CPU Gaussian rasterizer.',
          'elapsedSecondsExcludingCachedPose':time.monotonic()-start,
          'limits':['Surface Gaussians use two tangent axes and zero thickness. Stereo fixes center and covariance before RGB optimization.',
                    'A 3.1 m distance gate to an R31 exterior facade removes much of the foreground. Moving and reflective surfaces can survive this gate.',
                    'Metrics measure RGB only on the rendered layer at alpha >= 0.72. Unfilled regions use R31 in the hybrid viewer. Holdout cameras are independently registered and capture a different year.',
                    'The final renderer may use different projection, antialiasing, alpha cutoff, depth testing, and color management from the CPU fitting rasterizer.']}
    atomic_json(out/'block-splats.json',info)
    print('Wrote',info['count'],info['bytes'],info['sha256'],'heldout MAE',ev['maeSrgb255'],'coverage',ev['coverage'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('cameras',type=Path)
    ap.add_argument('--poses-only',action='store_true')
    args=ap.parse_args()
    cv2.setNumThreads(4)
    reconstruct(args.cameras,args.poses_only)
