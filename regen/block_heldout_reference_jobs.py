"""Freeze six held-out photograph views before comparing any representation.
Run with the select_sequence_block environment and --area <area-directory>.
"""
import argparse, hashlib, json, math, pathlib
import cv2
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--area',type=pathlib.Path,required=True);a=ap.parse_args()
root=a.area.resolve()/'block-splat-trial';contract=json.loads((root/'cameras.json').read_text());out=root/'heldout-comparison';out.mkdir(exist_ok=True)
route=np.array(contract['route']['enu'],float);lengths=np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1);total=lengths.sum();cum=np.r_[0,np.cumsum(lengths)]
width,height,vfov=640,400,60.;f=height/2/math.tan(math.radians(vfov/2));hfov=math.degrees(2*math.atan(width/(2*f)))
xx,yy=np.meshgrid((np.arange(width)+.5-width/2)/f,(np.arange(height)+.5-height/2)/f)
jobs=[];board=np.full((2*(height+34),3*width,3),245,np.uint8)
for column,fraction in enumerate((.2,.5,.8)):
    c=min(contract['heldout'],key=lambda c:abs(c['alongM']-total*fraction));eye=np.array(c['eyeEnu'],float)
    seg=min(np.searchsorted(cum,c['alongM'],side='right')-1,len(lengths)-1);tangent=route[seg+1,:2]-route[seg,:2];tangent/=np.linalg.norm(tangent)
    left=np.r_[-tangent[1],tangent[0],0.]
    for row,(side,sign) in enumerate((('left',1),('right',-1))):
        target=eye+sign*left*30+np.array([0,0,3]);forward=target-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
        rays=forward+xx[:,:,None]*right+yy[:,:,None]*down;R=np.array(c['rotationWorldToCamera']);dc=rays@R.T
        u=(.5+np.arctan2(dc[:,:,0],dc[:,:,2])/(2*np.pi))%1;v=.5-np.arctan2(-dc[:,:,1],np.hypot(dc[:,:,0],dc[:,:,2]))/np.pi
        original=cv2.imread(c['imagePath']);h,w=original.shape[:2];image=cv2.remap(original,(u*w-.5).astype(np.float32),(v*h-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        jid=f'p{int(fraction*100):02d}-{side}';path=out/f'{jid}-source.jpg';cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,95])
        job=dict(id=jid,fraction=fraction,side=side,width=width,height=height,verticalFovDeg=vfov,horizontalFovDeg=hfov,aimElevationDeg=math.degrees(math.atan2(3,30)),fromEnu=eye.tolist(),atEnu=target.tolist(),sourceCameraId=c['id'],driveId=c['driveId'],captureDate=c['captureDate'],requestedAlongM=total*fraction,actualAlongM=c['alongM'],referencePath=str(path),sourceImagePath=c['imagePath'],sourceImageSha256=c['imageSha256'],referenceSha256=hashlib.sha256(path.read_bytes()).hexdigest(),rotationWorldToCamera=R.tolist(),sourceUrl=c['sourceUrl'])
        jobs.append(job);y=row*(height+34);board[y+34:y+34+height,column*width:(column+1)*width]=image;cv2.putText(board,f'{jid} | heldout {c["captureDate"]}',(column*width+10,y+23),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1,cv2.LINE_AA)
manifest=dict(schema='block-heldout-reference-v1',frozenBeforeVariantComparison=True,contractSha256=hashlib.sha256((root/'cameras.json').read_bytes()).hexdigest(),selection='Nearest original heldout camera to 20/50/80 percent of selected road length; both street normals; 30m target with 3m rise. No identity rules.',coordinateSystem=contract['coordinateSystem'],comparisonContract={'cameraApi':'window.__view.pose(job.fromEnu,job.atEnu)','fov':'Set window.__view.camera.fov=job.verticalFovDeg; updateProjectionMatrix before capture','captureApi':'await window.__view.capture(640,400)','variantSwitch':'Digits 1,2,3 / three representation buttons at fixed pose','metricsRegion':'central rectangle x=[64,576),y=[40,360); no claimed semantic building mask','metricLimits':'Raw RGB/SSIM/edge errors include lighting, vegetation, parked vehicles and pose disagreement; they are comparison diagnostics rather than geometric ground truth.'},jobs=jobs)
(out/'reference-jobs.json').write_text(json.dumps(manifest,indent=2));cv2.imwrite(str(out/'reference-contact-sheet.jpg'),board,[cv2.IMWRITE_JPEG_QUALITY,95]);print(json.dumps({'jobsReady':len(jobs),'manifest':str(out/'reference-jobs.json'),'cameras':[dict(id=j['id'],source=j['sourceCameraId'],alongM=j['actualAlongM']) for j in jobs]},indent=2))
