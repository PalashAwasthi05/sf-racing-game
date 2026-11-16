"""Diagnostic photograph comparisons for frozen held-out browser captures."""
import argparse, collections, json, math, pathlib
import cv2
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--review',type=pathlib.Path,required=True);a=ap.parse_args();p=a.review.resolve()
jobs=json.loads((p/'reference-jobs.json').read_text());audit=json.loads((p/'capture-audit.json').read_text());assert audit['status']=='complete';assert len(audit['captures'])==18
regions={'central':[64,40,576,360],'upperCentral':[64,40,576,240]}
def scores(ref,im,bounds):
    x0,y0,x1,y1=bounds;ref=ref[y0:y1,x0:x1];im=im[y0:y1,x0:x1];diff=ref.astype(float)-im.astype(float);mse=np.mean(diff**2)
    x=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY).astype(float);y=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
    blur=lambda z:cv2.GaussianBlur(z,(11,11),1.5)
    ux,uy=blur(x),blur(y);vx=blur(x*x)-ux*ux;vy=blur(y*y)-uy*uy;cov=blur(x*y)-ux*uy
    ssim=np.mean(((2*ux*uy+6.5025)*(2*cov+58.5225))/((ux*ux+uy*uy+6.5025)*(vx+vy+58.5225)))
    ex=cv2.Canny(x.astype(np.uint8),60,160)>0;ey=cv2.Canny(y.astype(np.uint8),60,160)>0
    dx=cv2.distanceTransform((~ex).astype(np.uint8),cv2.DIST_L2,3);dy=cv2.distanceTransform((~ey).astype(np.uint8),cv2.DIST_L2,3)
    chamfer=(np.minimum(dx[ey],10).mean()+np.minimum(dy[ex],10).mean())/2 if ex.any() and ey.any() else 10.
    return dict(rgbMAE=float(np.abs(diff).mean()),rgbRMSE=float(math.sqrt(mse)),psnrDb=float(10*math.log10(255**2/mse)) if mse else 100.,graySSIM=float(ssim),edgeChamferPxClipped10=float(chamfer))
rows=[];tilew,tileh,header=640,400,34;board=np.full((6*(tileh+header),4*tilew,3),245,np.uint8)
for row,j in enumerate(jobs['jobs']):
    ref=cv2.imread(j['referencePath']);images=[ref]
    for variant in ('baseline','photo','splats'):
        im=cv2.imread(str(p/f'{j["id"]}-{variant}.png'));assert im.shape==ref.shape;images.append(im)
        rows.append(dict(jobId=j['id'],variant=variant,regions={name:scores(ref,im,bounds) for name,bounds in regions.items()}))
    small=np.full((2*(tileh+header),2*tilew,3),245,np.uint8)
    for col,(label,im) in enumerate(zip(('Heldout photograph','Current R31 mesh','Photo mesh','Free 3D splats'),images)):
        y=row*(tileh+header);x=col*tilew;board[y+header:y+header+tileh,x:x+tilew]=im;cv2.putText(board,j['id']+' | '+label,(x+10,y+23),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
        yy=(col//2)*(tileh+header);xx=(col%2)*tilew;small[yy+header:yy+header+tileh,xx:xx+tilew]=im;cv2.putText(small,j['id']+' | '+label,(xx+10,yy+23),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(p/f'{j["id"]}-comparison.jpg'),small,[cv2.IMWRITE_JPEG_QUALITY,95])
means={}
for variant in ('baseline','photo','splats'):
    subset=[r for r in rows if r['variant']==variant];means[variant]={name:{metric:float(np.mean([r['regions'][name][metric] for r in subset])) for metric in subset[0]['regions'][name]} for name in regions}
metrics=dict(schema='block-heldout-metrics-v1',regions=regions,perView=rows,meanAcrossSixViews=means,limits='No alignment, recoloring, semantic masks or per-variant postprocessing. Central region fixed before variants were viewed; upperCentral supplemental band excludes much of the capture vehicle. Both retain trees, people, parked vehicles, lighting/time changes, and registration error. Metrics are photograph similarity diagnostics, not measured geometric accuracy.',edgeDefinition='Canny gray thresholds60/160; symmetric mean distance between edges clipped10px; lower is better',ssimDefinition='Grayscale SSIM Gaussian11 sigma1.5 constants (0.01*255)^2,(0.03*255)^2; higher is better',frozenTrial=audit['trial'],loadedFileHashes=audit['loadedFiles'])
(p/'metrics.json').write_text(json.dumps(metrics,indent=2));cv2.imwrite(str(p/'source-and-three-versions-contact-sheet.jpg'),board,[cv2.IMWRITE_JPEG_QUALITY,95]);print(json.dumps(means,indent=2))
