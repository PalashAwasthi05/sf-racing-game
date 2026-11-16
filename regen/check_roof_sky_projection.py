"""Local conditional roof/observed-sky conflicts, never an automatic mesh edit.

Uses the actual exported roof geometry in its stored local XYZ frame, the exact
perspective source pose, and offline ADE sky probabilities. Other known building
prisms deliberately over-occlude uncertain background roof visibility.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from regen.propose_roof_boundaries import project_target


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SkyModel:
    def __init__(self, root):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        self.torch=torch
        self.device='mps' if torch.backends.mps.is_available() else 'cpu'
        path=sorted((root/'.models/models--nvidia--segformer-b2-finetuned-ade-512-512/snapshots').glob('*'))[-1]
        self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForSemanticSegmentation.from_pretrained(path,local_files_only=True,use_safetensors=False).to(self.device).eval()
        labels=self.model.config.id2label;self.codes={n:int(i) for i,n in labels.items()}
        self.identity={'configPath':str(path/'config.json'),'configSha256':sha(path/'config.json'),'device':self.device,'networkUsed':False}

    def infer(self, image):
        t=self.torch
        inputs=self.processor(images=image,return_tensors='pt').to(self.device)
        with t.no_grad():
            logits=self.model(**inputs).logits
            p=t.nn.functional.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False).softmax(1)[0]
            groups={}
            for group,names in {'sky':['sky'],'structure':['building','wall','house','skyscraper'],
                    'occluder':['tree','palm','plant','pole','streetlight','person','car','fence','traffic light']}.items():
                groups[group]=p[[self.codes[n] for n in names if n in self.codes]].sum(0).float().cpu().numpy()
        del inputs,logits,p
        return groups


def raster(mesh, projection, size):
    """Perspective-correct triangle depth; no backface culling."""
    w,h=size;camera=np.asarray(projection['cameraXYZ']);basis=np.asarray(projection['basisRightDownForward'])
    q=(mesh.vertices-camera)@basis.T
    xy=q[:,:2]/np.maximum(q[:,2,None],1e-8)*projection['focalPx']+np.array(size)/2-.5
    depth=np.full((h,w),np.inf,np.float32);face=np.full((h,w),-1,np.int32)
    for i,indices in enumerate(mesh.faces):
        z=q[indices,2]
        if np.any(z<=.05):continue
        a,b,c=xy[indices]
        x0=max(0,int(np.floor(min(a[0],b[0],c[0]))));x1=min(w-1,int(np.ceil(max(a[0],b[0],c[0]))))
        y0=max(0,int(np.floor(min(a[1],b[1],c[1]))));y1=min(h-1,int(np.ceil(max(a[1],b[1],c[1]))))
        if x0>x1 or y0>y1:continue
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        inside=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)
        inverse=u/z[0]+v/z[1]+(1-u-v)/z[2]
        zz=1/np.maximum(inverse,1e-9);old=depth[y0:y1+1,x0:x1+1]
        update=inside&(zz<old);old[update]=zz[update];face[y0:y1+1,x0:x1+1][update]=i
    return depth,face


def prism(mass, top):
    # Convex hull intentionally includes courtyard/concave voids only as an
    # occlusion proxy. It cannot create a false conflict; it may hide one.
    p=np.asarray(mass['outline'][0]);n=len(p)
    vertices=np.vstack([np.c_[p,np.full(n,mass['z0'])],np.c_[p,np.full(n,top)]])
    return trimesh.Trimesh(vertices=vertices,faces=[]).convex_hull


def compare(groups, depth, face, other_depth, projection, margin):
    h,w=depth.shape
    raw=(groups['sky']>=.95)&(groups['occluder']<=.05)&(groups['structure']<=.05)
    count,labels=cv2.connectedComponents(raw.astype(np.uint8))
    border=set(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])-{0}
    raw&=np.isin(labels,list(border))
    clearance=cv2.distanceTransform(raw.astype(np.uint8),cv2.DIST_L2,5)
    safe=clearance>margin
    # Require solid source content below in the registered target facade's
    # own projected horizontal span. Pure background columns abstain.
    solid=(groups['structure']>=.7)&(groups['occluder']<=.1)
    corners=np.asarray(projection['facadeCornersPx'])
    xs=np.arange(w)
    def edge_y(indices):
        p=corners[indices];p=p[np.argsort(p[:,0])]
        return np.interp(xs,p[:,0],p[:,1])
    top,bottom=edge_y([1,3]),edge_y([0,2])
    # Visibility support is the upper third of the registered source facade,
    # not a distant background building hundreds of pixels below its roof.
    # This ROI only causes abstention; it is never an observed eave height.
    source_band_limit=top+.35*np.maximum(bottom-top,0)
    solid&=np.arange(h)[:,None]<=source_band_limit[None,:]
    solid_below=np.maximum.accumulate(solid[::-1],axis=0)[::-1]
    x0,x1=sorted(corners[[1,3],0]);columns=(xs>=x0)&(xs<=x1)
    visible=np.isfinite(depth)&(depth<=other_depth+.02)
    eligible=visible&solid_below&columns[None,:]
    conflict=eligible&safe
    diagnostic={str(p):int((eligible&(groups['sky']>=p)&(groups['occluder']<=.05)).sum()) for p in (.5,.8,.9,.95)}
    residual=[]
    for x in np.flatnonzero(conflict.any(0)):
        yy=np.flatnonzero(conflict[:,x]);below=np.flatnonzero(solid[:,x]&(np.arange(h)>yy.min()))
        if len(below):residual.append(float(below[0]-yy.min()))
    facets,counts=np.unique(face[conflict],return_counts=True)
    return conflict,visible,{'visibleRoofPixels':int(visible.sum()),'eligibleRoofPixels':int(eligible.sum()),
        'conflictPixels':int(conflict.sum()),'conflictColumns':int(conflict.any(0).sum()),
        'fractionOfEligibleRoofPixels':float(conflict.sum()/max(1,eligible.sum())),
        'maximumSafeSkyClearancePx':float(clearance[conflict].max()) if conflict.any() else 0.,
        'diagnosticUnErodedSkyThresholdCounts':diagnostic,
        'diagnosticCountsNotConstraint':'Threshold sensitivity only; no structure-probability gate or margin applied to these counts.',
        'strictSkyMargin3pxConflictPixels':int((eligible&(clearance>3)).sum()),
        'maximumSkyProbabilityOnEligibleRoof':float(groups['sky'][eligible].max()) if eligible.any() else None,
        'skyConflictPixelsBeforeKnownOcclusion':int((np.isfinite(depth)&safe&solid_below&columns[None,:]).sum()),
        'skyPixelsAbstainedForNoUpperFacadeSupport':int((visible&safe&~solid_below).sum()),
        'sourceVisibilityBand':'confident structure below within upper35% of registered facade; top-edge horizontal span; eligibility only, not an eave measurement',
        'verticalDistanceToSolidBelowP50P95MaxPx':np.percentile(residual,[50,95,100]).tolist() if residual else [],
        'otherKnownBuildingOccludedRoofPixels':int((np.isfinite(depth)&~visible).sum()),
        'implicatedFacets':[{'faceIndex':int(i),'conflictPixels':int(n)} for i,n in zip(facets,counts) if i>=0]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('jobs','poses','metadata','calibration','scene','massing','cache','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--margin',type=float,default=5.)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent.parent
    jobs=json.loads(a.jobs.read_text());poses=json.loads(a.poses.read_text())['poses'];metadata=json.loads(a.metadata.read_text());pieces=metadata.get('pieces',metadata)
    calibration=json.loads(a.calibration.read_text())['images']
    primary={(j['side'],j['image']) for j in jobs};rows={j['building']:i for i,j in enumerate(jobs)}
    cached=json.loads((a.cache/'candidates.json').read_text());by_source={(r['side'],r['image']):r for r in cached['records']}
    scene=trimesh.load(a.scene,force='scene');masses=json.loads(a.massing.read_text())
    roofs={}
    for m in masses:
        meshes=[g for key,g in scene.geometry.items() if key.startswith('roofs__'+m['id']+'___')]
        if meshes:roofs[m['id']]=trimesh.util.concatenate(meshes)
    proxies={m['id']:prism(m,float(roofs[m['id']].bounds[1,2])) for m in masses if m['id'] in roofs}
    model=None;records=[];started=perf_counter()
    for index,job in enumerate(poses):
        row=rows.get(job['building']);record={'index':index,'reviewRow':row,'building':job['building'],'side':job['side'],'image':job['image'],
            'role':'primary' if (job['side'],job['image']) in primary else 'heldout','cameraXYValidated':False,'status':'unresolved'}
        try:
            matches=[p for p in pieces if p['id']==job['side']]
            if len(matches)!=1:raise ValueError('ambiguous exact source side join')
            piece=matches[0];view=piece['views'][job['view']]
            if view['image']!=job['image'] or Path(view['persp']).name!=Path(job['persp']).name:raise ValueError('source image/frame mismatch')
            calibration_record=calibration[job['image']]
            if np.linalg.norm(np.asarray(job['from'][:2])-calibration_record['cameraXY'])>.002:raise ValueError('pose/calibration camera XY mismatch')
            if view.get('registered')!=calibration_record.get('registration'):raise ValueError('source/calibration registration mode mismatch')
            if sha(calibration_record['sourcePath'])!=calibration_record['sourceSha256']:raise ValueError('calibrated panorama source hash stale')
            path=Path(job['persp']);im=Image.open(path).convert('RGB');digest=sha(path)
            if list(im.size)!=[job['width'],job['height']]:raise ValueError('source dimensions mismatch')
            projection=project_target(job,piece,im.size)
            prior=by_source.get((job['side'],job['image']));probpath=a.output/f'{index:02d}-sky-probabilities.npz'
            if prior and prior['sourceSha256']==digest:
                with np.load(a.cache/f'{prior["row"]:02d}-semantic-probabilities.npz') as ar:groups={k:ar[k].astype(np.float32) for k in ('sky','structure','occluder')}
                cachepath=a.cache/f'{prior["row"]:02d}-semantic-probabilities.npz';cache_kind='verified-primary-cache'
            elif probpath.exists() and (a.output/f'{index:02d}-source.json').exists() and json.loads((a.output/f'{index:02d}-source.json').read_text())['sourceSha256']==digest:
                with np.load(probpath) as ar:groups={k:ar[k].astype(np.float32) for k in ar.files}
                cachepath=probpath;cache_kind='verified-local-cache'
            else:
                if model is None:model=SkyModel(root)
                groups=model.infer(im);np.savez_compressed(probpath,**{k:v.astype(np.float16) for k,v in groups.items()});cachepath=probpath;cache_kind='new-offline-segmentation'
                (a.output/f'{index:02d}-source.json').write_text(json.dumps({'sourcePath':str(path),'sourceSha256':digest,'model':model.identity},indent=2))
            roof=roofs[job['building']];depth,faces=raster(roof,projection,im.size);other=np.full(depth.shape,np.inf,np.float32)
            for identity,proxy in proxies.items():
                if identity==job['building']:continue
                dd,_=raster(proxy,projection,im.size);np.minimum(other,dd,out=other)
            conflict,visible,metrics=compare(groups,depth,faces,other,projection,a.margin)
            out=np.asarray(im).copy();out[conflict]=(out[conflict]*.35+np.array([255,0,60])*.65).astype(np.uint8)
            contours,_=cv2.findContours(visible.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(out,contours,-1,(0,255,220),1)
            overlay=Image.fromarray(out);draw=ImageDraw.Draw(overlay);draw.rectangle((0,0,min(im.width,700),36),fill='white');draw.text((5,4),f'row{row:02d} {record["role"]}: {metrics["conflictPixels"]} conditional sky-conflict pixels',fill='black');draw.text((5,20),'cyan actual roof; red confident source sky; camera XY uncertain',fill='black');overlay.save(a.output/f'{index:02d}-overlay.jpg',quality=90)
            record.update(status='conditional-sky-conflict' if metrics['conflictPixels'] else 'no-conflict-detected',sourcePath=str(path),sourceSha256=digest,
                projection=projection,semanticCachePath=str(cachepath),semanticCacheSha256=sha(cachepath),semanticCacheKind=cache_kind,metrics=metrics,
                calibrationImageSourceSha256=calibration_record['sourceSha256'],calibrationPoseXYJoinToleranceM=.002,
                facetVerticesXYZ=[{'faceIndex':f['faceIndex'],'verticesXYZ':roof.triangles[f['faceIndex']].tolist()} for f in metrics['implicatedFacets']],
                hardConstraint=False,geometryChanged=False,occlusionMethod='other known building convex massing prisms to actual maximum roof Z; conservative over-occlusion',
                limitations=['Absolute camera XY/height and common registration bias unvalidated.','LiDAR and photo dates can differ.','No source roof/wall semantic inference.','Missing neighbors are not reconstructed; observed non-sky excludes them.','Cropped or pure background columns abstain.'])
        except (ValueError,KeyError,IndexError,OSError) as error:record['reason']=str(error)
        records.append(record)
        report={'algorithm':'conditional-actual-roof-in-observed-sky-v1','algorithmSha256':sha(__file__),
            'inputs':{name:{'path':str(getattr(a,name)),'sha256':sha(getattr(a,name))} for name in ('jobs','poses','metadata','calibration','scene','massing')},
            'coordinateFrame':'GLB mesh stored vertices local XYZ; scene root x,z,-y display transform deliberately not applied; jobs use local XYZ.',
            'skyProbabilityAtLeast':.95,'occluderProbabilityAtMost':.05,'structureProbabilityAtMost':.05,'safeSkyMarginPx':a.margin,
            'totalSeconds':perf_counter()-started,'records':records,'activeGeometryChanges':0}
        (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('index','reviewRow','role','status','reason')}|{'pixels':record.get('metrics',{}).get('conflictPixels')}),flush=True)
    for page in range((len(records)+7)//8):
        sheet=Image.new('RGB',(1440,760),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[page*8:page*8+8]):
            path=a.output/f'{r["index"]:02d}-overlay.jpg'
            if not path.exists():continue
            im=Image.open(path);im.thumbnail((355,335));x=j%4*360;y=j//4*380;sheet.paste(im,(x,y+20));draw.text((x+4,y+4),f'view{r["index"]:02d} row{r["reviewRow"]:02d} {r["role"]}',fill='black')
        sheet.save(a.output/f'contact-sheet-{page+1}.jpg',quality=90)
    summary=[];facet_evidence=[]
    for building,row in rows.items():
        views=[r for r in records if r['building']==building]
        source_facets={}
        for record in views:
            for f in record.get('metrics',{}).get('implicatedFacets',[]):
                source_facets.setdefault(f['faceIndex'],[]).append({'viewIndex':record['index'],'image':record['image'],
                    'sourceSha256':record['sourceSha256'],'conflictPixels':f['conflictPixels'],'cameraXYZ':record['projection']['cameraXYZ']})
        matched=[]
        for face,observations in source_facets.items():
            images={r['image'] for r in observations}
            cameras=np.array([r['cameraXYZ'] for r in observations]);baseline=float(np.linalg.norm(cameras[:,None,:2]-cameras[None,:,:2],axis=2).max())
            if len(images)>=2:
                matched.append(face)
                facet_evidence.append({'building':building,'reviewRow':row,'faceIndex':face,'independentImages':len(images),
                    'maximumHorizontalCameraBaselineM':baseline,'observations':observations,
                    'verticesXYZ':roofs[building].triangles[face].tolist(),'automaticHardConstraint':False,
                    'qualification':'Repeated pixel conflict, conditional on shared absolute camera registration. Does not establish the corrected surface or authorize deleting the whole triangle.'})
        summary.append({'building':building,'reviewRow':row,'sourceViews':len(views),
            'primaryConflictPixels':sum(r.get('metrics',{}).get('conflictPixels',0) for r in views if r['role']=='primary'),
            'heldoutConflictPixels':sum(r.get('metrics',{}).get('conflictPixels',0) for r in views if r['role']=='heldout'),
            'conflictingIndependentImages':len({r['image'] for r in views if r.get('metrics',{}).get('conflictPixels',0)}),
            'facetsWithTwoIndependentImageConflicts':len(matched),
            'unresolvedViews':[r['index'] for r in views if r['status']=='unresolved']})
    (a.output/'summary.json').write_text(json.dumps({'sourceViews':len(records),'primaryViews':sum(r['role']=='primary' for r in records),
        'heldoutViews':sum(r['role']=='heldout' for r in records),'rows':summary,'activeGeometryChanges':0},indent=2)+'\n')
    (a.output/'multi-view-facet-proposals.json').write_text(json.dumps({'status':'conditional-review-proposals-only','facets':facet_evidence},indent=2)+'\n')


if __name__=='__main__':main()
