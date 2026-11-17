"""Recover bounded, individual window outlines on paired, plane-rectified wall photographs."""
import argparse
import json
import math
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PIXELS_PER_METRE = 18


def feature_alignment(first, second):
    """Estimate the residual image displacement after projection onto the mapped wall."""
    detector = cv2.SIFT_create(nfeatures=650, contrastThreshold=.026)
    points_a, desc_a = detector.detectAndCompute(first, None)
    points_b, desc_b = detector.detectAndCompute(second, None)
    if desc_a is None or desc_b is None or len(points_b)<2 or len(points_a)<2:
        return None
    matcher = cv2.BFMatcher()
    def matched(a,b):
        return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance < .74*n.distance}
    pairs=matched(desc_a,desc_b)
    reverse=matched(desc_b,desc_a)
    common=[(points_a[i].pt,points_b[j].pt) for i,j in pairs.items() if reverse.get(j)==i]
    if len(common)<8:
        return None
    a,b=np.asarray(common,dtype='float32').transpose(1,0,2)
    # Varying camera tilt and wall distance produce a small projective warp.
    # Fit the actual observed plane using independent feature matches. Assess
    # positional uncertainty at each opening, not at an empty top corner.
    hom,inliers=cv2.findHomography(np.ascontiguousarray(b),np.ascontiguousarray(a),
                                  cv2.RANSAC,2.5)
    if hom is None or inliers.sum()<8 or inliers.mean()<.44:
        return None
    hits=a[inliers.ravel().astype(bool)]
    hull=cv2.convexHull(hits.reshape(-1,1,2))
    if cv2.contourArea(hull)<(3*PIXELS_PER_METRE)**2:return None
    return hom,hull


def intensity_alignment(first,second):
    # Repeated facades may have no distinctive keypoints. Align from the
    # predicted camera pose and require local agreement across the fitted wall.
    h,w=first.shape
    scale=min(1.,300/max(h,w))
    clahe=cv2.createCLAHE(2.,(6,6))
    small=[clahe.apply(cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)) for im in (first,second)]
    try:
        score,matrix=cv2.findTransformECC(*small,np.eye(2,3,dtype='float32'),cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.00001),None,5)
    except cv2.error:return None
    if score<.56:return None
    warp=np.linalg.inv(np.vstack([matrix,[0.,0.,1.]]));warp[:2,2]/=scale
    corners=np.array([[0,0],[w,0],[w,h],[0,h]])
    if np.max(np.linalg.norm(corners@warp[:2,:2].T+warp[:2,2]-corners,axis=1))>2.3*PIXELS_PER_METRE:return None
    aligned=cv2.warpPerspective(second,warp,(w,h))
    witnesses=[]
    for y in range(12,h-22,22):
        for x in range(12,w-22,22):
            pa=first[y:y+22,x:x+22];pb=aligned[y:y+22,x:x+22]
            if min(pa.std(),pb.std())<8 or pb.min()==0:continue
            if cv2.matchTemplate(pa,pb,cv2.TM_CCOEFF_NORMED).item()>.72:witnesses.append([x+11,y+11])
    if len(witnesses)<8:return None
    hull=cv2.convexHull(np.array(witnesses,dtype=np.float32).reshape(-1,1,2))
    if cv2.contourArea(hull)<(3*PIXELS_PER_METRE)**2:return None
    return warp,hull


def aligned_pair(first,second):
    return feature_alignment(first,second) or intensity_alignment(first,second)


def rectangles(rgb):
    """Return dark panes with a rectangular perimeter and locally brighter surround."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    masks=[cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,block,5) for block in (25,51)]
    masks.append(cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),25,75))
    # Stable dark/light regions recover jambs with soft or shadowed boundaries
    # where contours from a single edge threshold have a gap.
    regions,_=cv2.MSER_create(5,75,5500,.23,.15).detectRegions(gray)
    region_contours=[]
    for pts in regions:
        x,y,bw,bh=cv2.boundingRect(pts)
        if len(pts)<.63*bw*bh:continue
        region_contours.append(cv2.convexHull(pts))
    candidates=[]
    masks.append(None)
    for mask in masks:
        contours=region_contours if mask is None else cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
        for curve in contours:
            x,y,bw,bh=cv2.boundingRect(curve)
            if not (.40*PIXELS_PER_METRE<=bw<=4.8*PIXELS_PER_METRE and .70*PIXELS_PER_METRE<=bh<=4.0*PIXELS_PER_METRE):
                continue
            if min(x,y,w-x-bw,h-y-bh)<4 or not .16<=bw/bh<=3.6:continue
            if cv2.contourArea(curve)<.67*bw*bh:continue
            poly=cv2.approxPolyDP(curve,.035*cv2.arcLength(curve,True),True)
            if not 4<=len(poly)<=6:continue
            inner=rgb[y+2:y+bh-2,x+2:x+bw-2]
            color=np.median(inner.reshape(-1,3),axis=0)
            if color[1]>color[0]+7 and color[1]>color[2]+4:continue
            inside=float(np.median(gray[y+2:y+bh-2,x+2:x+bw-2]))
            edges=[gray[y-3:y,x:x+bw],gray[y+bh:y+bh+3,x:x+bw],gray[y:y+bh,x-3:x],gray[y:y+bh,x+bw:x+bw+3]]
            if sum(float(np.median(e))-inside>=6 for e in edges)<3:continue
            candidates.append((x,y,x+bw,y+bh))
    return suppress(candidates)


def overlap(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    area=np.prod(np.maximum(0,hi-lo))
    return float(area/(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-area))


def suppress(boxes):
    kept=[]
    for box in sorted(boxes,key=lambda b:-(b[2]-b[0])*(b[3]-b[1])):
        if all(overlap(box,b)<.47 for b in kept):kept.append(box)
    return kept


def repeat_witnesses(first,aligned,hull,seeds):
    """Find repeats of verified panes only where BOTH photos resolve the same observed profile."""
    h,w=first.shape; boxes=[]; seen_shapes=set()
    for box in seeds:
        x,y,r,b=np.rint(box).astype(int)
        bw,bh=r-x,b-y
        shape=(round(bw/4),round(bh/4))
        if shape in seen_shapes:continue
        seen_shapes.add(shape)
        pad=max(3,min(8,round(.22*min(bw,bh))))
        if min(x,y,w-r,h-b)<=pad:continue
        scores=[]
        for img in (first,aligned):
            tpl=img[y-pad:b+pad,x-pad:r+pad]
            if tpl.std()<9 or tpl.min()==0:break
            scores.append(cv2.matchTemplate(img,tpl,cv2.TM_CCOEFF_NORMED))
        if len(scores)!=2:continue
        score=np.minimum(*scores)
        peaks=(score>=.76)&(score==cv2.dilate(score,np.ones((9,9),np.uint8)))
        for yy,xx in zip(*np.where(peaks)):
            x0,y0=int(xx+pad),int(yy+pad)
            center=(x0+bw/2,y0+bh/2)
            if cv2.pointPolygonTest(hull,center,True)<-12:continue
            boxes.append((x0,y0,x0+bw,y0+bh))
    return boxes


def paired_panes(first,second):
    grays=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in (first,second)]
    matrix=aligned_pair(*grays)
    if matrix is None:return None
    hom,hull=matrix
    # Detect in each original crop independently. The same rectangle must close
    # in both images and survive both the measured and the uncorrected poses.
    second_boxes=rectangles(second)
    corrected=[]
    for x,y,r,b in second_boxes:
        corners=cv2.perspectiveTransform(np.float32([[[x,y],[x,b],[r,y],[r,b]]]),hom)[0]
        corrected.append([*corners.min(0),*corners.max(0)])
    pairs=[]
    for a in rectangles(first):
        mid=((a[0]+a[2])/2.,(a[1]+a[3])/2.)
        if cv2.pointPolygonTest(hull,mid,True)<-12:continue
        scores=[overlap(a,other) if np.linalg.norm(np.subtract(raw[:2],other[:2]))<2.3*PIXELS_PER_METRE else 0.
                for raw,other in zip(second_boxes,corrected)]
        if scores and max(scores)>=.57:
            pairs.append(np.mean([a,corrected[np.argmax(scores)]],axis=0))
    # Single signs, louvers or panels are also rectangular. Require peers with
    # compatible proportions sharing a sill or jamb line.
    if pairs:
        aligned=cv2.warpPerspective(grays[1],hom,(grays[0].shape[1],grays[0].shape[0]))
        pairs=suppress([*pairs,*repeat_witnesses(grays[0],aligned,hull,pairs)])
    out=[]
    for a in pairs:
        size=np.subtract(a[2:],a[:2])
        for b in pairs:
            if np.linalg.norm(np.subtract(a,b))<1:continue
            ratio=np.subtract(b[2:],b[:2])/size
            if np.all((ratio>.7)&(ratio<1.43)) and (abs(a[3]-b[3])<3 or abs(a[0]-b[0])<3):
                out.append(a);break
    return out


def bars(gray, vertical):
    # A mullion or transom must persist across the central span with two
    # opposed edges. One boundary between two reflections is insufficient.
    rows=gray if vertical else gray.T
    across=rows.shape[1]
    if across<16 or rows.shape[0]<11:return []
    crop=rows[round(rows.shape[0]*.15):round(rows.shape[0]*.85)].astype(np.float32)
    strength=np.zeros(across)
    for x in range(4,across-4):
        mid=crop[:,x]
        left=mid-crop[:,x-3];right=mid-crop[:,x+3]
        bright=np.minimum(left,right);dark=np.minimum(-left,-right)
        strength[x]=max(float(np.quantile(bright,.33)),float(np.quantile(dark,.33)))
    candidates=[x for x in range(4,across-4) if strength[x]>=12 and strength[x]>=strength[x-4:x+5].max()]
    peaks=[]
    for x in sorted(candidates,key=lambda j:-strength[j]):
        if all(abs(x-old)>=5 for old in peaks):peaks.append(x)
    return sorted((x+.5)/across for x in peaks)


def paired_style(first,second):
    style={'kind':'window'}
    for vertical,key in ((True,'mullionFractions'),(False,'transomFractions')):
        ap,bp=bars(first,vertical),bars(second,vertical)
        agreed=[]
        for a in ap:
            if not bp:continue
            b=min(bp,key=lambda q:abs(a-q))
            if abs(a-b)<=.07:
                bp.remove(b);fraction=(a+b)/2
                agreed.append(round(fraction if vertical else 1.-fraction,3))
        if agreed:style[key]=sorted(agreed)
    return style


def disjoint_observed_openings(openings):
    # Different scales can find a sash, its pane, and the whole frame. A
    # facade can contain only one host hole there. Prefer the observed outer
    # opening, then recover its internal divisions separately from both views.
    ordered=sorted(openings,key=lambda o:-np.prod(np.subtract(o['rect'][2:],o['rect'][:2])))
    kept=[]
    for op in ordered:
        a=op['rect']
        if not any(np.all(np.minimum(a[2:],k['rect'][2:])-np.maximum(a[:2],k['rect'][:2])>1e-6) for k in kept):
            kept.append(op)
    return kept


def review_sheet(reviews,output):
    if not reviews:return
    from PIL import ImageDraw
    ordered=sorted(reviews,key=lambda row:(row[0],row[1]))
    indices=np.rint(np.linspace(0,len(ordered)-1,min(5,len(ordered)))).astype(int)
    sheet=Image.new('RGB',(808,285*len(indices)),(29,33,40))
    draw=ImageDraw.Draw(sheet)
    for i,j in enumerate(indices):
        count,wall_id,thumbs=ordered[j]
        draw.text((5,i*285+2),f'{wall_id}, {count} outlines  |  matched reference view', fill='white')
        for k,thumb in enumerate(thumbs):sheet.paste(thumb,(6+403*k,i*285+23))
    sheet.save(output,quality=93)


def measure_subdivisions(area, packet, review_path):
    os.environ['SFW_AREA']=str(area.resolve())
    import frontage_crops_v15 as fc
    import auto_read_frontages_v24 as reader
    reader.PPM=PIXELS_PER_METRE
    cv2.setNumThreads(1)
    center=json.loads((area/'config.json').read_text())['region']['center']
    cameras=fc.load_cameras(fc.LocalFrame(center['lon'],center['lat']))
    count=0
    reviews=[]
    before=0
    after=0
    for item in packet['profiles']:
        if item.get('reading',{}).get('coverageMode')!='positive-only':continue
        before+=len(item['openings'])
        item['openings']=disjoint_observed_openings(item['openings'])
        after+=len(item['openings'])
        mid=np.asarray(item['sourceWallMidpointEN']);width=item['frameU'][1]-item['frameU'][0]
        zmin,zmax=item['reading']['observedZRangeM']
        height=item['heightRefM'];extent=zmax-zmin
        wall=dict(midpoint=mid,normal=np.asarray(item['normal']),lengthM=width,heightM=extent)
        ground=fc.ground_at(mid) or 0.;grays=[];images=[]
        for src in item['sourceIds'][:2]:
            cam=cameras[src.split(':',1)[1]];cam['groundZ']=fc.ground_at(cam['xy'])
            with Image.open(cam['path']) as im:
                im.draft('RGB',(3000,1500));im.thumbnail((4200,2100))
                rgb,_,_=reader.rectify(wall,cam,ground,np.asarray(im.convert('RGB')),base=zmin)
            images.append(np.rint(rgb*255).astype('uint8'))
            grays.append(cv2.cvtColor(images[-1],cv2.COLOR_RGB2GRAY))
        reg=aligned_pair(*grays)
        if reg is None:continue
        g=grays[0];warped=cv2.warpPerspective(grays[1],reg[0],(g.shape[1],g.shape[0]))
        for op in item['openings']:
            u0,v0,u1,v1=op['rect']
            x0,x1=np.rint(np.array([u0,u1])*width*PIXELS_PER_METRE).astype(int)
            y0,y1=np.rint((zmax-np.array([v1,v0])*height)*PIXELS_PER_METRE).astype(int)
            if not(0<=x0<x1<=g.shape[1] and 0<=y0<y1<=g.shape[0]):continue
            style=paired_style(g[y0:y1,x0:x1],warped[y0:y1,x0:x1])
            op['style'].update(style)
            count+=len(style)>1
            cv2.rectangle(images[0],(x0,y0),(x1,y1),(30,255,70),2)
        thumbs=[]
        for im in images:
            thumb=Image.fromarray(im);thumb.thumbnail((395,255));thumbs.append(thumb)
        reviews.append((len(item['openings']),item['id'],thumbs))
    review_sheet(reviews,review_path)
    packet['stats']['pairedPanes']=after
    stats=packet['stats']
    stats['nestedOrOverlappingCandidatesRemoved']=stats.get('nestedOrOverlappingCandidatesRemoved',0)+before-after
    packet['stats']['panesWithPairedSubdivisions']=count
    return count


def refine(area,source,output):
    os.environ['SFW_AREA']=str(area.resolve())
    import frontage_crops_v15 as fc
    import auto_read_frontages_v24 as reader
    reader.PPM=PIXELS_PER_METRE
    cv2.setNumThreads(1)
    center=json.loads((area/'config.json').read_text())['region']['center']
    cameras=fc.load_cameras(fc.LocalFrame(center['lon'],center['lat']))
    packet=json.loads(source.read_text())
    summary=Counter()
    details=[]
    @lru_cache(maxsize=2)
    def get_image(iid):
        with Image.open(cameras[iid]['path']) as f:
            f.draft('RGB',(3000,1500))
            f.thumbnail((4200,2100))
            return np.asarray(f.convert('RGB'))

    for profile in packet['profiles']:
        if not profile.get('sourceWallMidpointEN'):continue
        mid=np.asarray(profile['sourceWallMidpointEN']);n=np.asarray(profile['normal']);n=n/np.linalg.norm(n)
        tangent=np.array([-n[1],n[0]])
        length=float(profile['sourceWallLengthM']);height=float(profile['heightRefM'])
        material_w=min(length*.8,26);material_h=min(height*.88,23)
        center_u=mid@tangent
        profile.update(frameU=[round(center_u-material_w/2,3),round(center_u+material_w/2,3)],planeV=round(mid@n,3))
        reading=profile.setdefault('reading',{})
        reading.update(observedZRangeM=[1.1,round(min(height,1.1+material_h),3)],
                       visibleShare=round(material_w*material_h/(length*height),4))
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    frame=fc.LocalFrame(center['lon'],center['lat'])
    footprints=fc.load_footprints(frame)
    occluders=fc.Occluders(footprints);trees=fc.load_trees(frame)
    records=json.loads((area/'data/buildings.json').read_text())
    heights={'sf_'+r['sf16_bldgid'].replace('.','_'):float(r.get('hgt_median_m') or 0) for r in records}
    summary['sourceBuildings']=len(heights)
    walls=[]
    for bid,poly in footprints.items():
        h=heights.get(bid,0)
        if h<6:continue
        for part in getattr(poly,'geoms',[poly]):
            if not isinstance(part,Polygon):continue
            line=np.asarray(orient(part.simplify(.4),sign=1).exterior.coords)
            for a,b in zip(line,line[1:]):
                v=b-a;length=float(np.linalg.norm(v))
                if length<7:continue
                n=np.array([v[1],-v[0]])/length
                segments=math.ceil(length/52.)
                for i in range(segments):
                    walls.append((bid,a+v*(i+.5)/segments,n,length/segments,h))
    summary['examinedWallPortions']=len(walls)
    all_sources=json.loads((area/'imagery/source-index-registered.json').read_text())
    for wi,(bid,mid,n,length,height) in enumerate(walls):
        if wi%250==0:print('wall',wi,'of',len(walls),dict(summary),flush=True)
        tangent=np.array([-n[1],n[0]]);center_u=mid@tangent
        # Large views of distant tower crowns become too foreshortened to identify
        # panes. Sample the lower elevation and keep its metric height on output.
        base=3.0
        extent=min(height-base-.6,32)
        if extent<3:continue
        width=min(length*.96,53.)
        wall=dict(midpoint=mid,normal=n,lengthM=width,heightM=extent)
        views=fc.wall_candidates(wall,cameras,occluders,trees,54.)
        pool=[]
        for v in views:
            if v['incidence']<.60 or v['visibleFraction']<.83:continue
            cam=cameras[v['image']]
            if any(np.linalg.norm(cameras[o['image']]['xy']-cam['xy'])<3 for o in pool):continue
            pool.append(v)
            if len(pool)>=5:break
        if len(pool)<2:continue
        summary['facingsWithMultipleViews']+=1
        # Prioritize agreement within a mapped SfM cluster, while allowing
        # pairs across sequences when independently registered on this plane.
        pairs=[]
        for i,a in enumerate(pool):
            for b in pool[i+1:]:
                ca,cb=cameras[a['image']],cameras[b['image']]
                sep=np.linalg.norm(ca['xy']-cb['xy'])
                affinity=.18 if ca['clusterId'] and ca['clusterId']==cb['clusterId'] else 0.
                pairs.append((min(a['score'],b['score'])+affinity-.008*abs(sep-7),a['image'],b['image']))
        imgs={}; registered=False; wall_ground=fc.ground_at(mid) or 0.
        for _,a,b in sorted(pairs,reverse=True)[:4]:
            for iid in (a,b):
                if iid not in imgs:
                    cam=cameras[iid];cam['groundZ']=fc.ground_at(cam['xy'])
                    rgb,_,_=reader.rectify(wall,cam,wall_ground,get_image(iid),base=base)
                    imgs[iid]=np.rint(rgb*255).astype('uint8')
            boxes=paired_panes(imgs[a],imgs[b])
            registered|=boxes is not None
            if boxes is not None and len(boxes)>=3:break
        if registered:summary['registeredFacings']+=1
        if boxes is None or len(boxes)<3:continue
        summary['facingsWithPairedPanes']+=1
        summary['pairedPanes']+=len(boxes)
        # Store material and apertures as separate readings, as their measured
        # bounds differ. Only aperture profiles own opening placement.
        item=dict(status='image-measured-auto',match={'featureId':bid},normal=n.round(6).tolist(),
                  sourceIds=['mapillary:'+a,'mapillary:'+b],heightRefM=height,planeV=round(mid@n,3),
                  sourceWallMidpointEN=mid.round(3).tolist(),sourceWallLengthM=round(length,3))
        item.update(id=f'{bid}-paired-{wi}',groups=[],openings=[],
                    frameU=[round(center_u-width/2,3),round(center_u+width/2,3)],
                    reading=dict(coverageMode='positive-only',observedZRangeM=[base,round(base+extent,3)],
                                 visibleShare=round(width*extent/(length*height),4),
                                 method='two independent dark rectangular outlines with repeated sill or jamb'))
        for x,y,r,bottom in boxes:
            rect=[x/(PIXELS_PER_METRE*width),(base+extent-bottom/PIXELS_PER_METRE)/height,
                  r/(PIXELS_PER_METRE*width),(base+extent-y/PIXELS_PER_METRE)/height]
            item['openings'].append(dict(rect=np.round(rect,5).tolist(),style={'kind':'window'},evidence='paired-photo-outline'))
        details.append(item)
        for src in item['sourceIds']:
            rec=all_sources[src]
            packet['sources'][src]={'provider':'Mapillary','url':rec['sourceUrl'],'captureDate':rec['captureDate'],
                                    'registration':'provider pose with independent plane correspondence'}
    packet['profiles'].extend(details)
    packet['stats'].update(summary)
    packet['method']+=' Metric wall frames and explicit rectangular panes constrained by two-view registration and independent image boundaries.'
    measure_subdivisions(area,packet,output.with_suffix('.jpg'))
    save_packet(output,packet)
    print(packet['stats'],flush=True)

def save_packet(path,packet):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as partial:
        json.dump(packet,partial,indent=1)
    Path(partial.name).replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('area',type=Path);p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--measure-subdivisions-only',action='store_true');args=p.parse_args()
    if args.measure_subdivisions_only:
        packet=json.loads(args.input.read_text());print(measure_subdivisions(args.area,packet,args.output.with_suffix('.jpg')))
        save_packet(args.output,packet)
    else:refine(args.area,args.input,args.output)
