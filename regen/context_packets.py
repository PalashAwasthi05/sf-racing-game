"""Build full-height source context without changing a facade's metric datum.

A rectified main-wall crop is not a whole-building inventory. These local-only
packets include the sloping street below it and roof context above it. They are
inputs for re-reading, never observations or proof of hidden geometry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np


def context_bounds(length, height, ground_relative, *, margin=1., max_extension=10.):
    if not all(math.isfinite(v) and v>0 for v in (length,height,max_extension)):
        raise ValueError('positive finite frame dimensions required')
    ground=np.asarray(ground_relative,dtype=float)
    finite=ground[np.isfinite(ground)]
    needed=max(3., -float(finite.min())+.5) if len(finite) else 3.
    below=min(max_extension,needed)
    above=min(8.,max(3.,height*.25))
    return {'xMinM':-margin,'xMaxM':length+margin,'zMinM':-below,'zMaxM':height+above,
            'groundCoverageClipped':needed>max_extension,'groundSamplesAvailable':bool(len(finite)),
            'originalWidthM':length,'originalHeightM':height}


def native_rect_ppm(wall, camera_xyz, z0, bounds, pano_size, max_pixels=1600):
    """Conservative source angular sampling; never choose detail from a fixed upsample."""
    n=np.asarray(wall['normal'],dtype=float);n/=np.linalg.norm(n)
    t=np.array([-n[1],n[0],0.]); vertical=np.array([0.,0.,1.])
    midpoint=np.r_[wall['midpoint'],float(z0)]
    camera=np.asarray(camera_xyz,dtype=float)
    angular_pixels=min(pano_size[0]/(2*np.pi),pano_size[1]/np.pi)
    samples=[]
    for x in np.linspace(bounds['xMinM'],bounds['xMaxM'],5):
        for z in np.linspace(bounds['zMinM'],bounds['zMaxM'],5):
            point=midpoint+t*(x-bounds['originalWidthM']/2)+vertical*z
            ray=point-camera; distance=np.linalg.norm(ray)
            if distance<.5:continue
            direction=ray/distance
            samples.append(min(np.linalg.norm(axis-direction*np.dot(axis,direction)) for axis in (t,vertical))*angular_pixels/distance)
    if not samples:raise ValueError('camera coincides with context plane')
    native=float(np.quantile(samples,.1))
    extent=max(bounds['xMaxM']-bounds['xMinM'],bounds['zMaxM']-bounds['zMinM'])
    return max(1.,min(128.,native,max_pixels/extent)), {'sourcePpmP10':native,'sourcePpmMin':float(min(samples)),'sourcePpmMax':float(max(samples))}


def pixel_to_wall(x,y,bounds,ppm):
    return bounds['xMinM']+(x+.5)/ppm, bounds['zMaxM']-(y+.5)/ppm


def main():
    ap=argparse.ArgumentParser()
    for name in ('area','jobs','sides','calibration','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    os.environ['SFW_AREA']=str(args.area.resolve())
    import frontage_crops_v15 as fc
    import auto_read_frontages_v24 as ar
    from regen.register_cameras import apply_registration
    from regen.reading_provenance import load_registered_sides
    from areas.ring.poses import side_camera_geometry
    from panorama_balance import road_gain
    from PIL import Image,ImageDraw
    if args.output.exists():ap.error('use a new output directory')
    jobs=json.loads(args.jobs.read_text());jobs=jobs.get('poses',[]) if isinstance(jobs,dict) else jobs
    sides=load_registered_sides(args.sides)
    calibration=json.loads(args.calibration.read_text())['images']
    _,frame=fc.load_world()
    cameras=apply_registration(fc.load_cameras(frame),args.area/'imagery/registered-cameras.json')
    (args.output/'img').mkdir(parents=True)
    records=[]
    for job in jobs:
        sid=job['side']; source=sides.get(sid); item={'side':sid,'image':job['image'],'status':'unresolved'}
        records.append(item)
        if not source or source['status']!='ok':item['reason']='ambiguous-source-side';continue
        p=source['piece'];vs=[v for v in p['views'] if v['image']==job['image']]
        if len(vs)!=1 or job['image'] not in calibration:item['reason']='ambiguous-source-camera';continue
        v=vs[0];cam=cameras.get(v['image']);cal=calibration[v['image']]
        if not cam or not cam.get('exists'):item['reason']='panorama-unavailable';continue
        digest=hashlib.sha256(cam['path'].read_bytes()).hexdigest()
        if digest!=cal['sourceSha256'] or not np.allclose(cam['R'],cal['referenceRotation']):
            item['reason']='camera-or-source-hash-mismatch';continue
        if cam.get('registered',False)!=v.get('registered',False):item['reason']='camera-placement-mode-mismatch';continue
        wall=side_camera_geometry(p);n=np.asarray(wall['normal']);t=np.array([-n[1],n[0]])
        points=[wall['midpoint']+t*x+n*y for x in np.linspace(-p['lengthM']/2,p['lengthM']/2,9) for y in (0.,1.5,3.,6.,9.)]
        ground=[fc.ground_at(q)-p['groundZ'] for q in points]
        # On raised front gardens the garage meets the street several metres
        # beyond the host wall; include the photographed street datum too.
        ground.append(float(v['cameraGroundZ'])-float(p['groundZ']))
        bounds=context_bounds(float(p['lengthM']),float(p['heightM']),ground)
        extended={**wall,'lengthM':bounds['xMaxM']-bounds['xMinM'],'heightM':bounds['zMaxM']-bounds['zMinM']}
        camera={**cam,'R':np.asarray(cal['rotation']), 'groundZ':v['cameraGroundZ']}
        pano=np.asarray(fc.load_pano(cam['path']),np.float32)
        ar.PPM,sampling=native_rect_ppm(wall,[*camera['xy'],camera['groundZ']+fc.EYE_HEIGHT_M],p['groundZ'],bounds,(pano.shape[1],pano.shape[0]))
        pano=np.clip(pano*road_gain(pano),0,255).astype(np.uint8)
        raw,_,_=ar.rectify(extended,camera,p['groundZ'],pano,base=bounds['zMinM'])
        image=Image.fromarray(np.clip(raw*255,0,255).astype(np.uint8))
        name=sid+'__context.png';image.save(args.output/'img'/name)
        grid=image.copy();draw=ImageDraw.Draw(grid)
        for z in np.arange(math.ceil(bounds['zMinM']),math.floor(bounds['zMaxM'])+1):
            y=(bounds['zMaxM']-z)*ar.PPM-.5
            draw.line((0,y,image.width,y),fill=(180,180,180),width=1)
            draw.text((2,y+2),f'{z:g}m',fill=(255,40,0))
        for x in np.arange(math.ceil(bounds['xMinM']),math.floor(bounds['xMaxM'])+1):
            px=(x-bounds['xMinM'])*ar.PPM-.5
            draw.line((px,0,px,image.height),fill=(180,180,180),width=1)
        for z,col in ((0.,(255,0,0)),(float(p['heightM']),(255,160,0))):
            y=(bounds['zMaxM']-z)*ar.PPM-.5;draw.line((0,y,image.width,y),fill=col,width=2)
        grid.save(args.output/'img'/name.replace('.png','-grid.png'))
        item.update(status='prepared-context-only',bounds=bounds,ppm=ar.PPM,sampling=sampling,rect=str(args.output/'img'/name),
                    sourceGroundZ=float(p['groundZ']),frameU=p['frameU'],planeV=p['planeV'],normal=p['normal'],
                    cameraEyeAboveGroundM=float(fc.EYE_HEIGHT_M),
                    originalPerspective=job['persp'],sourcePath=str(cam['path']),sourceSha256=digest,
                    cameraXY=cam['xy'].tolist(),cameraGroundZ=v['cameraGroundZ'],rotation=cal['rotation'],
                    sourceRegistration=cam.get('registered',False),absolutePositionValidated=False,
                    pixelMapping='x=xMin+(column+.5)/ppm; z=zMax-(row+.5)/ppm; original facade datum unchanged')
        print(sid,item['status'],flush=True)
    (args.output/'packets.json').write_text(json.dumps({'schema':'facade-context-2','method':'street-to-roof-context-in-original-wall-frame',
        'productionIntegration':False,'observationsCreated':0,'packets':records},indent=2))


if __name__=='__main__':main()
