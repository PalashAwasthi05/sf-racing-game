"""Rectified frontage images: the wall plane seen square-on, with a metre grid, for measured readings.

A perspective crop lets a reader recognise a facade but not measure it. This renders, for each frontage in a
questionnaire, an orthographic image of its wall plane: pixel column = metres along the frontage from its left
end, pixel row = metres above the wall base, sampled from a cached panorama through the provider camera pose.
A 1 m grid (heavy every 5 m), axis labels and the wall-piece boundaries are drawn on, so an opening's edges are
read off as metres and become fractions of W and H directly. Occluders (trees, vehicles) smear onto the plane and
must be recognised as such by the reader. General: nothing here knows any building; identities are join keys.
"""
import argparse,json,pathlib,sys
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc

WORK=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--questionnaire',required=True);ap.add_argument('--ppm',type=float,default=40.,help='pixels per metre (reduced for long frontages)')
ap.add_argument('--max-width',type=int,default=2600);ap.add_argument('--views',type=int,default=2,help='panoramas per frontage');ap.add_argument('--out',type=pathlib.Path,default=WORK/'facade-priority-v15/crops/rectified')
a=ap.parse_args()
q=json.load(open(a.questionnaire));world,frame=fc.load_world();buildings={b['id']:b for b in world['buildings']};cams=fc.load_cameras(frame)
a.out.mkdir(parents=True,exist_ok=True)
def render(entry,cam,ppm):
    n=np.asarray(entry['normal'],float);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]]);fr=entry['frame'];left=fr['left'];W=fr['W']
    plane=float(np.mean([np.asarray(p['midpoint'])@n for p in entry['walls']]))
    z0=buildings[entry['buildingId']]['center'][2];Hm=max(entry['wallHeightsM']);Hm=min(max(Hm,4.),buildings[entry['buildingId']]['height']+1.)
    width=int(round(W*ppm));height=int(round(Hm*ppm))
    xs=(np.arange(width)+.5)/ppm;zs=Hm-(np.arange(height)+.5)/ppm
    P=np.empty((height,width,3));P[...,0]=u[0]*(left+xs)[None,:]+n[0]*plane;P[...,1]=u[1]*(left+xs)[None,:]+n[1]*plane;P[...,2]=(z0+zs)[:,None]
    C=np.array([cam['xy'][0],cam['xy'][1],z0+fc.EYE_HEIGHT_M]);dirs=P-C
    # a panorama behind the wall plane sees its back; refuse (the selector never picks those, this is a guard)
    if np.dot(C[:2]-np.array([u[0]*(left+W/2)+n[0]*plane,u[1]*(left+W/2)+n[1]*plane]),n)<=0:return None,None
    uu,vv=fc.world_dirs_to_pano_uv(dirs,cam['R']);pano=fc.load_pano(cam['path']);ph,pw=pano.shape[:2]
    px=(uu*pw-.5)%pw;py=np.clip(vv*ph-.5,0,ph-1);out=np.empty((height,width,3),np.uint8)
    for c in range(3):out[...,c]=map_coordinates(pano[...,c],np.stack([py,px]),order=1,mode='grid-wrap').astype(np.uint8)
    img=Image.fromarray(out).convert('RGB');d=ImageDraw.Draw(img)
    for m in range(int(W)+1):
        x=int(round(m*ppm));heavy=m%5==0;d.line([(x,0),(x,height)],fill=(255,255,0) if heavy else (255,255,255),width=2 if heavy else 1)
        if heavy:d.text((x+3,height-14),f'{m} m',fill=(255,255,0))
    for m in range(int(Hm)+1):
        y=int(round(height-m*ppm));heavy=m%5==0;d.line([(0,y),(width,y)],fill=(255,255,0) if heavy else (255,255,255),width=2 if heavy else 1)
        d.text((3,max(0,y-12)),f'{m}',fill=(255,255,0))
    for p in entry['walls']:
        r=p.get('xFractionRange')
        if not r:continue
        for f in r:
            x=int(round(f*W*ppm));d.line([(x,0),(x,height)],fill=(255,0,0),width=3)
        d.text((int(round(r[0]*W*ppm))+4,4),f"piece e{p['edge']} H={p['heightM']} m",fill=(255,80,80))
    d.text((width-260,4),f"{entry['buildingId']} {cam['id'] if 'id' in cam else ''} W={W:.1f} m",fill=(255,255,0))
    return img,{'widthM':round(W,2),'heightM':round(Hm,2),'ppm':ppm,'plane':round(plane,2),'z0':round(z0,2)}
for e in q:
    images=[];[images.append(c['image']) for c in sorted(e['crops'],key=lambda c:(c['viewRank'],c['distanceM'])) if c['image'] not in images]
    e['rectified']=[]
    for img_id in images[:a.views]:
        cam=cams.get(img_id)
        if cam is None or not cam.get('exists',True):continue
        ppm=min(a.ppm,a.max_width/max(e['frame']['W'],1.))
        img,meta=render(e,cam,ppm)
        if img is None:continue
        path=a.out/f"{e['profileIdHint']}__{img_id}.jpg";img.save(path,quality=88)
        e['rectified'].append({'path':str(path),'image':img_id,**meta,'cameraDistanceM':next((c['distanceM'] for c in e['crops'] if c['image']==img_id),None)})
pathlib.Path(a.questionnaire).write_text(json.dumps(q,indent=1))
print(json.dumps([{ 'frontage':e['profileIdHint'],'rectified':[r['path'].split('/')[-1]+f" ({r['widthM']}x{r['heightM']} m, {r['ppm']:.0f} px/m, cam {r['cameraDistanceM']} m)" for r in e['rectified']]} for e in q],indent=1))
