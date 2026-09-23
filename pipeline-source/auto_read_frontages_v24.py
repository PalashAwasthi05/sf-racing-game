"""Automatic frontage reading: every street-facing wall a cached panorama sees is rendered square-on and measured.

For each facade wall of a generated candidate (its evidence file lists wall midpoints, normals, lengths, heights),
the best camera is chosen with the crop module's own visibility and occlusion rules, the wall plane is rendered
orthographically (pixel column = metres along the wall, row = metres above its base), and three things are measured:
 * the wall material: the dominant colour of the non-opening pixels, mapped to a material family and kept as the
   observed colour (recipe#rrggbb);
 * the window grid: dark openings against that wall colour, their row and column periodicity giving floor pitch,
   bay pitch and opening proportions (only accepted when the periodicity is clear);
 * a shopfront ground floor: a large dark share of the lowest storey.
The result is one profile per building facing, marked `image-measured-auto`, written as an observation packet the
generator reads with the hand readings (hand readings of the same facing take precedence). Nothing here knows any
building or place; identities are join keys only. Usage:
  auto_read_frontages_v24.py --candidate work/clean-v25 --out work/pipeline/frontage-observations-auto.json
"""
import argparse,json,math,pathlib,re,sys,colorsys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,uniform_filter1d
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc

PPM=12.        # pixels per metre of the rectified wall
MIN_WALL_M=6.
def rectify(wall,cam,z0,pano):
    m=np.asarray(wall['midpoint'],float);n=np.asarray(wall['normal'],float);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])
    L=float(wall['lengthM']);H=float(wall['heightM'])
    W=max(8,int(L*PPM));Hp=max(8,int(H*PPM))
    xs=(np.arange(W)+.5)/PPM-L/2;zs=H-(np.arange(Hp)+.5)/PPM
    P=np.empty((Hp,W,3));P[...,0]=(m[0]+t[0]*xs)[None,:]+n[0]*.05;P[...,1]=(m[1]+t[1]*xs)[None,:]+n[1]*.05;P[...,2]=(z0+zs)[:,None]
    C=np.array([cam['xy'][0],cam['xy'][1],z0+fc.EYE_HEIGHT_M]);d=P-C
    u,v=fc.world_dirs_to_pano_uv(d,cam['R']);ph,pw=pano.shape[:2]
    img=np.stack([map_coordinates(pano[...,k],[v*(ph-1),u*(pw-1)],order=1,mode='wrap',output=np.float32) for k in range(3)],-1)
    return img/255.,H,L
def period(profile,lo,hi):
    """Strongest autocorrelation period (in samples) between lo and hi, with its correlation."""
    p=profile-profile.mean()
    if p.std()<1e-4 or hi<=lo+1:return None,0.
    ac=np.correlate(p,p,'full')[len(p)-1:];ac/=ac[0]
    lo,hi=int(max(2,lo)),int(min(len(ac)-2,hi))
    if hi<=lo:return None,0.
    k=lo+int(np.argmax(ac[lo:hi+1]));return k,float(ac[k])
def family(rgb):
    h,l,s=colorsys.rgb_to_hls(*[float(x) for x in rgb]);hd=h*360
    if l<.2:return ('metal-panel','dark')
    # weathered brick under overcast or shadow keeps a warm hue at low saturation
    if s>=.09 and (hd<26 or hd>345) and .16<=l<.68:return ('brick','dark' if l<.3 else 'default')
    if s>=.15 and 300<=hd<=345 and .16<=l<.68:return ('brick','dark' if l<.3 else 'default')   # a magenta cast on red brick
    if s>=.12 and 25<=hd<58 and l>=.42:return ('render','salmon' if hd<35 and s>.28 else 'default')
    if l>.7 and s<.2:return ('render','white')
    if s>=.25 and (hd<30 or hd>330):return ('brick','default')   # a saturated warm wall reads as brick, not painted render
    return ('render','gray')
def kmeans(x,k=4,iters=12,seed=0):
    rng=np.random.default_rng(seed);c=x[rng.choice(len(x),k,replace=False)]
    for _ in range(iters):
        lab=np.argmin(((x[:,None,:]-c[None])**2).sum(-1),1)
        for j in range(k):
            if (lab==j).any():c[j]=x[lab==j].mean(0)
    return c,lab
def fold(profile,per):
    """Mean of a profile over one period (phase-aligned), normalised to [0,1]."""
    per=int(per);m=len(profile)//per*per
    if m<per*2:return None
    f=profile[:m].reshape(-1,per).mean(0);f=f-f.min()
    return f/f.max() if f.max()>0 else None
def read(img,H,L,exclude=None):
    lum=img@np.array([.299,.587,.114]);r,g,b=img[...,0],img[...,1],img[...,2]
    Hp=img.shape[0];upper=slice(0,max(1,int(Hp*.85)))  # the lowest ~15 % carries cars, people and street clutter
    sky=(b>r+.06)&(lum>.55);leaf=(g>r+.03)&(g>b+.01)
    ok=~sky&~leaf;ok[int(Hp*.85):]=False
    if exclude is not None and exclude.shape==ok.shape:ok&=~exclude   # pixels another building hides (frontage_evidence_v34)
    px=img[ok];res={'skyShare':float(sky[upper].mean()),'foliageShare':float(leaf[upper].mean())}
    if len(px)<200:res.update(wallRgb=(np.median(img.reshape(-1,3),0)*255).astype(int).tolist(),family=('render','gray'),shopfront=False,darkShare=0.);return res
    sample=px[np.random.default_rng(1).choice(len(px),min(len(px),6000),replace=False)]
    cent,lab=kmeans(sample);share=np.bincount(lab,minlength=len(cent))/len(lab);cl=cent@np.array([.299,.587,.114])
    order=np.argsort(-share)
    # the dominant cluster, unless it is the dark glass cluster of a facade that is mostly openings
    wall=next((j for j in order if not (cl[j]<.2 and share[j]<.55)),order[0])
    # Frames and glass are the openings; on a brick facade with big pale frames they can outweigh the brick. A
    # saturated warm cluster covering a fifth of the wall is the wall.
    for j in order:
        hj,lj,sj=colorsys.rgb_to_hls(*[float(x) for x in cent[j]])
        if share[j]>=.2 and sj>=.12 and (hj*360<26 or hj*360>345) and .15<=lj<.7:wall=j;break
    wall_rgb=cent[wall];wall_l=float(cl[wall])
    res.update(wallRgb=(wall_rgb*255).round().astype(int).tolist(),family=family(wall_rgb),wallShare=round(float(share[wall]),2))
    # window grid from periodic edges: jambs make vertical edges per column, sills and heads horizontal edges per row
    gx=np.abs(np.diff(lum,axis=1,prepend=lum[:,:1]))*ok;gy=np.abs(np.diff(lum,axis=0,prepend=lum[:1]))*ok
    g=int(min(H,4.5)*PPM);band=slice(0,max(1,Hp-g))
    rows=uniform_filter1d(gy[band].mean(1),3);cols=uniform_filter1d(gx[band].mean(0),3)
    if Hp-g>=3*PPM and L>=MIN_WALL_M:
        fp,fcor=period(rows,2.8*PPM,5.5*PPM);bp,bcor=period(cols,1.2*PPM,min(9*PPM,len(cols)/2))
        if fp and fcor>.15 and not (bp and bcor>.15):bp,bcor=int(3.0*PPM),.15   # storeys clear, bays unclear: a 3 m bay
        if fp and bp and fcor>.15 and bcor>=.15:
            # opening height: share of the storey where jamb edges are strong; width: share of the bay where
            # sill/head edges are strong
            fr=fold(uniform_filter1d(gx[band].mean(1),3),fp);fc_=fold(uniform_filter1d(gy[band].mean(0),3),bp)
            hr=float(np.clip((fr>.35).mean(),.3,.85)) if fr is not None else .6;wr=float(np.clip((fc_>.35).mean(),.25,.85)) if fc_ is not None else .5
            res.update(floorPitchM=round(fp/PPM,2),bayPitchM=round(bp/PPM,2),openingRatio=round(wr,2),openingHeightRatio=round(hr,2),gridConfidence=round(min(fcor,bcor),2))
    dark=(lum<wall_l-.15)&~leaf
    up=dark[band][ok[band]].mean() if ok[band].any() else 0.
    gb=slice(max(0,Hp-g),max(max(0,Hp-g)+1,int(Hp-.6*PPM)))
    ground=dark[gb].mean() if dark[gb].size else 0.
    res['darkShare']=round(float(up),2);res['groundDarkShare']=round(float(ground),2)
    res['shopfront']=bool(ground>.5 and ground>up+.2 and leaf[gb].mean()<.25)
    return res
def profile_for(bid,building,wall,reading,img_id):
    fam,fin=reading['family'];src=['mapillary:'+img_id]
    if fam=='brick':
        # brick is red-to-orange: a cast outside that hue range is removed, keeping the measured lightness
        h,l,sat=colorsys.rgb_to_hls(*[v/255 for v in reading['wallRgb']]);hd=h*360
        hd=8. if hd>200 else min(max(hd,6.),25.)
        reading['wallRgb']=[int(round(v*255)) for v in colorsys.hls_to_rgb(hd/360,l,min(sat,.55))]
    if fam=='render' and fin in ('gray','white'):
        # a grey or white render is near-neutral: a strong tint left after white balance is residual cast, not paint
        h,l,sat=colorsys.rgb_to_hls(*[v/255 for v in reading['wallRgb']])
        reading['wallRgb']=[int(round(v*255)) for v in colorsys.hls_to_rgb(h,l,min(sat,.08))]
    p={'id':f"{bid}-auto-{int(round(math.degrees(math.atan2(wall['normal'][1],wall['normal'][0]))))%360}",'status':'image-measured-auto',
       'match':{'featureId':re.sub(r'_p\d+$','',bid),'osmWayId':(re.search(r'openstreetmap.org/way/(\d+)',building.get('sourceUrl','')) or [None,None])[1]},'normal':[round(float(x),3) for x in wall['normal']],
       'sourceIds':src,'heightRefM':round(float(wall['heightM']),2),
       'materialEvidence':{'family':fam,'finish':fin,'colorRgb':reading['wallRgb'],'sourceIds':src},'groups':[]}
    H=float(wall['heightM']);g=min(4.5,H*.35)/H
    if reading.get('floorPitchM'):
        style={'kind':'window','openingRatio':reading['openingRatio'],'openingHeightRatio':reading['openingHeightRatio'],'mullionFractions':[.5] if reading['bayPitchM']*reading['openingRatio']>1.8 else [],'transomFractions':[]}
        p['groups'].append({'region':[0.,g,1.,.97],'columnPitchM':reading['bayPitchM'],'rowPitchM':reading['floorPitchM'],'style':style})
        gstyle=({'kind':'shopfront','openingRatio':.85,'openingHeightRatio':.8,'mullionFractions':[.5],'transomFractions':[.8]} if reading['shopfront']
                else {**style,'kind':'window'})
        p['groups'].append({'region':[0.,.02,1.,g],'columnPitchM':reading['bayPitchM'],'rows':1,'style':gstyle})
    if not p['groups']:p['materialOnly']=True   # colour measured, openings not: the wall keeps its prior layout
    return p
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--max-walls',type=int,default=400);a=ap.parse_args()
    world=json.load(open(a.candidate/'public/world.json'));frame=fc.LocalFrame(world['center']['lon'],world['center']['lat'])
    buildings={b['id']:b for b in world['buildings']}
    ev=json.load(open(a.candidate/'public/facade-evidence.json'))
    walls=[w for w in ev['walls'] if w.get('lengthM',0)>=MIN_WALL_M and w.get('heightM',0)>=3 and w.get('buildingId') in buildings and w.get('status')!='envelope-graph']
    occ=fc.Occluders(fc.load_footprints(frame));trees=fc.load_trees(frame);cams=fc.load_cameras(frame)
    best={}   # (building, facing) -> (score, wall, cand)
    for w in walls:
        cands=fc.wall_candidates(w,cams,occ,trees,fc.MAX_DISTANCE_M)
        if not cands:continue
        c=cands[0];key=(w['buildingId'],int(round(math.degrees(math.atan2(w['normal'][1],w['normal'][0]))/10)))
        score=c['score']*w['lengthM']
        if key not in best or score>best[key][0]:best[key]=(score,w,c)
    # A hand reading of the same building facing takes precedence: those facings are not read automatically.
    hand=[]
    for f in sorted((pathlib.Path(__file__).resolve().parent/'pipeline').glob('frontage-observations*.json'))+[pathlib.Path(__file__).resolve().parent/'pipeline/facade-profiles.json']:
        if f.resolve()==a.out.resolve():continue
        hand+=json.load(open(f)).get('profiles',[])
    def covered(bid,normal):
        fid=re.sub(r'_p\d+$','',bid)
        return any(p['match'].get('featureId')==fid and np.dot(normal,p['normal'])>.95 for p in hand)
    best={k:v for k,v in best.items() if not covered(v[1]['buildingId'],v[1]['normal'])}
    chosen=sorted(best.values(),key=lambda x:-x[0])[:a.max_walls];profiles=[];stats={'wallsConsidered':len(walls),'facings':len(best),'read':0,'withGrid':0,'shopfronts':0}
    pano_cache={}
    for score,w,c in chosen:
        cam=cams[c['image']];bid=w['buildingId'];z0=buildings[bid]['center'][2]
        if c['image'] not in pano_cache:
            if len(pano_cache)>=3:pano_cache.pop(next(iter(pano_cache)))
            pano=np.asarray(fc.load_pano(cam['path']),dtype=np.float32)
            # Grey-world white balance per panorama: shade and overcast skew a capture cool (walls read violet or
            # blue); scaling each channel so the whole image averages neutral removes the cast, not the wall colours.
            # The neutral reference is the road surface just below the camera (asphalt and concrete are grey); a
            # grey-world average over walls would treat sunlit brick as a cast and grey it. Each channel is corrected
            # by at most 15 %.
            road=pano[int(pano.shape[0]*.74):int(pano.shape[0]*.9):4,::4].reshape(-1,3)
            road=road[(road.max(1)-road.min(1))<40]   # keep near-grey pixels: paint lines and kerb colours drop out
            mean=road.mean(0) if len(road)>500 else np.full(3,128.)
            gain=np.clip(mean.mean()/np.maximum(mean,1.),.85,1.15)
            pano_cache[c['image']]=np.clip(pano*gain,0,255).astype(np.uint8)
        img,H,L=rectify(w,cam,z0,pano_cache[c['image']]);r=read(img,H,L)
        p=profile_for(bid,buildings[bid],w,r,c['image']);p['reading']=r;p['camera']={'image':c['image'],'distanceM':c['distanceM'],'incidence':c['incidence']};profiles.append(p)
        stats['read']+=1;stats['withGrid']+=bool(r.get('floorPitchM'));stats['shopfronts']+=r['shopfront']
    a.out.write_text(json.dumps({'schemaVersion':1,'method':'automatic rectified reading (auto_read_frontages_v24)','profiles':profiles,'stats':stats},indent=1))
    print(json.dumps(stats))
if __name__=='__main__':main()
