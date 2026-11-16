"""Compile fixed shopfront lettering observed at the same opening in two photos.
Transfer only the two-view-confirmed glyph outlines; never photograph the whole
window, infer a brand from an address, or extrude a board where none was read.
Outputs are profile candidates. The target glass depth is modeled, not surveyed.
"""
import argparse, hashlib, json, os
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import box as sbox
from pipeline.facade_grammar import opening_outline


def ink(path,word):
    im=cv2.imread(str(path));x,y,X,Y=np.array(word['boxPx']).round().astype(int)
    if y<0 or x<0 or X>im.shape[1] or Y>im.shape[0] or min(Y-y,X-x)<5:return
    rgb=im[y:Y,x:X,::-1];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);bg=np.median(np.r_[gray[0,:],gray[-1,:]])
    t,g=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if bg>t:g=255-g
    if not .07<g.mean()/255<.47:return
    fg=rgb[g>0];diff=abs(float(np.median(gray[g>0]))-float(bg))
    if diff<40:return
    return cv2.resize(g,(192,40),interpolation=cv2.INTER_LINEAR)>128, np.percentile(fg,70,axis=0).astype(int)


def agreement(a,b):
    # Require both images to independently contain the same actual letter shapes.
    def compare(a,b):
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
        da=cv2.dilate(a.astype('uint8'),k)>0;db=cv2.dilate(b.astype('uint8'),k)>0
        return min(np.mean(da[b]),np.mean(db[a]))
    # OCR engines include slightly different padding around the same glyphs.
    # Compensate that padding only in the verification, never in model width.
    def tight(m):
        ys,xs=np.where(m)
        if len(xs)==0 or (xs.max()-xs.min())<.82*m.shape[1] or (ys.max()-ys.min())<.68*m.shape[0]:return m
        c=m[ys.min():ys.max()+1,xs.min():xs.max()+1].astype('float32')
        return cv2.resize(c,(m.shape[1],m.shape[0]))>.5
    return max(compare(a,b),compare(tight(a),tight(b)))



def rings(mask):
    # Trace the literal photographed letter silhouettes, including their counters.
    aa=cv2.resize(mask.astype('uint8')*255,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
    c,ht=cv2.findContours((aa>108).astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if ht is None:return []
    def chain(i):
        p=cv2.approxPolyDP(c[i],.85,True)[:,0,:]
        return np.round(np.c_[p[:,0]/(aa.shape[1]-1),1-p[:,1]/(aa.shape[0]-1)],5).tolist()
    out=[]
    for i,h in enumerate(ht[0]):
        if h[3]>=0 or cv2.contourArea(c[i])<14:continue
        outer=chain(i);j=h[2];holes=[]
        while j>=0:
            ring=chain(j)
            if len(ring)>=3:holes.append(ring)
            j=ht[0,j,0]
        if len(outer)>=3:out.append({'outer':outer,'holes':holes})
    return out


def main():
    ap=argparse.ArgumentParser()
    for k in ('area','profiles','proposals','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();area=a.area.resolve();os.environ['SFW_AREA']=str(area)
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    from regen.reading_provenance import load_registered_sides
    _,fr=fc.load_world();reg=area/'imagery/registered-cameras.json';regs=json.loads(reg.read_text());cams=apply_registration(fc.load_cameras(fr),reg);sides=load_registered_sides(area/'regen/sides-reg')
    g=json.loads(a.profiles.read_text());ps={p['id']:p for p in g['profiles']};audit=[]
    for q in json.loads(a.proposals.read_text())['matched']:
        p=ps[q['side']];piece=sides[q['side'][6:]]['piece'];w,h=piece['lengthM'],piece['heightM'];words=[q['primary'],*q['witnesses']];sources=q['cropFiles'];record={'side':q['side'],'text':words[0]['text']};audit.append(record)
        inks=[ink(sources[s['image']],s) for s in words]
        if any(v is None for v in inks):record['skip']='unresolved stroke contrast';continue
        agree=min(agreement(inks[0][0],v[0]) for v in inks[1:]);record['glyphAgreement']=round(float(agree),3)
        if agree<.78:record['skip']='conflicting letter masks';continue
        options=[]
        for i,o in enumerate(p['openings']):
            if o['style'].get('kind')!='shopfront' or o.get('sourceIndex') is None:continue
            if not all(opening_outline(o,w,h).buffer(-.02).covers(sbox(*s['boxM'])) for s in words):continue
            options.append((i,o))
        if len(options)!=1:record['skip']='no unique measured shopfront host';continue
        idx,o=options[0];n=np.array(p['normal']);n=n/np.linalg.norm(n);u=np.array([-n[1],n[0]]);rec=float(o['style'].get('recessM',.2));bounds=[]
        for s in words:
            camera=cams[s['image']];cg=regs.get(s['image'],{});xy=np.asarray(camera['xy']);z=fc.ground_at(xy)+(cg.get('cameraHeightM',fc.EYE_HEIGHT_M) if camera.get('registered')=='skyline' else fc.EYE_HEIGHT_M)
            normal_distance=float(xy@n)-p['planeV']
            if normal_distance<=1:break
            focus=np.array([float(xy@u)-p['frameU'][0],z-piece['groundZ']]*2)
            bounds.append(focus+(np.array(s['boxM'])-focus)*(1+rec/normal_distance))
        if len(bounds)!=len(words):record['skip']='unsupported camera/host plane';continue
        b=np.median(bounds,axis=0);scatter=float(np.max(np.abs(np.array(bounds)-b)));record['positionDisagreementM']=round(scatter,3)
        region=opening_outline(o,w,h).buffer(-.04);boxes=[sbox(*bb) for bb in bounds]
        # Do not impose a fixed global centimetre cutoff: choose a surface only
        # when *every* reprojected box lies wholly on it and witnesses overlap
        # in both axes. Store the residual; this is approximate placement.
        intersection=boxes[0]
        for shape in boxes[1:]:intersection=intersection.intersection(shape)
        overlap=intersection.area/max(k.area for k in boxes)
        record['sharedAreaFraction']=round(overlap,3)
        if not all(region.covers(k) for k in boxes) or overlap<.40 or scatter>.25*(b[2]-b[0]):
            record['skip']='reprojected boxes do not securely occupy one common place on the opening';continue
        rg=rings(inks[0][0])
        if len(rg)<3:record['skip']='unreadable outline';continue
        col=np.median([v[1] for v in inks],axis=0).astype(int)
        sign={'mount':'opening-glass','sourceOpeningIndex':o['sourceIndex'],'rect':(b/[w,h,w,h]).tolist(),'text':words[0]['text'],'sourceGlyphs':rg,'textMaterial':'#%02x%02x%02x'%tuple(col),'sourceIds':['mapillary:'+s['image'] for s in words],'sourceImageHashes':[hashlib.sha256(Path(sources[s['image']]).read_bytes()).hexdigest() for s in words],'placementUncertaintyM':scatter,'note':'Literal observed glyph outlines; attached to modeled shopfront glazing. No backing board inferred.'}
        infill=o['style'].get('upperInfill',{});x0,z0,x1,z1=o['rect']
        border=(z0+infill.get('fromLeafZ',2.)*(z1-z0))*h
        if b[1]>=border and infill.get('kind')=='source-described-opaque-panel':
            sign['mount']='opening-infill';sign['note']='Observed letter silhouettes on a two-view source-supported opaque opening panel.'
        signs=p.setdefault('signs',[]);signs[:]=[q for q in signs if not (q.get('sourceOpeningIndex')==o['sourceIndex'] and q.get('text')==sign['text'] and q.get('sourceImageHashes')==sign['sourceImageHashes'])]
        signs.append(sign);record.update(accepted=True,feature=p['match']['featureId'],opening=idx,mount=sign['mount'])
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'profiles.json').write_text(json.dumps(g,separators=(',',':')));(a.output/'audit.json').write_text(json.dumps(audit,indent=1));print(audit)
if __name__=='__main__':main()
