"""Associate a partially obscured repeat only with an actually readable word
and independent glyph evidence at the same physical position. A shared brand
elsewhere on the facade is insufficient."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from regen.locate_facade_words import candidates,same
from regen.reading_provenance import load_registered_sides
from regen.compile_opening_lettering import ink,agreement
def witness(primary,partial,source_img,target_img,piece,v):
    w=primary['key'];s=partial['key']
    if w==s or len(s)<4 or len(s)/len(w)<.70 or (not w.startswith(s) and not w.endswith(s)):return
    x,y,X,Y=primary['boxPx'];a,b,A,B=partial['boxPx'];height=Y-y
    if not .75<(B-b)/height<1.35:return
    left=(A-(X-x)) if w.endswith(s) else a
    px=[left,b,left+(X-x),B]
    if px[0]<0 or px[2]>=target_img.shape[1]:return
    pad,ppm=v['rectPad'],v['ppm'];h=piece['heightM']
    other={**partial,'key':w,'boxM':[(px[0]-pad)/ppm,h-(B-pad)/ppm,(px[2]-pad)/ppm,h-(b-pad)/ppm],'boxPx':px,'partialWitness':s}
    if not same(primary,other):return
    # Learn the exact fragment's position in the primary glyphs (OCR padding
    # and nonuniform letter widths rule out character-count extrapolation).
    x,y,X,Y=np.rint([x,y,X,Y]).astype(int);a,b,A,B=np.rint([a,b,A,B]).astype(int)
    src=source_img[y:Y,x:X];t=target_img[b:B,a:A]
    if min(*src.shape,*t.shape)<5:return
    h0,w0=src.shape
    for factor in (1.,.94,1.06):
        w1=round(t.shape[1]*h0/t.shape[0]*factor)
        if not .68*w0<w1<w0:continue
        tt=cv2.resize(t,(w1,h0));rr=cv2.matchTemplate(src,tt,cv2.TM_CCOEFF_NORMED);offset=int(rr.argmax())
        if rr.max()<.66 or (offset>.08*w0 if w.startswith(s) else offset+w1<.92*w0):continue
        scale=t.shape[1]/w1;L=a-offset*scale;R=L+w0*scale
        if L<1 or R>=target_img.shape[1]-1:continue
        other.update(boxPx=[L,b,R,B],boxM=[(L-pad)/ppm,h-(B-pad)/ppm,(R-pad)/ppm,h-(b-pad)/ppm],partialGlyphCorrelation=float(rr.max()))
        if same(primary,other):return other
def main():
 ap=argparse.ArgumentParser()
 for k in ('area','ocr','proposals','output'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();existing=json.loads(a.proposals.read_text());rows={r['path']:r for r in map(json.loads,a.ocr.read_text().splitlines())};sides=load_registered_sides(a.area/'regen/sides-reg')
 extra=[]
 for side in sorted({r['side'] for r in existing['matched']}):
    row=sides[side[6:]];piece=row['piece'];views=piece['views'];files={v['image']:str((row['path'].parent/'img'/v['rect']).resolve()) for v in views}
    readings={v['image']:candidates(rows[files[v['image']]],v,piece['lengthM'],piece['heightM']) for v in views if files[v['image']] in rows}
    words={q['primary']['key'] for q in existing['matched'] if q['side']==side}
    primary=next(q['primary']['image'] for q in existing['matched'] if q['side']==side)
    for w in readings.get(primary,[]):
        if w['key'] not in words or any(q['primary']==w for q in existing['matched']):continue
        ws=[]
        for v in views:
            if v['image']==primary:continue
            for w2 in readings.get(v['image'],[]):
                r=witness(w,w2,cv2.imread(files[primary],0),cv2.imread(files[v['image']],0),piece,v)
                if r:ws.append(r)
        if ws:extra.append(dict(side=side,primary=w,witnesses=ws,cropFiles=files))
 existing['matched'].extend(extra);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(existing,indent=1));print('confirmed partial words:',len(extra))
if __name__=='__main__':main()
