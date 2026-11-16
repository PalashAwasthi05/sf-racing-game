"""Observed sash layout in original aperture coordinates, shared by every bay
shape. A side-hung window can be a single leaf; its width does not establish a
centre rail. Fixed mullions/transoms, drawn elsewhere, own their own tracks."""
from shapely.geometry import box

def inner_bars(bounds,style):
    x,z,X,Z=bounds;w,h=X-x,Z-z
    if min(w,h)<=0:return []
    sash=style.get('sash');lites=style.get('lites')
    if not sash and not lites:return []
    nx,ny=(1,1) if not lites else (max(1,min(12,int(v))) for v in lites)
    xs=[x+w*float(f) for f in (style.get('mullionFractions') or []) if 0<float(f)<1];zs=[z+h*float(f) for f in (style.get('transomFractions') or []) if 0<float(f)<1]
    out=[]
    def vertical(u,lo,hi,th,near):
        if not any(abs(v-u)<=.021+th/2 for v in xs):out.append((box(u-th/2,lo,u+th/2,hi),near))
    def horizontal(v,lo,hi,th,near):
        if not any(abs(q-v)<=.021+th/2 for q in zs):out.append((box(lo,v-th/2,hi,v+th/2),near))
    rows=[z,Z];cols=[x,X]
    if sash=='double-hung':
        m=z+h*min(.95,max(.05,float(style.get('meetingRail',.5))));horizontal(m,x,X,.06,.02);rows=[z,m,Z]
    elif sash=='sliding' and w>.5:
        c=(x+X)/2;vertical(c,z,Z,.06,.02);cols=[x,c,X]
    for a,b in zip(cols,cols[1:]):
        for c,d in zip(rows,rows[1:]):
            for k in range(1,nx):vertical(a+(b-a)*k/nx,c,d,.022,.05)
            for k in range(1,ny):horizontal(c+(d-c)*k/ny,a,b,.022,.05)
    return out
