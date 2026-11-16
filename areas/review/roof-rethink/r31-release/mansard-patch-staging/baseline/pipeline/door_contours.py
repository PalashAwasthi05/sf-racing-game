"""An observed garage assembly has one original leaf coordinate system.
A building segment may own only part of its aperture; clip every layer to that
part after layout and leave the leaf open when the source reported it open.
"""
import math
import re
from shapely.geometry import box


def draw_observed_garage(draw_leaf,prism,original,owned,style,recess,facing,ground):
    x,z,X,Z=original.bounds
    def clipped(cx,cz,w,h,d,material,depth):
        region=box(cx-w/2,cz-h/2,cx+w/2,cz+h/2).intersection(owned)
        if region.area>1e-6:prism(region,depth+d/2,d,material)
    draw_leaf(clipped,(x,z,X,Z),style,depth=-recess)
    # A mark is a faint repeatable tone, not a claim of a particular inset depth.
    evidence=style.get('observedLeafMarks') or {};mat=style.get('material','')
    if style.get('doorKind')=='panel' and len(set(evidence.get('sourceHashes',[])))>=2 and re.fullmatch('#[0-9A-Fa-f]{6}',mat):
        base=[int(mat[i:i+2],16) for i in (1,3,5)]
        for m in evidence.get('marks',[]):
            a,b=m['u'];zz=m['z'];wid=m['widthM'];t=m['darkening']
            if not (0<=a<b<=1 and 0<zz<1 and .003<=wid<=.05 and 0<t<=.35):continue
            tint='#'+''.join('%02x'%round(c*(1-t)) for c in base)
            # Plain opaque leaves are 8 cm deep, centered at -recess.
            poly=box(x+a*(X-x),z+zz*(Z-z)-wid/2,x+b*(X-x),z+zz*(Z-z)+wid/2).intersection(owned)
            if not poly.is_empty:prism(poly,-recess+.042,.002,tint)
    fw=float(style.get('frameWidthM',.06));frame=style.get('frameMaterial','charcoal')
    if fw<=0:return
    border=original.difference(original.buffer(-fw,join_style=2))
    # Remove the threshold but keep the complete original upper arch and jambs.
    rim=border.difference(box(x-fw,z-fw,X+fw,z+fw)).intersection(owned)
    if not rim.is_empty:
        prism(rim,-recess+.06,.04,frame)
        if recess>.35:prism(rim,0.,max(.01,recess-.055),facing)
    cx=(x+X)/2;floor_z=ground(cx,-recess/2)
    if recess>.35 and z<=floor_z+.25:
        clipped(cx,floor_z+.04,X-x,.08,max(.05,recess-.08),facing,-recess/2)
