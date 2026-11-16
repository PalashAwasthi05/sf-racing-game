"""Bake observed letter contours onto the inner face of their own storefront.
The glazing is the modeled support; no board, extrusion or new opening is
manufactured to make a sign fit. Missing/occluded host = no lettering there.
"""
from shapely import affinity
from shapely.geometry import Polygon, box


def parts(sign,width,height,openings):
    if sign.get('mount') not in ('opening-glass','opening-infill') or len(set(sign.get('sourceImageHashes',[])))<2:return []
    target=sign.get('sourceOpeningIndex')
    if not isinstance(target,int):return []
    x,z,X,Z=[a*b for a,b in zip(sign.get('rect',[]),(width,height,width,height))]
    if not(x<X and z<Z):return []
    out=[]
    for o in openings:
        if o.get('sourceIndex')!=target or (o.get('style') or {}).get('kind')!='shopfront':continue
        st=o['style'];r=st.get('recessM',.2);fw=float(st.get('frameWidthM',.06))
        outer=o['_originalOutline'];owned=outer.buffer(-fw,join_style=2).intersection(o['_outline'])
        if owned.is_empty or not outer.covers(box(x,z,X,Z)):continue
        # Contour prisms end precisely at -recess. Rectangular pane cuboids
        # have their center at -recess and their front is 12.5 mm nearer.
        front=-r+(.004 if o.get('_contoured') else .0165)
        if sign['mount']=='opening-infill':
            panels=infill_parts(o)
            if not panels:continue
            region,depth,_,_=panels[0];owned=owned.intersection(region);front=depth+.003
        for glyph in sign.get('sourceGlyphs',[]):
            shape=Polygon(glyph['outer'],glyph.get('holes',[]))
            if not shape.is_valid:shape=shape.buffer(0)
            geom=affinity.affine_transform(shape,[X-x,0,0,Z-z,x,z]).intersection(owned)
            if geom.area>.00003:out.append((geom,front,.002,sign['textMaterial']))
    return out


def infill_parts(o):
    st=o.get('style',{});item=st.get('upperInfill') or {};fraction=item.get('fromLeafZ')
    if (st.get('kind')!='shopfront' or item.get('kind')!='source-described-opaque-panel'
        or len(set(item.get('sourceHashes',[])))<2 or not isinstance(fraction,(int,float)) or not 0<fraction<1):return []
    original=o['_originalOutline'];x,z,X,Z=original.bounds
    owned=original.buffer(-float(st.get('frameWidthM',.06)),join_style=2).intersection(o['_outline'])
    panel=owned.intersection(box(x,z+fraction*(Z-z),X,Z))
    if panel.is_empty:return []
    # Mounted inside the existing casing, just ahead of glazing bars. The
    # outer arch and any physical relief keep their original geometry.
    return [(panel,-float(st.get('recessM',.2))+.072,.012,item['material'])]
