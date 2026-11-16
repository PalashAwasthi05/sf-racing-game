"""Compile local multi-view paint evidence without extending colors along ways."""
import math
from shapely.geometry import shape, GeometryCollection
from shapely.ops import unary_union

def admitted_geometry(packet, plan, origin):
    if packet.get('origin') != origin:
        raise ValueError('Paint evidence origin differs from the generated world')
    pieces=[]
    allowed=plan['cycle'].buffer(2).intersection(plan['road'].union(plan['cycle'])).difference(plan['sidewalk'])
    for feature in packet.get('features',[]):
        p=feature.get('properties',{})
        if p.get('status')!='coarse-pixel-component-supported' or len(set(p.get('imageIds',[])))<2:continue
        agreement=p.get('agreementDistanceM',float('inf'))
        if not isinstance(agreement,(int,float)) or not math.isfinite(agreement) or agreement>.75:continue
        g=shape(feature['geometry'])
        if not g.is_valid or g.geom_type not in ('Polygon','MultiPolygon'):continue
        g=g.intersection(allowed)
        if not g.is_empty:pieces.append(g)
    # Union before drawing: duplicate view pairs must not create stacked decals.
    return unary_union(pieces) if pieces else GeometryCollection()

def compile(ctx):
    source=ctx['DATA']/'ground-paint-evidence.json'
    if not source.exists():return []
    packet=ctx['json'].load(open(source))
    if packet.get('worldCRS') != ctx['ll'].target_crs.to_string():
        raise ValueError('Paint evidence projected CRS differs from the generated world')
    g=admitted_geometry(packet,ctx['street_plan'],{'lat':ctx['lat'],'lon':ctx['lon']})
    count=0
    for piece in ctx['polys'](g):
        if piece.area<.4:continue
        # Remove sub-cell stair steps only; never fill a gap or extend a whole lane.
        smooth=piece.simplify(.18,preserve_topology=True)
        if abs(smooth.area-piece.area)/piece.area>.12:smooth=piece
        ctx['surface'](smooth,'markings','greenpaint',.128);count+=1
    ctx['counts']['observedPaintPatches']=count
    return [{'rule':'multi-view-ground-paint','patches':count,'areaM2':round(g.area,3),'imageVerified':False,'status':'coarse-projection; metric placement provisional','projectionUncertaintyM':1.25,'source':source.name}]
