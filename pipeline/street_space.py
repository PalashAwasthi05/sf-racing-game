"""Shared street surface ownership and bounded, evidence-led object placement.

No point on a routing graph is evidence for a physical pole. All street compilers
share these surfaces so furniture cannot independently occupy a carriageway,
bicycle track, crossing or building. Optional surveyed surfaces are area input.
"""
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, GeometryCollection
from shapely.ops import unary_union, nearest_points, polygonize
from .mobility import number, length_m


def prepare(ctx):
    boundary, buildings = ctx['boundary'], ctx['all_buildings']
    road = ctx['road_area']
    cycles, paths, crossings, observed = [], [], [], []
    track_ways=[]
    for w in ctx['ways']:
        t, g = w['tags'], w['g']
        if g.geom_type != 'LineString':
            continue
        h = t.get('highway')
        if h == 'cycleway':
            width = min(12, max(.5, length_m(t.get('width'), 3.2 if t.get('oneway') == 'no' else 1.8)))
            cycles.append(g.buffer(width/2, cap_style=2))
            if t.get('cycleway')!='crossing' and t.get('segregated')!='no':track_ways.append((w,width))
        elif t.get('footway') == 'crossing':
            crossings.append(g.buffer(1.5, cap_style=2))
        elif h in ['footway', 'pedestrian', 'path']:
            paths.append(g.buffer(min(8, max(1, number(t.get('width'), 2)))/2, cap_style=2))
    source_file = ctx['DATA'] / 'street-surfaces.geojson'
    if source_file.exists():
        for f in ctx['json'].load(open(source_file)).get('features', []):
            if f['properties'].get('kind') == 'sidewalk':
                observed.append(ctx['geom'](f['geometry']).intersection(boundary))
    cycle = unary_union(cycles).intersection(boundary).difference(buildings)
    crossing = unary_union(crossings).intersection(boundary)
    path = unary_union(paths).intersection(boundary).difference(buildings)
    rows = unary_union([ctx['geom'](r['the_geom']) for r in ctx['json'].load(open(ctx['DATA']/'row.json'))]).intersection(boundary)
    # A right-of-way boundary includes the sidewalk; it is not a curb. Retain
    # its varying remainder, supplemented by mapped walking paths, not a fixed
    # strip on every road. Bound the fallback so unknown plazas aren't filled.
    fallback = rows.intersection(road.buffer(10)).union(path).difference(road.buffer(.02))
    measured = unary_union(observed)
    if observed:
        road = road.difference(measured)
        # Surveyed edges take precedence near their own coverage only.
        sidewalk = measured.union(fallback.difference(measured.buffer(3)))
    else:
        sidewalk = fallback
    curb_file=ctx['DATA']/'curbs.json'
    curb_blocks=[]
    if curb_file.exists():
        curb_data=[ctx['geom'](r['the_geom']) for r in ctx['json'].load(open(curb_file))]
        # Only closed source boundaries establish an inside/outside. Open curb
        # fragments do not justify extending a wall or filling a road.
        for poly in polygonize(unary_union(curb_data)):
            if not poly.is_valid or poly.area<2:continue
            main_roads=unary_union([w['g'] for w,_,_ in ctx['roads'] if w['tags'].get('highway')!='service'])
            if poly.buffer(-.3).intersection(main_roads).length>5:continue
            if poly.intersection(buildings).area>5 or poly.area<600:
                curb_blocks.append(poly)
    curb_blocks=unary_union(curb_blocks)
    if not curb_blocks.is_empty:
        # City curb outlines capture corner arcs and median islands. Their
        # dates remain in the input manifest; they are not certified as-built.
        envelope=curb_blocks.boundary.buffer(10)
        road=road.union(rows.intersection(road.buffer(5)).intersection(envelope)).difference(curb_blocks).difference(buildings)
        measured=curb_blocks.intersection(rows.buffer(.15)).difference(buildings)
        sidewalk=sidewalk.difference(envelope).union(measured)
        # Small closed islands may stand entirely inside the carriageway.
        islands=[p for p in getattr(curb_blocks,'geoms',[curb_blocks]) if p.area<600 and p.intersection(buildings).area<1]
        sidewalk=sidewalk.union(unary_union(islands))
    separators=[]
    bike_file=ctx['DATA']/'bike-network.json'
    if bike_file.exists():
        inventories=[(ctx['geom'](r['shape']),r) for r in ctx['json'].load(open(bike_file)) if 'SAFE-HIT POSTS' in r.get('barrier','')]
        for w,width in track_ways:
            g=w['g'];mid=g.interpolate(.5,normalized=True)
            if not inventories or g.length<8:continue
            line,record=min(inventories,key=lambda a:mid.distance(a[0]))
            s=line.project(mid)
            u=np.array(g.interpolate(.6,normalized=True).coords[0])-g.interpolate(.4,normalized=True).coords[0]
            v=np.array(line.interpolate(min(line.length,s+3)).coords[0])-line.interpolate(max(0,s-3)).coords[0]
            agreement=abs(float(u@v))/max(.001,np.linalg.norm(u)*np.linalg.norm(v))
            if mid.distance(line)>10 or agreement<.92:continue
            # Inventory is a street-centre facility record, not each post. It
            # licenses a separator on the track's road-facing edge only.
            options=[g.offset_curve(side*(width/2+.25)) for side in [-1,1]]
            edge=min(options,key=lambda a:a.distance(line))
            edge=edge.difference(crossing.buffer(1.2)).intersection(boundary)
            separators.append({'geometry':edge,'sourceId':record.get('objectid'),'wayId':w['id']})
    separator_area=unary_union([x['geometry'].buffer(.28,cap_style=2) for x in separators])
    road=road.difference(separator_area)
    sidewalk = sidewalk.difference(buildings.buffer(.03)).difference(cycle).difference(road).difference(separator_area).intersection(boundary).buffer(0)
    curb = sidewalk.boundary.intersection(road.buffer(.15)).difference(crossing)
    rail_clearance=unary_union([w['g'].buffer(1.6) for w in ctx.get('rails',[])]).intersection(boundary)
    forbidden = road.union(cycle).union(crossing).union(buildings).union(rail_clearance)
    return dict(road=road, cycle=cycle, sidewalk=sidewalk, paths=path,
                crossings=crossing, curb=curb, forbidden=forbidden,
                occupied=[], records=[], separators=separators, surfaceSource='city curb outlines + ROW remainder; historical geometry, not as-built verified' if not curb_blocks.is_empty else 'surveyed + ROW fallback' if observed else 'ROW remainder + mapped paths; inferred curb',
                measuredSidewalkAreaM2=round(measured.area, 2))


def place(plan, point, radius, boundary, *, kind, source_id, max_shift=0., require_walk=False):
    """Preserve mapped coordinates when safe; bounded shifts stay explicit.

    Refuse an ambiguous placement instead of depositing an object in a road.
    Objects requiring a sidewalk must fit inside that surface.
    """
    p = Point(point)
    occupied = plan['occupied']
    # Everything is worked out inside a disc round the mapped point, wide enough for any admissible shift: the nearest
    # allowed point inside a disc round p is the nearest overall whenever the disc holds any allowed space, so the
    # outcome is the whole-area one. Buffering the whole area's roads and buildings for every object took hours for a
    # few thousand trees. A disc with no allowed space is widened (50 m, then the whole area) so a rejection's
    # required shift is the whole-area one too.
    def allowed_within(reach):
        win = None if reach is None else p.buffer(reach)
        def near(g, margin):
            return g if win is None else g.intersection(win.buffer(margin + .05))
        forbidden_near = near(plan['forbidden'], radius + .10)
        allowed = boundary.buffer(-radius).difference(forbidden_near.buffer(radius + .10))
        if win is not None: allowed = allowed.intersection(win)
        if require_walk:
            allowed = allowed.intersection(near(plan['sidewalk'], radius).buffer(-radius))
        if occupied:
            close = [(q, r) for q, r in occupied if win is None or q.distance(p) <= reach + r + radius + .15]
            if close: allowed = allowed.difference(unary_union([q.buffer(r+radius+.15) for q,r in close]))
        return allowed, forbidden_near
    for reach in (max_shift + radius + 1., 50., None):
        allowed, forbidden_near = allowed_within(reach)
        if not allowed.is_empty: break
    record = dict(kind=kind, sourceId=str(source_id), sourcePosition=[p.x,p.y], radiusM=radius)
    if allowed.is_empty:
        record.update(status='rejected', reason='no-supported-clear-space')
        plan['records'].append(record); return None
    q = p if allowed.covers(p) else nearest_points(p, allowed)[1]
    delta = p.distance(q)
    if delta > max_shift + 1e-6:
        record.update(status='rejected', reason='source-conflicts-with-street-space', requiredShiftM=round(delta,3))
        plan['records'].append(record); return None
    # Avoid nearest-point rounding putting the disc back across an edge.
    footprint=q.buffer(radius)
    if footprint.intersection(forbidden_near).area > 1e-5:   # the disc holds q and its footprint
        record.update(status='rejected', reason='footprint-intersection')
        plan['records'].append(record); return None
    occupied.append((q,radius))
    record.update(position=[q.x,q.y], displacementM=round(delta,3),
                  status='mapped' if delta < .01 else 'inferred-bounded-shift', imageVerified=False)
    plan['records'].append(record)
    return np.array([q.x,q.y])


def tree_candidates(ctx):
    result=[]; plan=ctx['street_plan']
    for t in ctx['json'].load(open(ctx['DATA']/'trees.json')):
        species=t.get('qspecies','').lower()
        if any(s in species for s in ['vacant','stump','removed','dead tree','planting site']):
            continue
        try:x,y=ctx['local'](float(t['longitude']),float(t['latitude']))
        except (ValueError,KeyError,TypeError):continue
        if not ctx['boundary'].buffer(-1).covers(Point(x,y)):continue
        # Inventory diameter is inches, not metres. Unmeasured trees keep a
        # modest default trunk; never scale the road exclusion to the canopy.
        radius=min(.65,max(.08,number(t.get('dbh'),12)*.0254/2))
        pos=place(plan,(x,y),radius,ctx['boundary'],kind='tree',source_id=t.get('treeid',''),max_shift=1.5)
        if pos is not None:result.append(dict(record=t,position=pos,radius=radius))
    return result


def furniture(ctx):
    plan=ctx['street_plan']; cube,cyl,elev=ctx['cube'],ctx['cylinder'],ctx['elev']
    road_refs={n.attrib['ref'] for w in ctx['osm'].findall('way')
               if any(t.attrib['k']=='highway' and t.attrib['v'] not in ['footway','path','pedestrian','cycleway','steps'] for t in w.findall('tag'))
               for n in w.findall('nd')}
    items=[]
    bin_file=ctx['DATA']/'trash-cans.json'
    if bin_file.exists():
        for item in ctx['json'].load(open(bin_file)):
            if item.get('removeddate'):continue
            try:x,y=ctx['local'](float(item['longitude']),float(item['latitude']))
            except (KeyError,ValueError,TypeError):continue
            items.append(('inventory:'+str(item['objectid']),{'amenity':'waste_basket'},x,y,'municipal inventory'))
    for node in ctx['osm'].findall('node'):
        t={a.attrib['k']:a.attrib['v'] for a in node.findall('tag')}
        x,y=ctx['local'](float(node.attrib['lon']),float(node.attrib['lat']))
        items.append((node.attrib['id'],t,x,y,'OSM'))
    bins=[]
    for node_id,t,x,y,source in items:
        kind = ('bin' if t.get('amenity')=='waste_basket' else 'bollard' if t.get('barrier')=='bollard'
                else 'lamp' if t.get('highway')=='street_lamp' else 'signal' if t.get('highway')=='traffic_signals'
                else 'hydrant' if t.get('emergency')=='fire_hydrant'
                else 'small-pole' if t.get('man_made') in ['utility_pole','bollard'] or t.get('power')=='pole' else None)
        if not kind:continue
        p=Point(x,y)
        if not ctx['boundary'].buffer(-2).covers(p):continue
        if kind=='bin' and any(p.distance(q)<3 for q in bins):continue
        logical = kind=='signal' and node_id in road_refs
        # Stop-line/intersection signal nodes describe traffic control, not the
        # support position. Infer at most one nearby safe curb support per node.
        max_shift=20 if logical else 1.5
        radius={'bin':.36,'bollard':.11,'lamp':.16,'signal':.14,'small-pole':.09,'hydrant':.23}[kind]
        pos=place(plan,(x,y),radius,ctx['boundary'],kind=kind,source_id=node_id,max_shift=max_shift,require_walk=logical or kind=='bin')
        if pos is None:continue
        plan['records'][-1]['source']=source
        if kind=='bin':bins.append(Point(pos))
        z=float(elev(*pos)) + (.26 if plan['sidewalk'].covers(Point(pos)) else .1)
        nearest=nearest_points(Point(pos),plan['road'])[1];toward=np.array(nearest.coords[0])-pos
        toward/=max(np.linalg.norm(toward),.001)
        angle=math.atan2(toward[1],toward[0])
        key='street_objects'
        if kind=='bin':
            # Open aperture, rim and slatted body, not an opaque solid cylinder.
            cyl([*pos,z+.04],[*pos,z+.14],.35,key,'charcoal',16)
            cyl([*pos,z+.16],[*pos,z+.86],.29,key,'dark',16)
            for h in [.20,.88,1.04]:
                for k in range(24):
                    a=k*math.tau/24;b=(k+1)*math.tau/24
                    aa=pos+.33*np.array([math.cos(a),math.sin(a)]);bb=pos+.33*np.array([math.cos(b),math.sin(b)])
                    cyl([*aa,z+h],[*bb,z+h],.024,key,'charcoal',5)
            for k in range(20):
                a=k*math.tau/20;q=pos+.33*np.array([math.cos(a),math.sin(a)])
                cyl([*q,z+.15],[*q,z+.89],.018,key,'charcoal',5)
            ctx['counts']['trashCans']+=1
        elif kind=='hydrant':
            cyl([*pos,z],[*pos,z+.63],.13,key,'white',12)
            cyl([*pos,z+.60],[*pos,z+.77],.17,key,'white',12)
            cross=np.array([-toward[1],toward[0]])
            cyl([*(pos-cross*.22),z+.48],[*(pos+cross*.22),z+.48],.08,key,'white',10)
            ctx['counts']['hydrants']+=1
        elif kind in ['bollard','small-pole']:
            h=length_m(t.get('height'),.95 if kind=='bollard' else 2.4)
            cyl([*pos,z],[*pos,z+h],radius,key,'charcoal' if kind=='bollard' else 'steel',10)
            cyl([*pos,z+h-.20],[*pos,z+h-.10],radius+.008,key,'white',10)
            ctx['counts']['bollards' if kind=='bollard' else 'smallPoles']+=1
        else:
            h=length_m(t.get('height'),3.5 if kind=='signal' else 8)
            cyl([*pos,z],[*pos,z+.30],radius,key,'charcoal',12)
            cyl([*pos,z+.25],[*pos,z+h],.075,key,'steel',10)
            if kind=='signal':
                normal=toward;head=pos+normal*.20
                cube([*head,z+h-.55],[.32,.30,1.0],key,'charcoal',angle)
                # Face toward the controlled road, rather than global north.
                for j,col in enumerate(['#7d2925','#92723a','#49836c']):
                    p2=head+normal*.17
                    cyl([*p2,z+h-.25-j*.27],[*(p2+normal*.035),z+h-.25-j*.27],.085,key,col,12)
                if logical:plan['records'][-1]['status']='inferred-signal-support-from-control-node'
            else:
                tip=pos+toward*1.5
                cyl([*pos,z+h],[*tip,z+h-.10],.045,key,'steel',8)
                cube([*tip,z+h-.16],[.85,.32,.12],key,'charcoal',angle)
                cube([*tip,z+h-.23],[.62,.24,.025],key,'white',angle)
            ctx['counts']['signals' if kind=='signal' else 'lamps']+=1
    for separator in plan.get('separators',[]):
        for line in ctx['lines'](separator['geometry']):
            for s in np.arange(2.5,line.length-2,5):
                p=line.interpolate(s)
                pos=place(plan,p,.05,ctx['boundary'],kind='cycle-separator-post',source_id=separator['sourceId'],max_shift=0)
                if pos is None:continue
                z=float(elev(*pos))+.12
                cyl([*pos,z],[*pos,z+.06],.14,'street_objects','charcoal',10)
                cyl([*pos,z+.06],[*pos,z+.86],.045,'street_objects','white',10)
                for h in [.60,.74]:cyl([*pos,z+h],[*pos,z+h+.045],.048,'street_objects','silver',10)
                plan['records'][-1].update(status='inferred-spacing-on-inventory-supported-separator',wayId=separator['wayId'],spacingM=5)
                ctx['counts']['cycleSeparatorPosts']+=1
    return plan['records']
