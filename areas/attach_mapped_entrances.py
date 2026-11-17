"""Join exterior street-level entrance nodes to the generated wall that owns their mapped footprint."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from shapely.geometry import LineString


def tags(element):
    return {t.attrib['k']: t.attrib['v'] for t in element.findall('tag')}


def run(area, source, output):
    packet = json.loads(source.read_text())
    evidence = json.loads((area/'evidence.json').read_text())
    root = ET.parse(area/'data/osm.xml').getroot()
    config = json.loads((area/'config.json').read_text())
    center = config['region']['center']
    job = json.loads((area/'area-source-job.json').read_text())
    project = Transformer.from_crs(4326, job['projectedCrs'], always_xy=True)
    origin = np.array(project.transform(center['lon'], center['lat']))
    nodes = {n.attrib['id']: n for n in root.findall('node')}

    def location(node):
        return np.array(project.transform(float(node.attrib['lon']), float(node.attrib['lat'])))-origin

    walls, segments = [], []
    for wall in evidence['facadeWalls']:
        if wall.get('zBaseM', 0) > .3 or wall['heightM'] < 2.4 or wall['lengthM'] < 3: continue
        if wall.get('exposure') in ('party', 'hidden', 'gap'): continue
        n = np.array(wall['normal'])
        u = np.array([-n[1], n[0]])
        mid = np.array(wall['midpoint'])
        segments.append(LineString([mid-u*wall['lengthM']/2, mid+u*wall['lengthM']/2]))
        walls.append(wall)
    index = STRtree(segments)
    new = {}
    stats = Counter()
    used = set()
    for way in root.findall('way'):
        t = tags(way)
        if t.get('building') in (None, 'no') or t.get('location')=='underground': continue
        ids = [s.attrib['ref'] for s in way.findall('nd')]
        doors = [k for k in ids if k in nodes and (tags(nodes[k]).get('entrance') in
                 ('yes','main','service','exit','staircase') or tags(nodes[k]).get('door') in
                 ('yes','hinged','automatic','overhead'))]
        if not doors or ids[0]!=ids[-1] or any(k not in nodes for k in ids): continue
        owner = Polygon([location(nodes[k]) for k in ids]).buffer(0)
        for k in dict.fromkeys(doors):
            if k in used: continue
            dt = tags(nodes[k])
            if dt.get('level','0') not in ('0','0.0') or dt.get('indoor')=='yes' or dt.get('railway')=='subway_entrance' or dt.get('fixme'): continue
            pos=location(nodes[k])
            point=Point(pos)
            hits=[]
            for i in index.query(point.buffer(1.6)):
                w=walls[i]
                n=np.array(w['normal'])
                d=segments[i].distance(point)
                if d>1.5 or not owner.covers(Point(pos-n*.75)) or owner.covers(Point(pos+n*.75)): continue
                along=segments[i].project(point)
                width=3.0 if dt.get('door')=='overhead' else 1.05
                if min(along,segments[i].length-along)<width/2: continue
                hits.append((d,int(i),along,width))
            hits.sort()
            if not hits or (len(hits)>1 and hits[1][0]-hits[0][0]<.25): continue
            _,i,along,width=hits[0]
            w=walls[i];n=np.array(w['normal']);u=np.array([-n[1],n[0]])
            mid=np.array(w['midpoint']);length=w['lengthM'];h=w['heightM']
            lower=float(mid@u)-length/2
            candidates=[p for p in packet['profiles'] if not p.get('materialOnly') and p.get('frameU')
                        and (p['match'].get('featureId') in (w['buildingId'],re.sub(r'_p[0-9]+$','',w['buildingId'])) or
                             str(p['match'].get('osmWayId'))==way.attrib['id'])
                        and np.dot(n,p['normal'])>.99 and abs(float(mid@np.array(p['normal']))-p['planeV'])<1.
                        and p['frameU'][0]+width/2<float(pos@np.array([-p['normal'][1],p['normal'][0]]))<p['frameU'][1]-width/2]
            if len(candidates)>1: continue
            if candidates:
                p=candidates[0];pn=np.array(p['normal']);pn=pn/np.linalg.norm(pn)
                along=float(pos@np.array([-pn[1],pn[0]]))-p['frameU'][0]
                length=p['frameU'][1]-p['frameU'][0];h=p['heightRefM']
            else:
                if i not in new:
                    new[i]=dict(id=f"mapped-entrances-{w['buildingId']}-{i}",match={'osmWayId':way.attrib['id'],'featureId':re.sub(r'_p[0-9]+$','',w['buildingId']),'partId':w['buildingId']},
                        normal=n.tolist(),frameU=[lower,lower+length],planeV=float(mid@n),heightRefM=h,material=w['material'],
                        sourceWallMidpointEN=mid.tolist(),sourceWallLengthM=length,sourceIds=[],groups=[],openings=[],
                        reading={'coverageMode':'positive-only','observedZRangeM':[0,2.4],'visibleShare':0})
                p=new[i]
            kind='rollup' if dt.get('door')=='overhead' else 'entry'
            style={'kind':kind}
            if kind=='entry':style['door']={'kind':'single'}
            sid='osm:node:'+k
            p['sourceIds'].append(sid)
            p['openings'].append(dict(rect=[(along-width/2)/length,0,(along+width/2)/length,min(h,2.8 if kind=='rollup' else 2.15)/h],
                style=style,evidence='mapped-entrance',sourceIds=[sid],dimensionSource='inferred'))
            packet['sources'][sid]={'provider':'OpenStreetMap','url':'https://www.openstreetmap.org/node/'+k}
            used.add(k);stats['mappedGroundEntrances']+=1
    packet['profiles'].extend(new.values())
    packet['stats'].update(stats)
    output.write_text(json.dumps(packet,indent=1))
    print(dict(stats))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('area','source','output'):parser.add_argument(key,type=Path)
    args=parser.parse_args();run(args.area,args.source,args.output)
