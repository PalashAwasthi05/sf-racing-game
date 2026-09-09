"""Adapt normalized generation output for the viewer without place lookups."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point
from shapely.ops import unary_union

def prepare(input_path, candidate_dir, output_dir):
    path=Path(input_path); candidate=Path(candidate_dir); out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(path.read_text()); report=json.loads((candidate/'report.json').read_text())
    points=np.load(path.parent/data['groundFile'])['points']; tree=cKDTree(points[:,:2])
    def elevation(xy):
        _,index=tree.query(xy,k=min(8,len(points)))
        return np.median(points[index,2],axis=-1)
    outlines=[shape(b['geometry']) for b in data['buildings']]
    extent=max(300, np.ceil(max(abs(v) for g in outlines for v in g.bounds)/5)*5+10)
    axis=np.arange(-extent,extent+1,2.0); xx,yy=np.meshgrid(axis,axis)
    ground=elevation(np.c_[xx.ravel(),yy.ravel()]).round(4).tolist()
    records={r['key']:r for r in report['records']}; buildings=[]
    for index,(item,g) in enumerate(zip(data['buildings'],outlines)):
        c=g.centroid; z=float(elevation([[c.x,c.y]])[0]); record=records[item['key']]
        buildings.append({'id':item['key'],'name':f'Building {index+1}',
          'center':[c.x,c.y,z],'height':max(0,record.get('baseRoofM',z)-z),'area':g.area,
          'heightSource':'Point cloud','footprintSource':'Mapped outline','roofParts':0,
          'facade':'Awaiting automatic image reconstruction','pointCount':0,'sourceUrl':'',
          'facadeEvidence':'Missing images','fittedWalls':0,'totalWalls':0})
    occupied=unary_union(outlines).buffer(1.0); views=[]
    for item in data['surfaces']:
        if item['layer']!='roads':continue
        street=shape(item['geometry']).difference(occupied)
        if street.is_empty:continue
        pieces=list(street.geoms) if hasattr(street,'geoms') else [street]
        for g in pieces:
            if g.area<30:continue
            p=g.representative_point()
            if p.distance(Point(0,0))>230:continue
            corners=np.asarray(g.minimum_rotated_rectangle.exterior.coords)
            edges=np.diff(corners,axis=0); vec=edges[np.argmax(np.linalg.norm(edges,axis=1))]
            vec=vec/np.linalg.norm(vec)*25
            views.append({'position':[p.x,p.y,1.25],'lookAt':[p.x+vec[0],p.y+vec[1],1.25],'label':'Mapped road'})
    views.sort(key=lambda v:(np.linalg.norm(v['position'][:2]),*v['position'][:2]))
    world={'center':{'lat':0,'lon':0},'buildings':buildings,'counts':{'buildings':len(buildings)},
      'ground':ground,'grid':{'min':-extent,'step':2,'n':len(axis)},'sources':[],'streetViews':views}
    (out/'world.json').write_text(json.dumps(world,separators=(',',':')))
    shutil.copy2(candidate/'candidate.glb',out/'candidate.glb')
    shutil.copy2(candidate/'report.json',out/'report.json')
    print(json.dumps({'buildings':len(buildings),'streetViews':len(views),'geometryBytes':(out/'candidate.glb').stat().st_size}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();prepare(args.input,args.candidate,args.output)
