"""Stream USGS 2023 EPT into a polygon-cropped, generator-compatible point archive.

Only a LAZ chunk and its filtered rows are decoded at once; shards and final staging
array live on disk. Reuses existing survey tile caches without modifying them.
"""
import argparse, collections, json, pathlib, shutil, sys, time
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape, box
from shapely.ops import transform
from shapely import intersects_xy
import laspy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from area_fetch import EPT, get


def run(boundary, maxdepth=8):
    area = pathlib.Path(__file__).resolve().parent
    data = area/'data'; cache=data/'lidar'; cache.mkdir(parents=True, exist_ok=True)
    roots = [area.parents[1]/'data'/'lidar', area.parent/'pacific-heights'/'data'/'lidar']
    counts=collections.Counter()
    def download(rel):
        dest=cache/rel.replace('/', '_')
        if dest.exists(): counts['ownCachedFiles']+=1; return dest
        for root in roots:
            src=root/dest.name
            if src.exists():
                shutil.copyfile(src,dest); counts['reusedCachedFiles']+=1; return dest
        raw=get(EPT+rel, timeout=120); temp=dest.with_suffix(dest.suffix+'.part'); temp.write_bytes(raw); temp.replace(dest)
        counts['downloadedFiles']+=1; return dest
    ep=data/'ept.json'
    if not ep.exists():
        source=area.parents[1]/'data'/'ept.json'
        if source.exists(): shutil.copyfile(source,ep)
        else: ep.write_bytes(get(EPT+'ept.json'))
    meta=json.loads(ep.read_text()); bounds=meta['bounds']; conform=meta['boundsConforming']
    raw=json.loads(pathlib.Path(boundary).read_text())
    if raw['type']=='FeatureCollection':
        from shapely.ops import unary_union
        geom=unary_union([shape(f['geometry']) for f in raw['features']])
    else: geom=shape(raw['geometry'] if raw['type']=='Feature' else raw)
    if geom.geom_type not in ('Polygon','MultiPolygon'): raise ValueError('Boundary must be polygon WGS84')
    poly=transform(Transformer.from_crs(4326,3857,always_xy=True).transform,geom)
    def overlaps(key):
        d,x,y,z=map(int,key.split('-')); size=(bounds[3]-bounds[0])/2**d
        xx,yy,zz=bounds[0]+x*size,bounds[1]+y*size,bounds[2]+z*size
        return zz<=conform[5] and zz+size>=conform[2] and poly.intersects(box(xx,yy,xx+size,yy+size))
    pending=['0-0-0-0']; visited=set(); keys=[]
    while pending:
        key=pending.pop()
        if key in visited: continue
        visited.add(key)
        for node,count in json.loads(download('ept-hierarchy/'+key+'.json').read_text()).items():
            if int(node.split('-')[0])>maxdepth or not overlaps(node): continue
            if count==-1: pending.append(node)
            elif count>0: keys.append(node)
    keys=sorted(set(keys)); (data/'lidar-nodes.json').write_text(json.dumps(keys,indent=2))
    print('nodes',len(keys),flush=True)
    shards=data/'lidar-cropped'; shards.mkdir(exist_ok=True)
    n=0; classes=collections.Counter(); pmin=np.full(3,np.inf); pmax=-pmin; high=0
    for index,key in enumerate(keys):
        shard=shards/(key+'.npy')
        # One EPT tile at a time; each chunk <=100k source returns.
        with laspy.open(download('ept-data/'+key+'.laz')) as reader:
            parts=[]
            for chunk in reader.chunk_iterator(100000):
                x,y,z=np.asarray(chunk.x),np.asarray(chunk.y),np.asarray(chunk.z)
                mask=intersects_xy(poly,x,y)&np.isfinite(z)
                if mask.any(): parts.append(np.column_stack((x[mask],y[mask],z[mask],np.asarray(chunk.classification)[mask],np.asarray(chunk.intensity)[mask])))
            arr=np.concatenate(parts) if parts else np.empty((0,5),dtype=np.float64)
        np.save(shard,arr); n+=len(arr)
        if len(arr):
            pmin=np.minimum(pmin,arr[:,:3].min(axis=0));pmax=np.maximum(pmax,arr[:,:3].max(axis=0))
            classes.update(dict(zip(*np.unique(arr[:,3].astype(int),return_counts=True))));high+=int((arr[:,2]>=300).sum())
        if index%20==0: print(index+1,'/',len(keys),'points',n,flush=True)
    staging=data/'points-staging.npy'; merged=np.lib.format.open_memmap(staging,mode='w+',dtype=np.float64,shape=(n,5)); start=0
    for key in keys:
        arr=np.load(shards/(key+'.npy'),mmap_mode='r'); merged[start:start+len(arr)]=arr; start+=len(arr)
    merged.flush(); archive=data/'points-partial.npz'; np.savez_compressed(archive,points=merged); archive.replace(data/'points.npz'); del merged; staging.unlink()
    qa={'source':EPT,'survey':'USGS CA_SanFrancisco_1_B23, 2023','boundary':str(pathlib.Path(boundary).resolve()),'boundaryWgs84Bounds':list(geom.bounds),'cropWebMercatorBounds':list(poly.bounds),'coordinateColumns':['EPSG:3857 x','EPSG:3857 y','absolute source z (m)','LAS classification','intensity'],'maxDepth':maxdepth,'nodeCount':len(keys),'pointCount':n,'pointMin':pmin.tolist(),'pointMax':pmax.tolist(),'classificationCounts':{str(k):int(v) for k,v in classes.items()},'pointsAtOrAbove300m':high,'verticalClipping':'none; all EPT source elevation nodes retained','memoryPolicy':'one EPT tile, 100000-return decode chunks; disk shards and memory-mapped consolidation','cache':dict(counts),'sourceBoundsConforming':conform}
    (data/'lidar-provenance.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('boundary');parser.add_argument('--maxdepth',type=int,default=8);args=parser.parse_args();run(args.boundary,args.maxdepth)
