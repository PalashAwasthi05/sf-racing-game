"""Extract bounded tree-only detail from shipped neighborhood cells, one at a time.

No base refresh, source-world decode or active manifest mutation is needed. All
positions, branch faces and leaf samples come from the pinned existing cells.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from neighborhood_canopies import _centres, _tints
from package_route_block import Source, flat_nodes, local_path, unpack_portable


def binding(index):
    return dict(version=3, sourceWorldHash=index['sourceWorldHash'],
                cells=[dict(id=c['id'], contentHash=c['contentHash']) for c in index['cells']])


def ensure_package(public, index_url):
    """Reuse a complete matching tree package; otherwise compile before publish."""
    public=Path(public)
    index=json.loads(local_path(public,index_url).read_text());bound=binding(index)
    digest=hashlib.sha256(json.dumps(bound,separators=(',',':')).encode()).hexdigest()[:16]
    url='/compare/tree-detail-'+digest+'.json';path=local_path(public,url)
    if path.is_file():
        existing=json.loads(path.read_text())
        skeleton=existing.get('skeleton',{})
        if (all(existing.get(k)==v for k,v in bound.items()) and skeleton.get('binary') and
            local_path(public,skeleton['binary']).is_file() and local_path(public,skeleton['binary']).stat().st_size==skeleton.get('compressedBytes') and
            all(local_path(public,c['binary']).is_file() and local_path(public,c['binary']).stat().st_size==c['compressedBytes']
                for c in existing.get('treeCells',[]))):
            return url
    return package(public,index_url)


def extract_cell(source):
    """Exact source samples and indexed bark, in absolute y-up coordinates."""
    leaves = []; barks = []; classes = set();frond_samples=0
    for node, matrix in flat_nodes(source.model):
        mesh = source.model['meshes'][node['mesh']]
        extra = {**mesh.get('extras', {}), **node.get('extras', {})}
        if extra.get('layer', node.get('name', '').split('__')[0]) != 'vegetation':
            continue
        name = str(extra.get('material') or node.get('name', '').split('___')[-1])
        leaf = bool(extra.get('foliageFanCentres')) or name.startswith('leaf') or name == 'frond'
        if not leaf and name not in ('bark', 'wood', 'trunk'):
            continue
        classes.add(name)
        for primitive in mesh['primitives']:
            material = source.model['materials'][primitive['material']]
            factor = np.array(material.get('pbrMetallicRoughness', {}).get('baseColorFactor', [1,1,1,1])[:3])
            frond=name=='frond' and not extra.get('foliageFanCentres') and primitive.get('mode',4)==4
            if frond:
                samples,_=_centres(source,primitive,False);frond_samples+=len(samples)
            if leaf and not frond:
                p, rows = _centres(source, primitive, bool(extra.get('foliageFanCentres')))
                color = _tints(source, primitive, rows)[:, :3] * factor
            else:
                p = source.array(primitive['attributes']['POSITION'])
                rows = np.arange(len(p)); color = _tints(source, primitive, rows)[:, :3] * factor
            if not len(p):
                continue
            p = (p @ matrix[:3,:3].T + matrix[:3,3]).astype('<f4')
            color = np.rint(np.clip(color, 0, 1)*65535).astype('<u2')
            if leaf and not frond:
                leaves.append((p, color))
                continue
            order = (source.array(primitive['indices']).reshape(-1) if 'indices' in primitive
                     else np.arange(len(p))).astype('<u4')
            normal = np.zeros_like(p)
            triangles = order.reshape(-1, 3)
            face = np.cross(p[triangles[:,1]]-p[triangles[:,0]], p[triangles[:,2]]-p[triangles[:,0]])
            for k in range(3):
                np.add.at(normal, triangles[:,k], face)
            lengths = np.linalg.norm(normal, axis=1)
            normal /= np.maximum(lengths[:,None], 1e-12)
            barks.append((p, np.rint(normal*127).astype('i1'), color, order, 'frond' if frond else 'bark'))
    payload = bytearray(); entries = []; source_samples = bark_triangles = 0
    def append(array):
        payload.extend(array.tobytes())
        while len(payload)%4:
            payload.append(0)
    for p, color in leaves:
        entries.append(dict(kind='leaves', offset=len(payload), vertices=len(p)))
        append(p); append(color); source_samples += len(p)
    for p, normal, color, order,kind in barks:
        entries.append(dict(kind=kind, offset=len(payload), vertices=len(p), indices=len(order)))
        append(p); append(normal); append(color); append(order)
        if kind=='bark':bark_triangles += len(order)//3
    bounds = None
    if leaves or barks:
        low = np.min([v[0].min(0) for v in leaves+barks], axis=0)
        high = np.max([v[0].max(0) for v in leaves+barks], axis=0)
        bounds = [float(low[0]-.75), float(-high[2]-.75), float(high[0]+.75), float(-low[2]+.75)]
    return bytes(payload), entries, dict(sourceSamples=source_samples+frond_samples, leafSamples=source_samples, frondTriangles=sum(len(v[3])//3 for v in barks if v[4]=='frond'), barkTriangles=bark_triangles), bounds, sorted(classes)


def attach_source_skeleton(public, cells):
    """Small neighboring source-axis records keep split crowns attached at cells.

    Sequential tree-only reads total ~50MB, never the source-world container.
    """
    segments={}
    for cell in cells:
        raw=gzip.decompress(local_path(public,cell['binary']).read_bytes())
        for entry in cell['entries']:
            if entry['kind']!='bark':continue
            n=entry['vertices'];offset=entry['offset']
            p=np.frombuffer(raw,dtype='<f4',count=n*3,offset=offset).reshape(-1,3)
            offset=(offset+n*12+3)//4*4;offset=(offset+n*3+3)//4*4;offset=(offset+n*6+3)//4*4
            index=np.frombuffer(raw,dtype='<u4',count=entry['indices'],offset=offset)
            start=0
            while start+11<len(index):
                root,tip=index[start+1],index[start+11];end=start+12
                while end+11<len(index) and index[end+1]==root and index[end+11]==tip:end+=12
                sectors=(end-start)//12
                if 5<=sectors<=32:
                    a,b=p[root],p[tip];radius=float(np.linalg.norm(p[index[start]]-a));length=float(np.linalg.norm(b-a))
                    if length>max(.2,radius*5) and b[1]-a[1]>length*.1:
                        identity=tuple(np.round(np.r_[a,b,radius],5))
                        segments[identity]=dict(root=a.astype(float).tolist(),tip=b.astype(float).tolist(),radius=radius)
                        start=end;continue
                start+=3
    values=list(segments.values())
    def owner(p):return ','.join(str(round(v*100))for v in p)
    stems=[s for s in values if s['tip'][1]-s['root'][1]>np.linalg.norm(np.array(s['tip'])-s['root'])*.97]
    from scipy.spatial import cKDTree
    if stems:
        tree=cKDTree([s['tip']for s in stems])
        distances,indices=tree.query([s['root']for s in values])
        stem_ids={id(s)for s in stems}
        for s,d,i in zip(values,distances,indices):s['owner']=owner(s['tip'] if id(s)in stem_ids else stems[i]['tip'] if d<.25 else s['root'])
    else:
        for s in values:s['owner']=owner(s['root'])
    low=np.array([np.minimum(s['root'],s['tip'])for s in values]);high=np.array([np.maximum(s['root'],s['tip'])for s in values])
    for cell in cells:
        b=cell['boundsEN']
        if values:
            selected=(high[:,0]>=b[0]-6)&(low[:,0]<=b[2]+6)&(high[:,2]>=-b[3]-6)&(low[:,2]<=-b[1]+6)
            cell['skeletonIds']=np.flatnonzero(selected).tolist()
        else:cell['skeletonIds']=[]
        cell.pop('skeleton',None)
    owners={key:i for i,key in enumerate(sorted({s['owner']for s in values}))}
    data=np.array([s['root']+s['tip']+[s['radius'],owners[s['owner']]]for s in values],dtype='<f4').reshape(-1,8).tobytes()
    encoded=gzip.compress(data,compresslevel=6,mtime=0);digest=hashlib.sha256(encoded).hexdigest()
    url='/compare/tree-skeleton-'+digest[:16]+'.bin';local_path(public,url).write_bytes(encoded)
    return dict(binary=url,binaryHash=digest,compressedBytes=len(encoded),byteLength=len(data),segments=len(values))


def package(public, index_url='/compare/neighborhood-stream.json'):
    index = json.loads(local_path(public, index_url).read_text())
    bound = binding(index)
    # Match JSON.stringify's insertion order and compact encoding in the viewer.
    digest = hashlib.sha256(json.dumps(bound, separators=(',', ':')).encode()).hexdigest()[:16]
    cells = []; stats = dict(cells=0, sourceSamples=0, leafSamples=0, frondTriangles=0, barkTriangles=0, encodedBytes=0, decodedBytes=0, maxInputCellBytes=0)
    for cell in index['cells']:
        if not cell.get('stats', {}).get('layers', {}).get('vegetation'):
            continue
        with tempfile.TemporaryFile() as stream:
            unpack_portable(public, cell, stream)
            source = Source(stream)
            payload, entries, counts, bounds, classes = extract_cell(source)
        if not entries:
            continue
        encoded = gzip.compress(payload, compresslevel=6, mtime=0)
        binary_hash = hashlib.sha256(encoded).hexdigest()
        url = '/compare/tree-detail-cell-' + binary_hash[:16] + '.bin'
        local_path(public, url).write_bytes(encoded)
        cells.append(dict(id=cell['id'], contentHash=cell['contentHash'], boundsEN=bounds,
                          binary=url, binaryHash=binary_hash, compressedBytes=len(encoded),
                          byteLength=len(payload), entries=entries, classes=classes, **counts))
        stats['cells'] += 1; stats['encodedBytes'] += len(encoded); stats['decodedBytes'] += len(payload)
        stats['maxInputCellBytes'] = max(stats['maxInputCellBytes'], cell['uncompressedBytes'])
        for key, value in counts.items():
            stats[key] += value
        if len(cells)%40 == 0:
            print(json.dumps(dict(phase='tree-detail', **stats)), flush=True)
    skeleton=attach_source_skeleton(public,cells);stats['skeletonSegments']=skeleton['segments']
    manifest = dict(**bound, treeCells=cells, stats=stats, skeleton=skeleton)
    url = '/compare/tree-detail-' + digest + '.json'
    local_path(public, url).write_text(json.dumps(manifest, separators=(',', ':'))+'\n')
    print(json.dumps(dict(manifest=url, **stats)), flush=True)
    return url


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('public', type=Path)
    parser.add_argument('--index', default='/compare/neighborhood-stream.json')
    args = parser.parse_args()
    package(args.public, args.index)
