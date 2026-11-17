"""Texture-free real facade geometry, sequentially derived from shipped cells.

No full-world decode and no mutable manifest writes. Solid primitives simplify
within their own material with 5cm absolute error and locked open boundaries;
glazing triangles never simplify. Quantization adds at most half a batch axis
step, reported explicitly. Two appearance groups per owner, in bounded batches.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np
from package_route_block import Source, flat_nodes, building_id, unpack_portable, json_bytes, local_path
from package_drive_residuals import material_color, vertex_colors, immutable_write

VERSION = 1
MAX_BATCH_VERTICES = 60000


def layer_of(node, mesh):
    return node.get('extras', {}).get('layer') or mesh.get('extras', {}).get('layer') or node.get('name', '').split('__')[0]


class SolidSimplifier:
    def __init__(self):
        self.process = subprocess.Popen([shutil.which('node') or 'node', str(Path(__file__).with_name('simplify_facade_middle.mjs'))], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def reduce(self, positions, indices):
        # Weld positional UV seams inside one material only. Border constraints
        # retain jamb/window silhouettes; glazing uses the untouched path below.
        p, inverse = np.unique(positions, axis=0, return_inverse=True)
        mapped = inverse[indices].astype('<u4')
        if len(mapped) >= 192:
            stream = self.process.stdin
            stream.write(struct.pack('<II', len(p), len(mapped)))
            stream.write(np.ascontiguousarray(p, dtype='<f4').tobytes()); stream.write(mapped.tobytes()); stream.flush()
            header = self.process.stdout.read(4)
            if len(header) != 4: raise ValueError('Facade simplifier stopped')
            count, = struct.unpack('<I', header)
            if count > len(mapped) or count % 3: raise ValueError('Invalid facade simplifier result')
            raw = self.process.stdout.read(count*4)
            if len(raw) != count*4: raise ValueError('Truncated facade simplifier result')
            # Tiny closed primitives can legally collapse entirely at 5cm.
            # Keep them instead; the middle tier never deletes a whole part.
            if count: mapped = np.frombuffer(raw, dtype='<u4')
        used, compact = np.unique(mapped, return_inverse=True)
        # Map each output location to a source corner, retaining its tint.
        _, first = np.unique(inverse, return_index=True)
        return p[used], first[used], compact.astype('<u4')

    def close(self):
        self.process.stdin.close(); self.process.stdout.close()
        if self.process.wait(): raise RuntimeError('Facade simplifier failed')


def packed_vertices(positions, indices, colors):
    """Hard-edge vertices; never smooth the corners of a window or a building."""
    faces = indices.reshape(-1, 3)
    normal = np.cross(positions[faces[:, 1]]-positions[faces[:, 0]], positions[faces[:, 2]]-positions[faces[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    normals = np.repeat(np.rint(np.clip(normal, -1, 1)*127).astype('i1'), 3, axis=0)
    points = positions[indices].astype('<f4'); tints = colors[indices]
    key = np.ascontiguousarray(np.column_stack((points.view('u1').reshape(-1, 12), normals.view('u1'), tints)))
    _, first, remap = np.unique(key.view('V18').reshape(-1), return_index=True, return_inverse=True)
    return points[first], normals[first], tints[first], remap.astype('<u4')


def split_batches(parts, limit=MAX_BATCH_VERTICES):
    """Split at triangle boundaries, retaining every input triangle exactly once."""
    pending = []; count = 0
    for p, n, c, ix in parts:
        # Splitting by index count is conservative: unique vertices <= corners.
        for start in range(0, len(ix), limit//3*3):
            selection = ix[start:start+limit//3*3]
            used, compact = np.unique(selection, return_inverse=True)
            if count+len(used) > limit and pending:
                yield merge(pending); pending=[]; count=0
            pending.append((p[used], n[used], c[used], compact.astype('<u4'))); count += len(used)
    if pending: yield merge(pending)


def merge(parts):
    offsets = np.cumsum([0]+[len(p) for p, _, _, _ in parts[:-1]])
    return (*(np.concatenate([part[k] for part in parts]) for k in range(3)), np.concatenate([part[3]+off for part, off in zip(parts, offsets)]))


def write_batch(output, batch, glazing):
    p, n, c, ix = batch
    lo, hi = p.min(axis=0).astype(float), p.max(axis=0).astype(float)
    step = np.maximum((hi-lo)/65535, 1e-9)
    q = np.rint((p-lo)/step).clip(0, 65535).astype('<u2')
    error = float(np.abs((q*step+lo)-p).max())
    start = output.tell()
    for array in (q, n, c): output.write(np.ascontiguousarray(array).tobytes())
    output.write(bytes(-output.tell()%4))
    output.write(ix.astype('<u2').tobytes()); output.write(bytes(-output.tell()%4))
    return {'offset': start, 'vertices': len(p), 'indices': len(ix), 'min': lo.tolist(), 'step': step.tolist(),
            'glazing': glazing, 'quantizationErrorM': error,
            'residentBytes': len(p)*18+len(ix)*2}


def compile_cell(source, cell, output, simplifier, feature=None):
    groups = {False: [], True: []}; cache = {}; buildings = set()
    stats = dict(sourceTriangles=0, triangles=0, vertices=0, sourceGlazingTriangles=0, glazingTriangles=0, residentBytes=0, batches=0, quantizationErrorM=0.)
    for node, matrix in flat_nodes(source.model):
        mesh = source.model['meshes'][node['mesh']]; bid = building_id(node, mesh)
        if (feature is not None and bid != feature) or not bid or layer_of(node, mesh) not in ('buildings', 'facades', 'facade'): continue
        buildings.add(bid)
        for primitive in mesh['primitives']:
            if primitive.get('mode', 4) != 4: continue
            attrs = primitive['attributes']; p = source.array(attrs['POSITION'])
            p = (p@matrix[:3, :3].T+matrix[:3, 3]).astype('<f4')
            ix = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(p), dtype='<u4')
            material = source.model['materials'][primitive['material']]
            label = material.get('name', '').lower()+' '+str(node.get('extras', {}).get('material', '')).lower()
            glazing = any(word in label for word in ('glass', 'glazing'))
            tint = vertex_colors(source, primitive)
            colors = np.ones((len(p), 3)) if tint is None else tint
            colors *= material_color(source, primitive['material'], cache)
            colors = np.rint(np.clip(colors, 0, 1)*255).astype('u1')
            stats['sourceTriangles'] += len(ix)//3
            if glazing: stats['sourceGlazingTriangles'] += len(ix)//3
            elif len(np.unique(colors, axis=0)) == 1:
                p, used, ix = simplifier.reduce(p, ix); colors = colors[used]
            # Nonconstant vertex tint boundaries remain exact rather than
            # accepting a geometry-only simplifier's color interpolation loss.
            groups[glazing].append(packed_vertices(p, ix, colors))
    batches = []
    for glazing, parts in groups.items():
        for batch in split_batches(parts):
            record = write_batch(output, batch, glazing); batches.append(record)
            stats['triangles'] += record['indices']//3; stats['vertices'] += record['vertices']
            if glazing: stats['glazingTriangles'] += record['indices']//3
            stats['residentBytes'] += record['residentBytes']
            stats['quantizationErrorM'] = max(stats['quantizationErrorM'], record['quantizationErrorM'])
    stats.update(batches=len(batches), buildings=len(buildings))
    if stats['sourceGlazingTriangles'] != stats['glazingTriangles']: raise ValueError('Glazing topology lost')
    if sorted(buildings) != ([feature] if feature is not None else sorted(cell['stats']['buildingIds'])): raise ValueError('Facade owner coverage differs from source cell')
    return batches, stats


def over_budget(entry, owner_count):
    batches = entry['batches']
    largest = max((batches[i+1]['offset'] if i+1<len(batches) else entry['byteLength'])-b['offset'] for i,b in enumerate(batches))
    stats = entry['stats']
    return (2*entry['compressedBytes']+largest+1024**2 > 6*1024**2
            or stats['residentBytes']+stats['vertices']*(2 if owner_count<4096 else 4) > 16*1024**2
            or stats['triangles']>300000 or len(batches)>40)


def compile_entries(source, cell, temp, public, simplifier, owner_count, feature=None, allow_split=True):
    packed=temp/'facades.bin'
    with packed.open('wb') as out:
        batches,stats=compile_cell(source,cell,out,simplifier,feature)
    compressed=temp/'facades.gz'
    with packed.open('rb') as src, compressed.open('wb') as target:
        with gzip.GzipFile(fileobj=target,mode='wb',mtime=0,filename='') as dst:
            shutil.copyfileobj(src,dst,1024*1024)
    raw=compressed.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    url='/compare/facade-cell-'+digest[:16]+'.bin.gz'
    entry={k:cell[k] for k in ('id','boundsEN','tileEN')}
    entry.update(sourceHash=cell['contentHash'],url=url,sha256=digest,compressedBytes=len(raw),byteLength=packed.stat().st_size,batches=batches,stats=stats)
    if allow_split and feature is None and len(cell['stats']['buildingIds'])>1 and over_budget(entry,owner_count):
        for bid in sorted(cell['stats']['buildingIds']):
            yield from compile_entries(source,cell,temp,public,simplifier,owner_count,bid)
        return
    if feature is not None:
        entry.update(id=cell['id']+'@'+feature,parentId=cell['id'],buildingId=feature)
        # Tight conservative bounds of this feature's packed positions. The
        # per-axis step is the full unquantized range divided by 65535.
        lo=np.min([b['min'] for b in batches],axis=0)
        hi=np.max([np.asarray(b['min'])+np.asarray(b['step'])*65535 for b in batches],axis=0)
        entry['boundsEN']=[float(lo[0]),float(-hi[2]),float(hi[0]),float(-lo[2])]
    immutable_write(local_path(public,url),raw)
    yield entry


def package(public, index_url='/compare/neighborhood-stream.json', limit=None):
    public = Path(public).resolve(); index_path = local_path(public, index_url)
    manifest = json.loads(index_path.read_bytes())
    entries=[]; totals={k:0 for k in ('sourceTriangles','triangles','vertices','sourceGlazingTriangles','glazingTriangles','residentBytes','batches','buildings','compressedBytes')}
    simplifier=SolidSimplifier()
    try:
        with tempfile.TemporaryDirectory(prefix='facade-middle-') as scratch:
            temp=Path(scratch)
            cells=[c for c in manifest['cells'] if c['stats'].get('buildings')]
            for cell in cells[:limit]:
                with tempfile.TemporaryFile() as source_file:
                    unpack_portable(public,cell,source_file)
                    for entry in compile_entries(Source(source_file),cell,temp,public,simplifier,len(manifest['cells']),allow_split=manifest.get('middlePartitions')=='building-v1'):
                        entries.append(entry)
                        for k in totals: totals[k] += entry['compressedBytes'] if k=='compressedBytes' else entry['stats'][k]
                        print(json.dumps({'cell':entry['id'],**entry['stats'],'compressedBytes':entry['compressedBytes']}),flush=True)
    finally: simplifier.close()
    result={'version':VERSION,'format':'facade-middle-v1','sourceWorldHash':manifest['sourceWorldHash'],
            'sourceCells':[{ 'id':c['id'],'contentHash':c['contentHash']} for c in manifest['cells']],
            'cells':entries,'stats':totals,'complete':limit is None,
            'policy':{'solidErrorM':.05,'openBoundaries':'locked per source material primitive','glazing':'every source glazing triangle, unchanged positions before quantization',
                      'textures':0,'physics':'display only; existing permanent source collision hulls remain authoritative','maxBatchVertices':MAX_BATCH_VERTICES}}
    data=json_bytes(result);url='/compare/neighborhood-facades-'+hashlib.sha256(data).hexdigest()[:16]+'.json'
    immutable_write(local_path(public,url),data)
    print(json.dumps({'index':url,'stats':totals,'maxCellResidentBytes':max(c['stats']['residentBytes'] for c in entries),'maxQuantizationErrorM':max(c['stats']['quantizationErrorM'] for c in entries)}),flush=True)
    return result,url


def validate_asset(public, asset, manifest, verify_files=True):
    """Bind to the entire detail-cell set, including owners with no buildings."""
    expected={c['id']:c['contentHash'] for c in manifest['cells']}
    actual={c['id']:c['contentHash'] for c in asset.get('sourceCells',[])}
    if (asset.get('version') != VERSION or asset.get('format') != 'facade-middle-v1' or not asset.get('complete')
        or asset.get('sourceWorldHash') != manifest['sourceWorldHash'] or actual != expected
        or len(asset['sourceCells']) != len(expected)):
        raise ValueError('Facade index differs from current source world/detail cells')
    owners={c['id']:c for c in manifest['cells'] if c['stats'].get('buildings')}
    groups={}
    for entry in asset['cells']:groups.setdefault(entry.get('parentId',entry['id']),[]).append(entry)
    if set(groups)!=set(owners):raise ValueError('Facade coverage incomplete')
    for owner,parts in groups.items():
        if len(parts)==1 and parts[0]['id']==owner and 'parentId' not in parts[0]:continue
        if sorted(p.get('buildingId','') for p in parts)!=sorted(owners[owner]['stats']['buildingIds']):
            raise ValueError('Incomplete building partition')
        if any(p['id']!=owner+'@'+p['buildingId'] for p in parts):raise ValueError('Invalid building partition ID')
    for cell in asset['cells']:
        owner=owners[cell.get('parentId',cell['id'])]
        split='parentId' in cell
        if (cell['sourceHash'] != owner['contentHash'] or (not split and cell['boundsEN'] != owner['boundsEN'])
            or cell['stats']['glazingTriangles'] != cell['stats']['sourceGlazingTriangles']
            or cell['stats']['buildings'] != (1 if split else owner['stats']['buildings'])):
            raise ValueError('Facade owner identity/topology mismatch')
        if cell['url'] != '/compare/facade-cell-'+cell['sha256'][:16]+'.bin.gz':
            raise ValueError('Unpinned facade data URL')
        if verify_files:
            path=local_path(public,cell['url']);digest=hashlib.sha256();count=0
            with path.open('rb') as stream:
                while block:=stream.read(1024*1024):digest.update(block);count+=len(block)
            if count != cell['compressedBytes'] or digest.hexdigest() != cell['sha256']:
                raise ValueError('Facade data length/hash mismatch')
    return asset


def repartition(public,index_url):
    """Reuse checked small owners when upgrading an already generated archive."""
    manifest=json.loads(local_path(public,index_url).read_bytes())
    if manifest.get('middlePartitions')!='building-v1':raise ValueError('Base facade atlas must retain per-building identities before partitioning')
    url=manifest['facadeMiddle']
    asset=json.loads(local_path(public,url).read_bytes())
    validate_asset(public,asset,manifest)
    owners={c['id']:c for c in manifest['cells']}
    split={c['id'] for c in asset['cells'] if not c.get('parentId') and c['stats']['buildings']>1 and over_budget(c,len(owners))}
    if not split:return asset,url
    entries=[]
    simplifier=SolidSimplifier()
    try:
        with tempfile.TemporaryDirectory(prefix='facade-split-') as tmp:
            for old in asset['cells']:
                if old['id'] not in split:
                    entries.append(old);continue
                owner=owners[old['id']]
                with tempfile.TemporaryFile() as src:
                    unpack_portable(public,owner,src)
                    source=Source(src)
                    for bid in sorted(owner['stats']['buildingIds']):
                        entries.extend(compile_entries(source,owner,Path(tmp),public,simplifier,len(owners),bid))
    finally:simplifier.close()
    totals={k:sum(c['compressedBytes'] if k=='compressedBytes' else c['stats'][k] for c in entries) for k in asset['stats']}
    # Every source face and glazing triangle belongs to exactly one component.
    for key in ('buildings','sourceTriangles','sourceGlazingTriangles','glazingTriangles','triangles'):
        if totals[key]!=asset['stats'][key]:raise ValueError('Partition lost source geometry')
    result={**asset,'cells':entries,'stats':totals}
    validate_asset(public,result,manifest)
    data=json_bytes(result);url='/compare/neighborhood-facades-'+hashlib.sha256(data).hexdigest()[:16]+'.json'
    immutable_write(local_path(public,url),data)
    return result,url


def publish(public, index_url='/compare/neighborhood-stream.json', asset_url=None):
    """Atomically attach a completed immutable sidecar; preserve every other field.

    An already generated asset can be supplied to avoid repeating the sequential
    build. No source/base/detail chunk is rewritten by this metadata operation.
    """
    public=Path(public).resolve();index_path=local_path(public,index_url)
    before=index_path.read_bytes();manifest=json.loads(before)
    if manifest.get('scope') != 'neighborhood-stream' or manifest.get('version') != 1:
        raise ValueError('Completed neighborhood index required')
    # Publishing is for mutable aliases, never an existing content hash URL.
    import re
    if re.search(r'-[a-f0-9]{16}\.json$',index_url):raise ValueError('Cannot mutate an immutable neighborhood index')
    if asset_url is None:asset,asset_url=package(public,index_url)
    else:
        if not re.fullmatch(r'/compare/neighborhood-facades-[a-f0-9]{16}\.json',asset_url):raise ValueError('Unpinned facade asset URL')
        data=local_path(public,asset_url).read_bytes()
        if hashlib.sha256(data).hexdigest()[:16] != Path(asset_url).stem.rsplit('-',1)[1]:raise ValueError('Facade index hash mismatch')
        asset=json.loads(data)
    validate_asset(public,asset,manifest)
    updated={**manifest,'facadeMiddle':asset_url};data=json_bytes(updated)
    immutable_url='/compare/neighborhood-stream-'+hashlib.sha256(data).hexdigest()[:16]+'.json'
    immutable_write(local_path(public,immutable_url),data)
    if index_path.read_bytes() != before:raise RuntimeError('Neighborhood changed during facade publication; alias untouched')
    with tempfile.NamedTemporaryFile(dir=index_path.parent,prefix='.facade-publish-',delete=False) as stream:
        temporary=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
    try:
        if index_path.read_bytes() != before:raise RuntimeError('Neighborhood changed before facade publication; alias untouched')
        temporary.replace(index_path)
    finally:temporary.unlink(missing_ok=True)
    return {'manifest':index_url,'immutableManifest':immutable_url,'facadeMiddle':asset_url,'baseAndDetailUnchanged':True,'stats':asset['stats']}


def audit(public, asset_url, index_url='/compare/neighborhood-stream.json'):
    """Compare every shipped glazing triangle to the decoded middle-tier output.

    Reads one original cell and one small facade binary at a time. This checks
    actual correspondence, beyond the generator's counts and analytic bounds.
    """
    public=Path(public).resolve();manifest=json.loads(local_path(public,index_url).read_bytes())
    asset=json.loads(local_path(public,asset_url).read_bytes());validate_asset(public,asset,manifest)
    original={c['id']:c for c in manifest['cells']};reports=[]
    for cell in asset['cells']:
        expected=[]
        with tempfile.TemporaryFile() as source_file:
            unpack_portable(public,original[cell.get('parentId',cell['id'])],source_file);source=Source(source_file)
            for node,matrix in flat_nodes(source.model):
                mesh=source.model['meshes'][node['mesh']]
                if cell.get('buildingId') and cell['buildingId']!=building_id(node,mesh):continue
                if not building_id(node,mesh) or layer_of(node,mesh) not in ('buildings','facades','facade'):continue
                for primitive in mesh['primitives']:
                    material=source.model['materials'][primitive['material']]
                    label=material.get('name','').lower()+' '+str(node.get('extras',{}).get('material','')).lower()
                    if primitive.get('mode',4)!=4 or not any(word in label for word in ('glass','glazing')):continue
                    p=source.array(primitive['attributes']['POSITION']);p=(p@matrix[:3,:3].T+matrix[:3,3]).astype('<f4')
                    ix=source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(len(p))
                    expected.append(p[ix])
        raw=gzip.decompress(local_path(public,cell['url']).read_bytes());actual=[]
        for batch in cell['batches']:
            if not batch['glazing']:continue
            n=batch['vertices'];offset=batch['offset']
            q=np.frombuffer(raw,dtype='<u2',count=n*3,offset=offset).reshape(-1,3)
            p=q*np.array(batch['step'])+batch['min']
            index_offset=(offset+n*12+3)//4*4
            ix=np.frombuffer(raw,dtype='<u2',count=batch['indices'],offset=index_offset)
            actual.append(p[ix])
        error=0.;count=0
        if expected or actual:
            expected=np.concatenate(expected);actual=np.concatenate(actual)
            if expected.shape!=actual.shape:raise ValueError('Glazing triangle order/count mismatch')
            error=float(np.abs(actual-expected).max());count=len(expected)//3
            if error>cell['stats']['quantizationErrorM']+1e-8:raise ValueError('Glazing position changed beyond quantization bound')
        reports.append({'id':cell['id'],'buildings':cell['stats']['buildings'],'glazingTrianglesCompared':count,'maxCoordinateErrorM':error})
    report={'version':1,'facadeMiddle':asset_url,'sourceWorldHash':manifest['sourceWorldHash'],'ownersCompared':len(reports),
            'buildings':sum(c['buildings'] for c in reports),'glazingTrianglesCompared':sum(c['glazingTrianglesCompared'] for c in reports),
            'maxGlazingCoordinateErrorM':max(c['maxCoordinateErrorM'] for c in reports),'cells':reports,
            'method':'Every source glazing triangle, in original order, against decoded output triangle coordinates; no nearest-neighbor approximation.'}
    data=json_bytes(report);url='/compare/neighborhood-facades-quality-'+hashlib.sha256(data).hexdigest()[:16]+'.json'
    immutable_write(local_path(public,url),data);print(json.dumps({'qualityReport':url,**{k:v for k,v in report.items() if k!='cells'}}),flush=True)
    return report,url


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('public',type=Path);cli.add_argument('--index',default='/compare/neighborhood-stream.json');cli.add_argument('--limit',type=int)
    cli.add_argument('--publish',action='store_true',help='Atomically set facadeMiddle on the matching mutable neighborhood alias')
    cli.add_argument('--asset',help='Reuse this completed immutable facade index instead of rebuilding')
    cli.add_argument('--audit',action='store_true',help='Compare every real source glazing triangle against the completed asset')
    cli.add_argument('--repartition',action='store_true',help='Emit an immutable middle index partitioning existing oversized owners')
    args=cli.parse_args()
    if args.limit is not None and args.publish:cli.error('Partial builds cannot publish')
    if args.asset and not (args.publish or args.audit):cli.error('--asset requires --publish or --audit')
    if args.repartition:
        result,url=repartition(args.public.resolve(),args.index)
        print(json.dumps({'index':url,'parts':len(result['cells']),'splitParts':sum('parentId' in c for c in result['cells'])}))
    elif args.audit:
        if not args.asset:cli.error('--audit requires --asset')
        audit(args.public,args.asset,args.index)
    elif args.publish:print(json.dumps(publish(args.public,args.index,args.asset),indent=2))
    else:package(args.public,args.index,args.limit)
