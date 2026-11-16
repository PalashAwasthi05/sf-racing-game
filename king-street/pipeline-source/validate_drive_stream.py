"""Validate published drive-stream packages without decoding images or using a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from package_route_block import Source, flat_nodes, node_bounds, building_id, unpack_portable, local_path, json_bytes
from package_drive_stream import layer_of, station, convex_hull


def validate(public, manifest='/compare/drive-stream.json'):
    public=Path(public).resolve();data=json.loads(local_path(public,manifest).read_text())
    if data.get('version')!=1 or data.get('scope')!='drive-stream':raise ValueError('Unexpected manifest')
    route=np.asarray(data['routeEN'],dtype=float);length=sum(np.linalg.norm(b-a) for a,b in zip(route,route[1:]))
    if abs(length-data['lengthM'])>.01:raise ValueError('Route length mismatch')
    previous=0.;owners={};hulls={};reports=[];road_triangles=[]
    for cell in data['cells']:
        if abs(cell['startM']-previous)>.001 or cell['endM']<=cell['startM']:raise ValueError('Invalid cell stations')
        previous=cell['endM']
    if abs(previous-length)>.01:raise ValueError('Cells do not cover route')
    with tempfile.TemporaryDirectory(prefix='drive-stream-validate-') as scratch:
        for cell in [data['base']]+data.get('pieces',data['cells']):
            is_base=cell is data['base'];part_id='base' if is_base else cell['id']
            with (Path(scratch)/'part.glb').open('w+b') as stream:
                unpack_portable(public,cell,stream);source=Source(stream);m=source.model
                nodes=flat_nodes(m)
                if len(nodes)!=len(m['nodes']) or any(n.get('children') for n in m['nodes']):raise ValueError('Non-flat scene')
                lo,hi=np.full(2,np.inf),np.full(2,-np.inf);buildings=set();layers={};tris=verts=0
                for image in m.get('images',[]):source.view_range(image['bufferView'])
                for node,matrix in nodes:
                    mesh=m['meshes'][node['mesh']];layer=layer_of(node,mesh);layers[layer]=layers.get(layer,0)+1
                    nlo,nhi=node_bounds(source,node,matrix);lo=np.minimum(lo,nlo);hi=np.maximum(hi,nhi)
                    proxy=node.get('extras',{}).get('class')=='collision_hull'
                    bid=building_id(node,mesh)
                    if proxy:
                        if not is_base:raise ValueError('Proxy in streamed scene')
                        hulls[bid]=node['extras']['streamCell']
                        for primitive in mesh['primitives']:
                            mat=m['materials'][primitive['material']]
                            if mat.get('name')!='drive_stream_collision_hull_'+hulls[bid]:raise ValueError('Proxy material grouping mismatch')
                    elif bid:
                        if is_base and layer!='foundations':raise ValueError('Detailed building unexpectedly permanent')
                        if not is_base:buildings.add(bid)
                    elif not is_base and layer!='vegetation':raise ValueError('Support unexpectedly duplicated in streamed cell')
                    for primitive in mesh['primitives']:
                        positions=source.array(primitive['attributes']['POSITION']);count=len(positions);verts+=count
                        for aid in primitive['attributes'].values():
                            if m['accessors'][aid]['count']!=count:raise ValueError('Attribute count mismatch')
                        indices=source.array(primitive['indices']).reshape(-1) if 'indices'in primitive else np.arange(count)
                        if len(indices) and indices.max()>=count:raise ValueError('Out-of-range index')
                        mode=primitive.get('mode',4)
                        if mode==4:
                            if len(indices)%3:raise ValueError('Incomplete triangle')
                            tris+=len(indices)//3
                        if is_base and layer=='roads':
                            world=positions@matrix[:3,:3].T+matrix[:3,3]
                            road_triangles.append(world[indices.reshape(-1,3)][:,:,[0,2]]*[1,-1])
                if np.any(lo<np.array(cell['boundsEN'][:2])-.01) or np.any(hi>np.array(cell['boundsEN'][2:])+.01):
                    # Support bounds previously described retained source geometry; include proxy hulls too.
                    if not is_base:raise ValueError('Manifest bounds omit geometry')
                gpu_checks=None
                if 'gpu' in cell:
                    gpu=cell['gpu']
                    if gpu.get('portableHash')!=cell['contentHash'] or gpu.get('profile')!='s3tc-v1':raise ValueError('GPU package pin mismatch')
                    with (Path(scratch)/'gpu.glb').open('w+b') as gpu_stream:
                        unpack_portable(public,gpu,gpu_stream);compressed=Source(gpu_stream);gm=compressed.model
                        for field in ('nodes','meshes','accessors'):
                            if gm[field]!=m[field]:raise ValueError('GPU package changed geometry metadata')
                        for view_id in {a['bufferView'] for a in m['accessors']}:
                            def digest_view(src,view):
                                offset,size=src.view_range(view);src.stream.seek(src.offset+offset);digest=hashlib.sha256()
                                while size:
                                    chunk=src.stream.read(min(size,1024*1024));digest.update(chunk);size-=len(chunk)
                                return digest.digest()
                            if digest_view(source,view_id)!=digest_view(compressed,view_id):raise ValueError('GPU package changed geometry bytes')
                        dds=0
                        for original,image in zip(m.get('images',[]),gm.get('images',[])):
                            cpu=image.get('extras',{}).get('viewerCpuBufferView',image['bufferView'])
                            if digest_view(source,original['bufferView'])!=digest_view(compressed,cpu):raise ValueError('Original image companion changed')
                            if image['mimeType']=='image/vnd-ms.dds':
                                dds+=1
                                if image.get('extras',{}).get('viewerCpuImageSha256')!=original.get('extras',{}).get('viewerImageSha256'):raise ValueError('CPU image hash metadata mismatch')
                            if digest_view(compressed,image['bufferView']).hex()!=image.get('extras',{}).get('viewerImageSha256'):raise ValueError('GPU image hash metadata mismatch')
                        gpu_checks={'contentHash':gpu['contentHash'],'ddsImages':dds,'exactGeometryBytes':True,'exactOriginalImageCompanions':True}
                for bid in buildings:
                    if bid in owners:raise ValueError('Feature duplicated across streamed cells')
                    owners[bid]=part_id
                reports.append({'id':part_id,'contentHash':cell['contentHash'],'compressedBytes':cell['compressedBytes'],
                                'uncompressedBytes':cell['uncompressedBytes'],'nodes':len(nodes),'images':len(m.get('images',[])),
                                'vertices':verts,'triangles':tris,'buildings':len(buildings),'layers':layers,
                                'actualBoundsEN':[*lo.tolist(),*hi.tolist()], 'gpu':gpu_checks})
    if owners!=hulls:raise ValueError('Detailed/proxy feature ownership differs')
    road=np.concatenate(road_triangles);a,b,c=road[:,0],road[:,1],road[:,2]
    det=(b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(c[:,0]-a[:,0])*(b[:,1]-a[:,1])
    good=np.abs(det)>1e-9;a,b,c,det=a[good],b[good],c[good],det[good]
    samples=[]
    for p,q in zip(route,route[1:]):
        for t in np.linspace(0,1,int(np.ceil(np.linalg.norm(q-p)/2))+1):samples.append(p+(q-p)*t)
    misses=[]
    for point in samples:
        u=((b[:,0]-point[0])*(c[:,1]-point[1])-(c[:,0]-point[0])*(b[:,1]-point[1]))/det
        v=((c[:,0]-point[0])*(a[:,1]-point[1])-(a[:,0]-point[0])*(c[:,1]-point[1]))/det
        if not np.any((u>=-1e-5)&(v>=-1e-5)&(u+v<=1+1e-5)):misses.append(point.tolist())
    if misses:raise ValueError(f'{len(misses)} route samples outside retained road surfaces: {misses[:5]}')
    return {'manifest':manifest,'sourceWorldHash':data['sourceWorldHash'],'lengthM':length,'roadSampleSpacingM':2,
            'roadSamples':len(samples),'roadSamplesCovered':len(samples)-len(misses),'buildings':len(owners),
            'exclusiveBuildingOwnership':True,'proxyOwnershipMatches':True,'flatSupportedGLBs':True,
            'supportExclusiveToBase':True,'parts':reports,'cells':[{k:c[k] for k in ('id','street','fromStreet','toStreet','startM','endM')} for c in data['cells']],
            'limitations':['Lighting/LOD proxies use convex source-foundation footprints and simple extrusions; collision preserves original complete foundations.',
                            'Collision and visual GPU behavior require the separate guarded driving verification.',
                            'Mapped roads support route selection; traffic rules and signals are simulated by the viewer.']}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('public',type=Path);cli.add_argument('--report',type=Path,required=True)
    args=cli.parse_args();report=validate(args.public);args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('parts','cells')},indent=2))
