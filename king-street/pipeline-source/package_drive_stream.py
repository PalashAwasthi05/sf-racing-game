"""Compile a topology-selected driving corridor as permanent support + exclusive streamed cells.

Offline only: sequential GLB decoding/copying, bounded by one source primitive.
Road records, origin, seed route and source package are inputs; no place names/coordinates
are embedded. Real intersection-to-intersection road records define the cell stations.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from pyproj import Transformer
from package_route_block import (Source, Writer, flat_nodes, node_bounds, building_id,
    touches, en_positions, local_path, json_bytes, write_json, unpack_portable,
    scope_world, points_touch, COPY_BYTES)
from package_startup import emit_chunks, prune_material_resources


def layer_of(node, mesh):
    return node.get('extras', {}).get('layer') or mesh.get('extras', {}).get('layer') or node.get('name', '').split('__')[0]


def station(points, route):
    """Nearest mapped-route station and unsigned lateral distance, vectorized."""
    points = np.asarray(points, dtype=float)
    best = np.full(len(points), np.inf); along = np.zeros(len(points)); offset = 0.
    for a, b in zip(route, route[1:]):
        a, b = np.asarray(a), np.asarray(b); d = b-a; length = np.linalg.norm(d)
        t = np.clip((points-a) @ d / (length*length), 0, 1)
        distance = np.linalg.norm(points-a-t[:, None]*d, axis=1)
        keep = distance < best
        best[keep] = distance[keep]; along[keep] = offset+t[keep]*length
        offset += length
    return along, best


def select_route(records, center, seed, context_radius, blocks=4, margin=42):
    project = Transformer.from_crs('EPSG:4326', 'EPSG:26910', always_xy=True)
    ox, oy = project.transform(center['lon'], center['lat'])
    roads = []
    heading = np.asarray(seed[-1],dtype=float)-seed[0]; heading /= np.linalg.norm(heading)
    for record in records:
        if record.get('active') is not True or record.get('classcode') in ('0', '1', '6'):
            continue
        pts = np.array([project.transform(*p[:2]) for p in record['line']['coordinates']])-[ox, oy]
        if len(pts) < 2 or np.linalg.norm(pts[-1]-pts[0]) < 20:
            continue
        one = record.get('oneway', '')
        if one == 'T': pts = pts[::-1]
        elif one != 'F' and (pts[-1]-pts[0]) @ heading < 0: pts = pts[::-1]
        direction = pts[-1]-pts[0]; direction /= np.linalg.norm(direction)
        if direction @ heading < math.cos(math.radians(15)):
            continue
        if np.linalg.norm(pts, axis=1).max() > context_radius-margin:
            continue
        roads.append((record, pts))
    if not roads: raise ValueError('No supported active road near seed')
    seed_point = np.asarray(seed).mean(axis=0)
    def distance(road):
        return float(station([seed_point], road[1])[1][0])
    nearest = min(range(len(roads)), key=lambda i: distance(roads[i]))
    if distance(roads[nearest]) > 12: raise ValueError('Seed does not align with mapped road')
    chain = [nearest]
    # First add predecessor, then successors; gives real blocks on both sides of seed.
    for backwards in [True] + [False]*(blocks-1):
        if len(chain) >= blocks: break
        edge = roads[chain[0 if backwards else -1]][1][0 if backwards else -1]
        candidates = [(float(np.linalg.norm(points[-1 if backwards else 0]-edge)), i)
                      for i, (_, points) in enumerate(roads) if i not in chain]
        if not candidates: continue
        gap, selected = min(candidates)
        if gap <= 2:
            chain.insert(0, selected) if backwards else chain.append(selected)
    while len(chain) < blocks:
        edge = roads[chain[-1]][1][-1]
        candidates = [(float(np.linalg.norm(points[0]-edge)), i) for i, (_,points) in enumerate(roads) if i not in chain]
        if not candidates or min(candidates)[0] > 2: raise ValueError('Insufficient connected straight mapped blocks')
        chain.append(min(candidates)[1])
    route=[]; cells=[]; offset=0.
    for index in chain:
        record, pts=roads[index]
        if route and np.linalg.norm(np.asarray(route[-1])-pts[0]) > .01: route.append(pts[0].tolist())
        if not route: route.append(pts[0].tolist())
        route.extend(pts[1:].tolist())
        length=sum(np.linalg.norm(b-a) for a,b in zip(pts,pts[1:]))
        cells.append({'id':'road-'+str(record['cnn']), 'roadId':str(record['cnn']), 'street':record.get('streetname'),
                      'fromStreet':record.get('f_st'), 'toStreet':record.get('t_st'),
                      'startM':offset, 'endM':offset+length, 'routeEN':pts.tolist()})
        offset+=length
    return route,cells


def convex_hull(points):
    pts=sorted(set(map(tuple, points)))
    if len(pts)<3: return np.asarray(pts)
    def cross(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves=[]
    for seq in (pts, pts[::-1]):
        half=[]
        for point in seq:
            while len(half)>1 and cross(half[-2],half[-1],point)<=0:half.pop()
            half.append(point)
        halves.append(half[:-1])
    return np.asarray(halves[0]+halves[1])


class Plan:
    def __init__(self, source, route, cells, margin):
        self.route=np.asarray(route,dtype=float);self.cells=cells;self.margin=margin
        self.ends=np.array([c['endM'] for c in cells]);self.owners={};self.hulls={};self.feature_stations={}
        self.box=[self.route[:,0].min()-margin,self.route[:,1].min()-margin,self.route[:,0].max()+margin,self.route[:,1].max()+margin]
        self.materials=copy.deepcopy(source.model.get('materials',[]))
        self.hull_materials=[]
        for cell in cells:
            self.hull_materials.append(len(self.materials))
            self.materials.append({'name':'drive_stream_collision_hull_'+cell['id'],'pbrMetallicRoughness':{'baseColorFactor':[.5,.5,.5,1],'roughnessFactor':1},'extras':{'class':'collision_hull','streamCell':cell['id']}})
        grouped={}
        nodes=flat_nodes(source.model)
        for node,matrix in nodes:
            mesh=source.model['meshes'][node['mesh']];bid=building_id(node,mesh)
            if not bid:continue
            lo,hi=node_bounds(source,node,matrix)
            if bid not in grouped:grouped[bid]=[lo,hi,[],np.inf,-np.inf]
            item=grouped[bid];item[0]=np.minimum(item[0],lo);item[1]=np.maximum(item[1],hi)
            # foundations are source footprint extrusions; use them for convex proxy outline.
            if layer_of(node,mesh)=='foundations':item[2].append((node,matrix))
            for primitive in mesh['primitives']:
                a=source.model['accessors'][primitive['attributes']['POSITION']]
                if 'min' in a and 'max' in a:
                    corners=np.array([[x,y,z] for x in (a['min'][0],a['max'][0]) for y in (a['min'][1],a['max'][1]) for z in (a['min'][2],a['max'][2])])
                    transformed=corners@matrix[:3,:3].T+matrix[:3,3]
                    item[3]=min(item[3],transformed[:,1].min());item[4]=max(item[4],transformed[:,1].max())
        for bid,(lo,hi,foundations,bottom,top) in grouped.items():
            # AABB radial test in route's straight frame includes complete edge buildings.
            corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]],(lo+hi)/2])
            _,dist=station(corners,self.route)
            if dist.min()>margin and not touches(lo,hi,self.box):continue
            # Avoid the route AABB's large empty wedge along an angled corridor.
            direction=self.route[-1]-self.route[0];direction/=np.linalg.norm(direction)
            rel=corners-self.route[0];long=rel@direction;lat=rel@np.array([-direction[1],direction[0]])
            if long.max() < -margin or long.min()>self.ends[-1]+margin or lat.min()>margin or lat.max() < -margin:continue
            projected,_=station(corners,self.route);self.feature_stations[bid]=(float(projected.min()),float(projected.max()),float(station([(lo+hi)/2],self.route)[0][0]))
            s,_=station([(lo+hi)/2],self.route);self.owners[bid]=int(np.clip(np.searchsorted(self.ends,s[0],side='right'),0,len(cells)-1))
            pts=[]
            for node,matrix in foundations:
                for primitive in source.model['meshes'][node['mesh']]['primitives']:
                    a=source.array(primitive['attributes']['POSITION']);world=a@matrix[:3,:3].T+matrix[:3,3]
                    pts.extend(world[:,[0,2]].tolist())
            if not pts:pts=[[lo[0],-lo[1]],[lo[0],-hi[1]],[hi[0],-lo[1]],[hi[0],-hi[1]]]
            self.hulls[bid]=(convex_hull(pts),float(bottom),float(top))

    def keep_node(self,node,mesh,bid,cell):
        if bid:
            if layer_of(node,mesh)=='foundations':return cell is None and bid in self.owners
            return cell is not None and self.owners.get(bid)==cell
        layer=layer_of(node,mesh);material={**mesh.get('extras',{}),**node.get('extras',{})}.get('material','')
        # Trunks remain visible + collidable in permanent support; foliage alone streams.
        streamed=layer=='vegetation' and material not in ('bark','trunk','wood')
        return streamed if cell is not None else not streamed

    def select(self,positions,matrix,indices,fan,mode,cell):
        en=en_positions(positions,matrix)
        if fan or mode==0:
            s,d=station(en,self.route);keep=d<=self.margin
            if cell is not None:keep &= np.clip(np.searchsorted(self.ends,s,side='right'),0,len(self.cells)-1)==cell
            used=np.flatnonzero(keep)
            return used,np.array([0,0,0],dtype=np.uint32) if fan else np.arange(len(used),dtype=np.uint32)
        tris=indices.reshape(-1,3);points=en[tris]
        # Oriented AABB intersection retains every crossing triangle exactly once.
        direction=self.route[-1]-self.route[0];direction/=np.linalg.norm(direction)
        rel=points-self.route[0];long=rel@direction;lat=rel@np.array([-direction[1],direction[0]])
        keep=(long.max(axis=1)>=-self.margin)&(long.min(axis=1)<=self.ends[-1]+self.margin)&(lat.min(axis=1)<=self.margin)&(lat.max(axis=1)>=-self.margin)
        if cell is not None:
            s,_=station(points.mean(axis=1),self.route)
            keep &= np.clip(np.searchsorted(self.ends,s,side='right'),0,len(self.cells)-1)==cell
        return np.unique(tris[keep].reshape(-1),return_inverse=True)

    def append_hulls(self,source,writer,nodes,meshes):
        for bid,(hull,bottom,top) in self.hulls.items():
            n=len(hull)
            if n<3:continue
            p=np.array([[x,y,z] for y in (bottom,top) for x,z in hull],dtype='<f4');idx=[]
            for i in range(n):
                j=(i+1)%n;idx.extend([i,j,j+n,i,j+n,i+n])
            for i in range(1,n-1):idx.extend([0,i+1,i,n,n+i,n+i+1])
            normals=np.zeros_like(p);normals[:,1]=1
            primitive={'attributes':{'POSITION':writer.accessor(p,{'componentType':5126,'type':'VEC3','min':[],'max':[]}), 'NORMAL':writer.accessor(normals,{'componentType':5126,'type':'VEC3'})},'indices':writer.accessor(np.array(idx,dtype='<u2').reshape(-1,1),{'componentType':5123,'type':'SCALAR'}),'material':self.hull_materials[self.owners[bid]]}
            extras={'layer':'buildings','featureId':bid,'class':'collision_hull','streamCell':self.cells[self.owners[bid]]['id']}
            nodes.append({'name':'buildings__'+bid+'___collision_hull','mesh':len(meshes),'extras':extras})
            meshes.append({'primitives':[primitive],'extras':extras})


def weld_support(attributes, indices, height_field=False):
    """Share byte-identical vertices while retaining the original face/UV seams."""
    positions = attributes['POSITION']
    columns = [np.ascontiguousarray(a).reshape(len(a), -1).view(np.uint8) for a in attributes.values()]
    if 'NORMAL' not in attributes and not height_field:
        # The viewer derives area-weighted vertex normals. At a curb two corners
        # can have identical UVs and positions but different incident faces.
        tri = indices.reshape(-1, 3)
        p, q, r = (positions[tri[:, i]].astype(np.float64) for i in range(3))
        face = np.cross(r - q, p - q)
        normals = np.zeros(positions.shape, dtype=np.float64)
        for i in range(3):
            np.add.at(normals, tri[:, i], face)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        columns.append(normals.astype('<f4').view(np.uint8))
    keys = np.ascontiguousarray(np.concatenate(columns, axis=1))
    rows = keys.view(np.dtype((np.void, keys.strides[0]))).ravel()
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    order = np.argsort(first)
    return first[order], np.argsort(order)[inverse.ravel()[indices]]


def compile_part(source, bin_stream, packed_stream, plan, cell=None):
    box = plan.box
    writer = Writer(bin_stream)
    model = source.model
    nodes = flat_nodes(model)
    out_nodes, out_meshes = [], []
    included_buildings = set()
    vertices = triangles = 0
    bounds_lo, bounds_hi = np.full(2,np.inf), np.full(2,-np.inf)
    layers = {}
    for node, matrix in nodes:
        mesh = model['meshes'][node['mesh']]
        bid = building_id(node, mesh)
        whole = bool(bid and bid in plan.owners and (
            (cell is None and layer_of(node,mesh)=='foundations') or
            (cell is not None and plan.owners[bid]==cell)))
        if not plan.keep_node(node, mesh, bid, cell) or (not whole and not touches(*node_bounds(source, node, matrix), box)):
            continue
        primitives = []
        for primitive in mesh['primitives']:
            attrs = primitive['attributes']
            positions = source.array(attrs['POSITION'])
            count = len(positions)
            fan = bool(node.get('extras', {}).get('foliageFanCentres') or mesh.get('extras', {}).get('foliageFanCentres'))
            indices = source.array(primitive['indices']).reshape(-1) if 'indices' in primitive else np.arange(count, dtype=np.uint32)
            if indices.dtype.kind != 'u' or (len(indices) and int(indices.max()) >= count):
                raise ValueError('Invalid mesh indices')
            mode = primitive.get('mode', 4)
            if mode == 4 and len(indices) % 3:
                raise ValueError('Incomplete triangle')
            if whole:
                used, order = np.arange(count), indices
            else:
                used, order = plan.select(positions, matrix, indices, fan, mode, cell)
            if cell is None and getattr(plan, 'weld_full_support', False) and mode == 4 and not fan and len(used) == count and count > 1000:
                arrays = {name: positions if name == 'POSITION' else source.array(aid) for name, aid in attrs.items()}
                used, order = weld_support(arrays, indices, layer_of(node, mesh) in ('terrain', 'landscape'))
                del arrays
            if not len(used):
                del positions, indices, used, order
                continue
            selected_en=en_positions(positions[used],matrix)
            bounds_lo=np.minimum(bounds_lo,selected_en.min(axis=0));bounds_hi=np.maximum(bounds_hi,selected_en.max(axis=0))
            del selected_en
            out = {k:copy.deepcopy(v) for k,v in primitive.items() if k not in ('attributes', 'indices')}
            out['attributes'] = {}
            for name, aid in attrs.items():
                array = positions if name == 'POSITION' else source.array(aid)
                if len(array) != count:
                    raise ValueError('Mismatched vertex attribute counts')
                out['attributes'][name] = writer.accessor(array[used], model['accessors'][aid])
                del array
            dtype, component = ('<u2', 5123) if len(used) <= 65536 else ('<u4', 5125)
            out['indices'] = writer.accessor(np.asarray(order, dtype=dtype).reshape(-1,1), {'componentType':component, 'type':'SCALAR'})
            primitives.append(out)
            vertices += len(used); triangles += len(order)//3 if mode == 4 else 0
            del positions, indices, used, order
        if primitives:
            layer=layer_of(node,mesh);layers[layer]=layers.get(layer,0)+1
            if bid:
                included_buildings.add(bid)
            out_node = {k:copy.deepcopy(v) for k,v in node.items() if k not in ('mesh', 'children', 'matrix', 'translation', 'rotation', 'scale')}
            out_node.update(mesh=len(out_meshes), matrix=matrix.reshape(-1, order='F').tolist())
            out_nodes.append(out_node)
            out_meshes.append({**{k:copy.deepcopy(v) for k,v in mesh.items() if k != 'primitives'}, 'primitives':primitives})
    if cell is None:
        plan.append_hulls(source, writer, out_nodes, out_meshes)
    if not out_nodes:
        raise ValueError('Route extent selected no meshes')
    result = copy.deepcopy(model)
    result['materials'] = copy.deepcopy(plan.materials)
    result.update(nodes=out_nodes, meshes=out_meshes, scenes=[{'nodes':list(range(len(out_nodes)))}], scene=0)
    prune_material_resources(result)
    copied_images = {}
    for image in result.get('images', []):
        if image.get('extras', {}).pop('neighborhoodWriterView', False):
            # Offline atlas builders already appended this view to this Writer.
            # Ordinary source image indices still follow the unchanged copy path.
            continue
        old = image['bufferView']
        start,length=source.view_range(old);source.stream.seek(source.offset+start);digest=hashlib.sha256()
        while length:
            chunk=source.stream.read(min(COPY_BYTES,length));digest.update(chunk);length-=len(chunk)
        image['extras']={**image.get('extras',{}),'viewerImageSha256':digest.hexdigest()}
        if old not in copied_images:
            copied_images[old] = writer.copy_view(source, old)
        image['bufferView'] = copied_images[old]
    result.update(accessors=writer.accessors, bufferViews=writer.views, buffers=[{'byteLength':bin_stream.tell()}])
    text = json_bytes(result).rstrip(b'\n'); text += b' ' * (-len(text)%4)
    packed_stream.write(struct.pack('<5I', 0x46546c67, 2, 28+len(text)+bin_stream.tell(), len(text), 0x4e4f534a))
    packed_stream.write(text); packed_stream.write(struct.pack('<II', bin_stream.tell(), 0x004e4942))
    bin_stream.seek(0); shutil.copyfileobj(bin_stream, packed_stream, COPY_BYTES)
    return {'meshes':len(out_meshes), 'images':len(result.get('images', [])), 'vertices':vertices, 'triangles':triangles,
            'buildings':len(included_buildings), 'buildingIds':sorted(included_buildings), 'layers':layers,
            'boundsEN':[*bounds_lo.tolist(),*bounds_hi.tolist()]} 


def emit_sequential(packed, temp, public):
    """Reuse the established wire format with one compressor allocation at a time."""
    from package_startup import gzip_chunk, CHUNK_SIZE, VERSION
    size=packed.stat().st_size; paths=[]; digest=hashlib.sha256(); total=0
    for i in range((size+CHUNK_SIZE-1)//CHUNK_SIZE):
        path=temp/f'part-{i}.gz';gzip_chunk(packed,i*CHUNK_SIZE,path)
        with path.open('rb') as stream:
            while data:=stream.read(COPY_BYTES):digest.update(data);total+=len(data)
        paths.append(path)
    hash_id=digest.hexdigest()[:12];urls=[]
    for i,path in enumerate(paths):
        name=f'world-{hash_id}-{i:02}.bin';path.replace(public/name);urls.append('/'+name)
    return {'format':'gzip-chunks','version':VERSION,'contentHash':hash_id,'compressedBytes':total,
            'uncompressedBytes':size,'chunks':urls}


def package(public, trial_path, streets_path, margin=42, blocks=4):
    public=Path(public).resolve();trial_file=local_path(public,trial_path)
    trial_bytes=trial_file.read_bytes();trial=json.loads(trial_bytes)
    source_file=local_path(public,trial['manifest']);source_bytes=source_file.read_bytes();original=json.loads(source_bytes)
    source_hash=original['contentHash'];portable=json.loads(local_path(public,f'/startup-v3-{source_hash}.json').read_text())
    if portable.get('version')!=3 or portable.get('sourceHash')!=source_hash or portable.get('format')!='gzip-chunks':
        raise ValueError('Pinned portable startup-v3 required')
    street_bytes=Path(streets_path).read_bytes();world=json.loads(local_path(public,original['world']).read_text())
    route,cells=select_route(json.loads(street_bytes),world['center'],trial['routeEN'],world['contextRadiusM'],blocks,margin)
    print(json.dumps({'phase':'route','cells':cells,'lengthM':cells[-1]['endM']}),flush=True)
    (public/'compare').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.drive-stream-',dir=public) as scratch:
        temp=Path(scratch)
        with (temp/'source.glb').open('w+b') as stream:
            unpack_portable(public,portable,stream);source=Source(stream)
            plan=Plan(source,route,cells,margin)
            print(json.dumps({'phase':'plan','buildings':len(plan.owners)}),flush=True)
            descriptors=[]
            for cell in [None]+list(range(len(cells))):
                name='base' if cell is None else cells[cell]['id'];packed=temp/(name+'.glb')
                with (temp/'binary').open('w+b') as binary, packed.open('wb') as output:
                    stats=compile_part(source,binary,output,plan,cell)
                descriptor=emit_sequential(packed,temp,public)
                descriptor['stats']=stats;descriptor['boundsEN']=stats['boundsEN']
                descriptor['sourceWorldHash']=source_hash
                if cell is None:base=descriptor
                else:descriptors.append({**cells[cell],**descriptor})
                packed.unlink();print(json.dumps({'phase':name,**stats,'compressedBytes':descriptor['compressedBytes']}),flush=True)
        scope_world(world,set(plan.owners),plan.box);world['scope']='drive-stream'
        world['streetViews']=[v for v in world.get('streetViews',[]) if station([v['position'][:2]],route)[1][0]<=margin]
        urls={'world':write_json(public,'drive-stream-world',world)}
        for key in ('cars','overhead'):
            if key not in original:continue
            data=json.loads(local_path(public,original[key]).read_text())
            if key=='cars':data['cars']=[c for c in data.get('cars',[]) if station([[c['x'],c['y']]],route)[1][0]<=margin+4]
            else:
                data['poles']=[p for p in data.get('poles',[]) if station([[p['x'],p['y']]],route)[1][0]<=margin+3]
                data['wires']=[w for w in data.get('wires',[]) if points_touch(w['p'],plan.box,w.get('r',0)+2)]
            urls[key]=write_json(public,'drive-stream-'+key,data)
        descriptor={'version':1,'scope':'drive-stream','sourceWorldHash':source_hash,
                    'sourcePortableHash':portable['contentHash'],'compilerVersion':1,'routeEN':route,
                    'lengthM':cells[-1]['endM'],'marginM':margin,'cells':descriptors,'base':base,**urls,
                    'roadSource':{'sha256':hashlib.sha256(street_bytes).hexdigest(),'selection':'active drivable mapped roads; seed heading; endpoint-connected; source context extent'},
                    'ownership':{'buildings':'complete feature owned by nearest route station of full feature bounds center',
                                 'foliage':'exclusive point/triangle ownership by nearest route station',
                                 'support':'all ground, mapped fixtures, original complete foundations and bark only in permanent base',
                                 'proxies':'lighting/LOD only: source-foundation convex footprint extrusion; per-cell material drive_stream_collision_hull_<cellid>; replace atomically with detailed cell'}}
        from scene_residency import annotate_manifest
        annotate_manifest(public,descriptor)
        immutable=write_json(public,'drive-stream',descriptor)
        if trial_file.read_bytes()!=trial_bytes or source_file.read_bytes()!=source_bytes or Path(streets_path).read_bytes()!=street_bytes:
            raise RuntimeError('Pinned compiler inputs changed; alias not published')
        alias=temp/'alias.json';alias.write_bytes(json_bytes(descriptor));alias.replace(public/'compare/drive-stream.json')
    return {'manifest':'/compare/drive-stream.json','immutableManifest':immutable,'lengthM':descriptor['lengthM'],
            'baseCompressedBytes':base['compressedBytes'],'cellsCompressedBytes':sum(c['compressedBytes'] for c in descriptors),
            'buildings':len(plan.owners),'cells':[{k:c[k] for k in ('id','startM','endM','compressedBytes','boundsEN')} for c in descriptors]}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('public',type=Path);cli.add_argument('trial');cli.add_argument('streets',type=Path)
    cli.add_argument('--margin',type=float,default=42);cli.add_argument('--blocks',type=int,default=4)
    args=cli.parse_args();print(json.dumps(package(args.public,args.trial,args.streets,args.margin,args.blocks),indent=2))
