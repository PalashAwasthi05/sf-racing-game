"""Bounded permanent skyline foliage from source leaf samples, not tree envelopes.

append_canopies(source, writer, nodes, meshes, plan) is called once while writing
base support. Outputs are ordinary meshes: runtime leaf expansion must NOT run
on them. Each belongs to the same streamCell as its corresponding detail samples.
All source leaf samples contribute to grounded crown alpha projections.
Source inputs/public files/colliders are never modified.
"""
import copy
import io
import math

import numpy as np

from package_drive_stream import layer_of

MIB = 1024 * 1024


def _centres(source, primitive, compact):
    """Return actual source sample positions and their attribute row indices."""
    p = source.array(primitive['attributes']['POSITION'])
    if compact or primitive.get('mode', 4) == 0:
        return p, np.arange(len(p))
    order = (source.array(primitive['indices']).reshape(-1) if 'indices' in primitive
             else np.arange(len(p)))
    if len(order) % 12 == 0 and len(order):
        fans = order.reshape(-1, 12)
        c = fans[:, 2]
        if all(np.array_equal(p[c], p[fans[:, k]]) for k in (5, 8, 11)):
            return p[c], c
    # Two-triangle source leaves: keep their true card centres, not triangle centres.
    if len(order) % 6 == 0 and len(order):
        cards = order.reshape(-1, 6)
        centres = np.empty((len(cards),3), dtype='<f4')
        for start in range(0,len(cards),65536):
            centres[start:start+65536] = p[cards[start:start+65536]].mean(axis=1)
        return centres, cards[:, 0]
    return np.empty((0, 3)), np.empty(0, dtype=int)


def _tints(source, primitive, rows):
    out = np.ones((len(rows), 4), dtype=np.float32)
    aid = primitive['attributes'].get('COLOR_0')
    if aid is not None:
        raw = source.array(aid)[rows]
        values = raw.astype(np.float32)
        if source.model['accessors'][aid].get('normalized') and raw.dtype.kind in 'ui':
            values /= np.iinfo(raw.dtype).max
        out[:, :values.shape[1]] *= values
    return np.clip(out, 0, 1)


def _thin(p, spacing, limit):
    """One existing sample per voxel; deterministic extrema preserve source bounds."""
    if not len(p):
        return np.empty(0, dtype=int)
    _, first = np.unique(np.floor(p / spacing).astype(np.int64), axis=0, return_index=True)
    extrema = np.unique(np.r_[p.argmin(axis=0), p.argmax(axis=0)])
    if len(first) > limit:
        first = first[np.linspace(0, len(first)-1, limit, dtype=int)]
    rest = np.setdiff1d(first, extrema, assume_unique=False)
    return np.r_[extrema[:limit], rest[:max(0, limit-len(extrema))]]



def _linear_to_srgb(color):
    c=np.clip(color,0,1)
    return np.rint(np.where(c<=.0031308,c*12.92,1.055*c**(1/2.4)-.055)*255).astype('u1')


def collect_canopies(source,plan):
    """Collect leaf samples and source bark only; never decode other world layers."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    leaves=[];leaf_colors=[];leaf_owners=[];barks=[];bark_colors=[];bark_owners=[]
    def owners(node,p):
        key=plan.node_groups.get(id(node));fixed=plan.owner_tiles.get(key) if key is not None else None
        if fixed is not None:return np.full(len(p),fixed,dtype='i4')
        tiles=np.floor(p[:,[0,2]]*[1,-1]/plan.tile_size).astype(int)
        return np.array([plan.tile_index[tuple(t)] for t in tiles],dtype='i4')
    for node,matrix in plan.nodes:
        mesh=source.model['meshes'][node['mesh']]
        if layer_of(node,mesh)!='vegetation':continue
        extra={**mesh.get('extras',{}),**node.get('extras',{})}
        name=str(extra.get('material') or node.get('name','').split('___')[-1])
        leaf=bool(extra.get('foliageFanCentres')) or name.startswith('leaf') or name=='frond'
        bark=name in ('bark','trunk','wood')
        if not (leaf or bark):continue
        for primitive in mesh['primitives']:
            old=primitive.get('material')
            if old is None:continue
            factor=np.array(source.model['materials'][old].get('pbrMetallicRoughness',{}).get('baseColorFactor',[1,1,1,1])[:3])
            if leaf:
                p,rows=_centres(source,primitive,bool(extra.get('foliageFanCentres')))
                if not len(p):continue
                p=(p@matrix[:3,:3].T+matrix[:3,3]).astype('f4')
                leaves.append(p);leaf_colors.append((_tints(source,primitive,rows)[:,:3]*factor).astype('f4'));leaf_owners.append(owners(node,p))
            else:
                p=source.array(primitive['attributes']['POSITION']);p=(p@matrix[:3,:3].T+matrix[:3,3]).astype('f4')
                order=source.array(primitive['indices']).reshape(-1,3) if 'indices' in primitive else np.arange(len(p)).reshape(-1,3)
                tri=p[order];barks.append(tri);bark_colors.append(np.tile(factor,(len(tri),1)).astype('f4'));bark_owners.append(owners(node,tri.mean(axis=1)))
    if not leaves:return None
    p=np.concatenate(leaves);colors=np.concatenate(leaf_colors);own=np.concatenate(leaf_owners)
    tri=np.concatenate(barks) if barks else np.empty((0,3,3),dtype='f4')
    bark_color=np.concatenate(bark_colors) if barks else np.empty((0,3),dtype='f4')
    bark_owner=np.concatenate(bark_owners) if barks else np.empty(0,dtype='i4')
    if len(tri):
        unique,inverse=np.unique(tri.reshape(-1,3),axis=0,return_inverse=True);indices=inverse.reshape(-1,3)
        row=np.r_[indices[:,0],indices[:,0]];col=np.r_[indices[:,1],indices[:,2]]
        count,label=connected_components(coo_matrix((np.ones(len(row),dtype='u1'),(row,col)),shape=(len(unique),len(unique))),directed=False)
        low=np.full((count,3),np.inf);high=np.full((count,3),-np.inf)
        np.minimum.at(low,label,unique);np.maximum.at(high,label,unique)
        # Branch-bearing source components are whole trees too: a narrow-width
        # filter incorrectly discarded most park trunks in the first prototype.
        roots=np.flatnonzero((high-low)[:,1]>=2)
        if not len(roots):roots=np.array([int(np.argmax((high-low)[:,1]))])
        bottom=unique[:,1]<=low[label,1]+.15
        total=np.zeros((count,2));number=np.bincount(label[bottom],minlength=count)
        np.add.at(total,label[bottom],unique[bottom][:,[0,2]])
        xz=total/np.maximum(number[:,None],1)
        anchors=np.column_stack([xz[roots,0],high[roots,1],xz[roots,1]])
        tree=cKDTree(anchors*[1,.35,1]);_,leaf_tree=tree.query(p*[1,.35,1])
        _,bark_tree=tree.query(tri.mean(axis=1)*[1,.35,1])
        component_to_tree=np.full(count,-1,dtype=int);component_to_tree[roots]=np.arange(len(roots))
        assigned=component_to_tree[label[indices[:,0]]];known=assigned>=0;bark_tree[known]=assigned[known]
    else:
        # Leaf-only sources stay coherent in small spatial crowns; no invented
        # trunk or ground extension is added when the source has no bark.
        _,leaf_tree=np.unique(np.floor(p/6).astype(int),axis=0,return_inverse=True)
        count=int(leaf_tree.max())+1;anchors=np.zeros((count,3));number=np.bincount(leaf_tree)
        np.add.at(anchors,leaf_tree,p);anchors/=number[:,None];bark_tree=np.empty(0,dtype=int)
    return dict(positions=p,colors=colors,owners=own,leaf_tree=leaf_tree,
                bark=tri,bark_colors=bark_color,bark_owners=bark_owner,bark_tree=bark_tree,anchors=anchors)


def crown_records(data):
    """Per-tree/per-owner views preserve the existing exclusive tile replacement."""
    groups={}
    for kind,trees,owners in [('leaves',data['leaf_tree'],data['owners']),('bark',data['bark_tree'],data['bark_owners'])]:
        order=np.lexsort((owners,trees));pairs=np.column_stack([trees[order],owners[order]])
        if not len(order):continue
        starts=np.r_[0,1+np.flatnonzero(np.any(pairs[1:]!=pairs[:-1],axis=1)),len(order)]
        for a,b in zip(starts[:-1],starts[1:]):
            key=tuple(map(int,pairs[a]));groups.setdefault(key,{})[kind]=order[a:b]
    records=[]
    for (tid,owner),group in sorted(groups.items()):
        leaf=group.get('leaves',np.empty(0,dtype=int));bark=group.get('bark',np.empty(0,dtype=int))
        points=[]
        if len(leaf):points.extend([data['positions'][leaf].min(0)-.72,data['positions'][leaf].max(0)+.72])
        if len(bark):points.extend([data['bark'][bark].min((0,1)),data['bark'][bark].max((0,1))])
        lo=np.min(points,0);hi=np.max(points,0)
        for axis in (0,2):
            records.append(dict(tree=tid,owner=owner,leaf=leaf,bark=bark,axis=axis,
                                lo=lo,hi=hi,widthM=float(hi[axis]-lo[axis]),heightM=float(hi[1]-lo[1])))
    return records


def pack_crowns(records,width=2048,height=1024):
    """One bounded atlas with the same metric resolution everywhere."""
    density=8.
    for _ in range(120):
        for r in records:
            r['size']=(max(6,min(96,math.ceil(r['widthM']*density))),max(8,min(128,math.ceil(r['heightM']*density))))
        x=y=row=0;fits=True
        for r in sorted(records,key=lambda r:(-r['size'][1],r['tree'],r['owner'],r['axis'])):
            w,h=r['size'];w+=2;h+=2
            if x+w>width:x=0;y+=row;row=0
            if y+h>height:fits=False;break
            r['atlas']=(x+1,y+1,w-2,h-2);x+=w;row=max(row,h)
        if fits:return density
        density*=.94
    raise ValueError('Too many source crowns for bounded atlas; increase budget or stream canopies')


def render_crown(data,record,size=None):
    """Orthographic alpha from ALL owned source samples and real bark triangles."""
    from PIL import Image,ImageDraw
    w,h=size or record['size'];axis=record['axis'];other=2 if axis==0 else 0
    lo,hi=record['lo'],record['hi'];sx=(w-1)/max(.01,hi[axis]-lo[axis]);sy=(h-1)/max(.01,hi[1]-lo[1])
    image=Image.new('RGBA',(w,h));draw=ImageDraw.Draw(image)
    leaves=record['leaf'];barks=record['bark']
    # Depth-order both source bark faces and leaf clusters. No random sampling,
    # convex canopy hull, density thinning or detached sample-card geometry.
    z=np.r_[data['bark'][barks,:,other].mean(1),data['positions'][leaves,other]]
    order=np.argsort(z,kind='stable');nb=len(barks)
    leaf_rgb=_linear_to_srgb(data['colors'][leaves]);bark_rgb=_linear_to_srgb(data['bark_colors'][barks])
    for index in order:
        if index<nb:
            t=data['bark'][barks[index]];xy=np.column_stack([(t[:,axis]-lo[axis])*sx,(hi[1]-t[:,1])*sy])
            draw.polygon([tuple(q) for q in xy],fill=tuple(bark_rgb[index])+(255,))
        else:
            k=index-nb;p=data['positions'][leaves[k]];x=(p[axis]-lo[axis])*sx;y=(hi[1]-p[1])*sy
            # A coverage footprint matching the runtime's 1.05-1.65m sprays.
            rx=max(.65,.66*sx);ry=max(.65,.51*sy)
            draw.ellipse((x-rx,y-ry,x+rx,y+ry),fill=tuple(leaf_rgb[k])+(255,))
    return image


def append_canopies(source,writer,nodes,meshes,plan,*,max_bytes=12*MIB,**_legacy):
    """Append grounded source-crown impostors, default <=12MiB incl texture mips.

    Generated images use the established neighborhoodWriterView resource path.
    The source bark itself is projected down to its original lowest vertex;
    no synthetic trunk or extension to a guessed terrain level is introduced.
    """
    import hashlib
    from PIL import Image
    data=collect_canopies(source,plan)
    stats=dict(sourceSamples=0,retainedSamples=0,meshes=0,geometryBytes=0,textureBytes=0,
               estimatedResidentBytes=0,crowns=0,barkTriangles=0,atlasWidth=0,atlasHeight=0)
    if data is None:return stats
    records=crown_records(data)
    width,height=(2048,1024) if max_bytes>=11*MIB else (1024,1024)
    texture_bytes=math.ceil(width*height*4*4/3)
    geometry_bound=len(records)*(4*19+6*4)
    if texture_bytes+geometry_bound>max_bytes:
        width=height=1024;texture_bytes=math.ceil(width*height*4*4/3)
    if texture_bytes+geometry_bound>max_bytes:raise ValueError('Canopy atlas and source cards exceed byte budget')
    density=pack_crowns(records,width,height)
    atlas=Image.new('RGBA',(width,height))
    for r in records:
        image=render_crown(data,r);x,y,_,_=r['atlas'];atlas.paste(image,(x,y))
    encoded=io.BytesIO();atlas.save(encoded,format='PNG',optimize=False);png=encoded.getvalue()
    model=source.model;images=model.setdefault('images',[]);image_id=len(images)
    images.append({'bufferView':writer.append(png),'mimeType':'image/png','extras':{'neighborhoodWriterView':True,'viewerImageSha256':hashlib.sha256(png).hexdigest()}})
    samplers=model.setdefault('samplers',[]);sampler=len(samplers);samplers.append({'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071})
    textures=model.setdefault('textures',[]);texture=len(textures);textures.append({'source':image_id,'sampler':sampler})
    grouped={}
    for r in records:grouped.setdefault(r['owner'],[]).append(r)
    mid=len(plan.materials)
    material_extra={'class':'distant_exterior','bakedAppearance':True,'canopyImpostor':True,'layer':'vegetation'}
    plan.materials.append({'name':'neighborhood_canopy_impostor','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'baseColorTexture':{'index':texture},'roughnessFactor':1,'metallicFactor':0},'alphaMode':'MASK','alphaCutoff':.35,'doubleSided':True,'extras':material_extra})
    for owner,items in sorted(grouped.items()):
        positions=[];normals=[];uvs=[];indices=[]
        for r in items:
            lo,hi=r['lo'],r['hi'];axis=r['axis'];other=2 if axis==0 else 0
            p=np.tile((lo+hi)/2,(4,1));p[:,axis]=[lo[axis],hi[axis],hi[axis],lo[axis]];p[:,1]=[lo[1],lo[1],hi[1],hi[1]]
            normal=np.zeros(3);normal[other]=1;normal[1]=.35;normal/=np.linalg.norm(normal)
            x,y,w,h=r['atlas'];uv=np.array([[x/width,(y+h)/height],[(x+w)/width,(y+h)/height],[(x+w)/width,y/height],[x/width,y/height]])
            index=np.array([0,1,2,0,2,3])+4*len(positions)
            if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),normal)<0:index=index.reshape(-1,3)[:,::-1].ravel()
            positions.append(p);normals.append(np.tile(normal,(4,1)));uvs.append(uv);indices.append(index)
        p=np.concatenate(positions).astype('<f4');normal=np.rint(np.concatenate(normals)*127).astype('i1');uv=np.rint(np.concatenate(uvs)*65535).astype('<u2')
        index=np.concatenate(indices).astype('<u2' if len(p)<=65535 else '<u4')
        cell=plan.cells[owner]['id'];extra={'class':'distant_exterior','streamCell':cell,'bakedAppearance':True,'canopyImpostor':True,'fallbackKind':'source_canopy_impostor','layer':'vegetation'}
        attrs={}
        for key,array,component,kind,normalized in [('POSITION',p,5126,'VEC3',False),('NORMAL',normal,5120,'VEC3',True),('TEXCOORD_0',uv,5123,'VEC2',True)]:
            meta={'componentType':component,'type':kind}
            if normalized:meta['normalized']=True
            if key=='POSITION':meta.update(min=[],max=[])
            attrs[key]=writer.accessor(array,meta)
        primitive={'attributes':attrs,'material':mid,'indices':writer.accessor(index[:,None],{'componentType':5123 if index.dtype.itemsize==2 else 5125,'type':'SCALAR'})}
        nodes.append({'name':'vegetation__'+cell+'___source_canopy_impostor','mesh':len(meshes),'extras':extra.copy()});meshes.append({'primitives':[primitive],'extras':extra.copy()})
        stats['geometryBytes']+=sum(a.nbytes for a in (p,normal,uv,index));stats['meshes']+=1
    stats.update(sourceSamples=len(data['positions']),retainedSamples=len(data['positions']),crowns=len(np.unique(data['leaf_tree'])),barkTriangles=len(data['bark']),textureBytes=texture_bytes,estimatedResidentBytes=texture_bytes+stats['geometryBytes'],atlasWidth=width,atlasHeight=height,pixelsPerM=density,cards=len(records),encodedBytes=len(png))
    return stats
