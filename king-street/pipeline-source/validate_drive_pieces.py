"""Prove piece partition preserves the parent package's exact visible/collision geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from package_route_block import Source, flat_nodes, building_id, unpack_portable, local_path, json_bytes
from package_drive_stream import layer_of


def summarize(public,descriptor):
    with tempfile.TemporaryFile() as stream:
        unpack_portable(public,descriptor,stream);source=Source(stream);model=source.model
        features={};fans=[];triangles=0;centres=0;foundations=set();hull_owners={}
        def material(index):
            value=json.loads(json.dumps(model['materials'][index]))
            def replace(part):
                for key,entry in part.items():
                    if key.endswith('Texture') and isinstance(entry,dict) and 'index'in entry:
                        texture=model['textures'][entry['index']];image=model['images'][texture['source']]
                        entry['index']={'image':image['extras']['viewerImageSha256'],
                                        'sampler':model.get('samplers',[])[texture['sampler']] if 'sampler'in texture else None,
                                        'texture':{k:v for k,v in texture.items() if k not in ('source','sampler')}}
                    elif isinstance(entry,dict):replace(entry)
            replace(value);return value
        for node,matrix in flat_nodes(model):
            mesh=model['meshes'][node['mesh']];bid=building_id(node,mesh);layer=layer_of(node,mesh)
            if node.get('extras',{}).get('class')=='collision_hull':hull_owners[bid]=node['extras']['streamCell'];continue
            if layer=='foundations':foundations.add(bid)
            fan=bool(node.get('extras',{}).get('foliageFanCentres') or mesh.get('extras',{}).get('foliageFanCentres'))
            for primitive in mesh['primitives']:
                count=model['accessors'][primitive['attributes']['POSITION']]['count']
                if fan:
                    positions=source.array(primitive['attributes']['POSITION']);centres+=len(positions)
                    signature=json_bytes({'name':node.get('name'),'matrix':matrix.tolist(),'material':material(primitive['material'])})
                    fans.extend(hashlib.sha256(signature+row.tobytes()).hexdigest() for row in positions)
                    continue
                index=source.array(primitive['indices']).reshape(-1) if 'indices'in primitive else np.arange(count)
                if primitive.get('mode',4)==4:triangles+=len(index)//3
                digest=hashlib.sha256(json_bytes({'node':{k:v for k,v in node.items() if k not in ('mesh','matrix','translation','rotation','scale')},
                    'matrix':matrix.tolist(),'primitive':{k:v for k,v in primitive.items() if k not in ('attributes','indices','material')},
                    'material':material(primitive['material']) if 'material'in primitive else None}))
                for name,aid in sorted(primitive['attributes'].items()):
                    accessor=model['accessors'][aid]
                    digest.update(json_bytes([name,{k:v for k,v in accessor.items() if k not in ('bufferView','byteOffset','min','max')}]))
                    digest.update(source.array(aid).tobytes())
                digest.update(index.astype('<u4').tobytes())
                features.setdefault(bid or '__support__',[]).append(digest.hexdigest())
        return {'features':{bid:sorted(values) for bid,values in features.items()},'fans':sorted(fans),'sourceTriangles':triangles,
                'foliageCentres':centres,'renderTriangles':triangles+centres*4,'foundations':sorted(foundations),'hullOwners':hull_owners}


def validate(public):
    public=Path(public).resolve();data=json.loads((public/'compare/drive-stream.json').read_text());pieces=data['pieces']
    old=data['piecePartition'];parents=data['cells']
    if [c['contentHash'] for c in parents]!=old['parentContentHashes']:raise ValueError('Parent package hash changed')
    parent_features={};piece_features={};parent_fans=[];piece_fans=[];owners={};parent_triangles=piece_triangles=parent_centres=piece_centres=0
    for descriptors,features,fans,is_piece in [(parents,parent_features,parent_fans,False),(pieces,piece_features,piece_fans,True)]:
        previous=0.
        for descriptor in descriptors:
            if abs(descriptor['startM']-previous)>.001:raise ValueError('Partition is not contiguous')
            previous=descriptor['endM'];summary=summarize(public,descriptor)
            for bid,component_hashes in summary['features'].items():
                if bid in features:raise ValueError('Feature duplicated in partition')
                features[bid]=component_hashes
                if is_piece:owners[bid]=descriptor['id']
            fans.extend(summary['fans'])
            if is_piece:piece_triangles+=summary['sourceTriangles'];piece_centres+=summary['foliageCentres']
            else:parent_triangles+=summary['sourceTriangles'];parent_centres+=summary['foliageCentres']
            if is_piece:
                parent=next(c for c in parents if c['id']==descriptor['parentId'])
                if descriptor['startM']<parent['startM']-.001 or descriptor['endM']>parent['endM']+.001:raise ValueError('Piece crosses semantic block')
                if not 45<=descriptor['endM']-descriptor['startM']<=60:raise ValueError('Piece outside target length range')
    if parent_features!=piece_features:raise ValueError('Piece feature geometry/material bytes differ from parents')
    if sorted(parent_fans)!=sorted(piece_fans):raise ValueError('Piece canopy centres differ from parents')
    if parent_triangles!=piece_triangles or parent_centres!=piece_centres:raise ValueError('Source geometry counts differ')
    base=summarize(public,data['base']);old_base=summarize(public,old['parentBase'])
    if base['features']!=old_base['features'] or base['fans']!=old_base['fans']:raise ValueError('Permanent visible/collision support changed')
    if base['hullOwners']!=owners:raise ValueError('Proxy ownership differs from piece feature ownership')
    if len(base['foundations'])!=len(owners):raise ValueError('Incomplete permanent source foundations')
    return {'manifest':'/compare/drive-stream.json','pieces':len(pieces),'semanticBlocks':len(parents),'sourceWorldHash':data['sourceWorldHash'],
        'parentPackagesUnchanged':True,'exactFeatureGeometryAndMaterialsPreserved':True,'exactCanopyCentresPreserved':True,
        'exactPermanentCollisionAndVisibleSupportPreserved':True,'proxyOwnershipMatchesPieces':True,
        'buildings':len(owners),'sourceTrianglesBefore':parent_triangles,'sourceTrianglesAfter':piece_triangles,
        'foliageCentresBefore':parent_centres,'foliageCentresAfter':piece_centres,
        'renderTrianglesBefore':parent_triangles+parent_centres*4,'renderTrianglesAfter':piece_triangles+piece_centres*4,
        'triangleCounting':'Source triangles exclude degenerate compact-foliage Mesh sentinels; each retained fan centre regenerates four rendered triangles.',
        'parts':[{k:piece[k] for k in ('id','parentId','startM','endM','compressedBytes','stats')} for piece in pieces]}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('public',type=Path);cli.add_argument('--report',type=Path,required=True)
    args=cli.parse_args();report=validate(args.public);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='parts'},indent=2))
