"""Add smaller whole-feature geometry pieces beneath unchanged semantic road blocks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from package_drive_stream import Plan, compile_part, emit_sequential
from package_route_block import Source, unpack_portable, local_path, json_bytes, write_json


def route_at(route, distance):
    remaining=distance
    for a,b in zip(route,route[1:]):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float);length=np.linalg.norm(b-a)
        if remaining<=length:return (a+(b-a)*remaining/length).tolist()
        remaining-=length
    return list(route[-1])


def subdivide_plan(plan, parents, source_materials, target=50):
    original_owners=dict(plan.owners);pieces=[]
    for parent_index,parent in enumerate(parents):
        start,end=parent['startM'],parent['endM'];length=end-start
        count=max(1,round(length/target));owners=[bid for bid,index in original_owners.items() if index==parent_index]
        centres=sorted(plan.feature_stations[bid][2] for bid in owners)
        cuts=[start]
        for i in range(1,count):
            ideal=start+length*i/count
            low=max(start+45*i,cuts[-1]+45);high=min(start+60*i,end-45*(count-i))
            if low>high:low,high=ideal,ideal
            candidates=[ideal]+[(a+b)/2 for a,b in zip(centres,centres[1:]) if low<=(a+b)/2<=high]
            candidates=[float(np.clip(cut,low,high)) for cut in candidates]
            def score(cut):
                crossing=sum(lo<cut<hi for bid in owners for lo,hi,_ in [plan.feature_stations[bid]])
                return (crossing,abs(cut-ideal))
            cuts.append(min(candidates,key=score))
        cuts.append(end)
        for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
            piece={k:copy.deepcopy(parent[k]) for k in ('roadId','street','fromStreet','toStreet') if k in parent}
            piece.update(id=parent['id']+f'-piece-{i+1:02}',parentId=parent['id'],startM=a,endM=b,
                         routeEN=[route_at(plan.route,a),route_at(plan.route,b)],pieceIndex=i,piecesInParent=count)
            pieces.append(piece)
    plan.cells=pieces;plan.ends=np.array([piece['endM'] for piece in pieces]);plan.materials=copy.deepcopy(source_materials);plan.hull_materials=[]
    for piece in pieces:
        plan.hull_materials.append(len(plan.materials));plan.materials.append({'name':'drive_stream_collision_hull_'+piece['id'],
            'pbrMetallicRoughness':{'baseColorFactor':[.5,.5,.5,1],'roughnessFactor':1},'extras':{'class':'collision_hull','streamCell':piece['id']}})
    for bid in plan.owners:
        parent_id=parents[original_owners[bid]]['id'];station=plan.feature_stations[bid][2]
        candidates=[i for i,piece in enumerate(pieces) if piece['parentId']==parent_id]
        plan.owners[bid]=next((i for i in candidates if station<pieces[i]['endM']),candidates[-1])
    return pieces


def package(public,target=50):
    public=Path(public).resolve();path=public/'compare/drive-stream.json';before=path.read_bytes();manifest=json.loads(before)
    if 'pieces' in manifest:raise ValueError('Piece manifest already exists; rerun from its pinned parent manifest explicitly')
    source_hash=manifest['sourceWorldHash'];portable=json.loads(local_path(public,'/startup-v3-'+source_hash+'.json').read_text())
    parents=copy.deepcopy(manifest['cells']);previous_base=copy.deepcopy(manifest['base']);outputs=[]
    with tempfile.TemporaryDirectory(prefix='.drive-pieces-',dir=public) as scratch:
        temp=Path(scratch)
        with (temp/'source.glb').open('w+b') as stream:
            unpack_portable(public,portable,stream);source=Source(stream);plan=Plan(source,manifest['routeEN'],parents,manifest['marginM'])
            pieces=subdivide_plan(plan,parents,source.model.get('materials',[]),target)
            print(json.dumps({'phase':'piece-plan','pieces':pieces,'buildings':[sum(owner==i for owner in plan.owners.values()) for i in range(len(pieces))]}),flush=True)
            for index in [None]+list(range(len(pieces))):
                name='base' if index is None else pieces[index]['id'];packed=temp/'part.glb'
                with (temp/'binary').open('w+b') as binary,packed.open('wb') as output:
                    stats=compile_part(source,binary,output,plan,index)
                descriptor=emit_sequential(packed,temp,public);descriptor.update(stats=stats,boundsEN=stats['boundsEN'],sourceWorldHash=source_hash)
                if index is None:manifest['base']=descriptor
                else:outputs.append({**pieces[index],**descriptor})
                print(json.dumps({'phase':name,'buildings':stats['buildings'],'triangles':stats['triangles'],'compressedBytes':descriptor['compressedBytes']}),flush=True)
        manifest['pieces']=outputs
        manifest['piecePartition']={'version':1,'targetLengthM':target,'parentManifestSha256':hashlib.sha256(before).hexdigest(),
            'cuts':'near target station, minimize projected footprint crossings; every whole feature assigned to exactly one piece',
            'parentPackagesUnchanged':True,'parentBase':previous_base,
            'parentContentHashes':[parent['contentHash'] for parent in parents]}
        if path.read_bytes()!=before:raise RuntimeError('Parent manifest changed during piece compilation')
        immutable=write_json(public,'drive-stream',manifest);alias=temp/'manifest.json';alias.write_bytes(json_bytes(manifest));alias.replace(path)
    return {'manifest':'/compare/drive-stream.json','immutableManifest':immutable,'pieces':len(outputs),
            'buildings':sum(piece['stats']['buildings'] for piece in outputs),
            'portableBytes':sum(piece['compressedBytes'] for piece in outputs)}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('public',type=Path);cli.add_argument('--target',type=float,default=50)
    args=cli.parse_args();print(json.dumps(package(args.public,args.target),indent=2))
