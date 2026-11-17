"""Repack exact quality-gated DDS views from a source world's optional GPU package.

Keeps portable geometry and original encoded CPU image companions byte-identical.
No image decode, mip generation, quality relaxation, GPU work or parallel jobs.
"""
import argparse
import copy
import hashlib
import io
import json
import re
from pathlib import Path
import shutil
import struct
import tempfile
import time

from PIL import Image
from package_route_block import Source, local_path, unpack_portable, json_bytes, write_json, COPY_BYTES
from package_drive_stream import emit_sequential
from startup_gpu_textures import image_roles, PROFILE, ALGORITHM


def view_hash(source,index):
    start,length=source.view_range(index);source.stream.seek(source.offset+start);digest=hashlib.sha256()
    while length:
        data=source.stream.read(min(COPY_BYTES,length))
        if not data:raise ValueError('Truncated image')
        digest.update(data);length-=len(data)
    return digest.hexdigest()


def view_bytes(source,index):
    start,length=source.view_range(index);source.stream.seek(source.offset+start);return source.stream.read(length)


def package(public,report,stream='drive-stream',alias=None):
    if stream not in ('drive-stream','neighborhood-stream'):
        raise ValueError('Unrecognized spatial package')
    alias=alias or stream
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*-stream',alias) or (alias!=stream and stream!='neighborhood-stream'):
        raise ValueError('Invalid spatial package alias')
    started=time.monotonic();public=Path(public).resolve();path=public/'compare'/(alias+'.json');before=path.read_bytes();manifest=json.loads(before)
    if manifest['scope'] != stream:
        raise ValueError('Manifest scope mismatch')
    gpu=json.loads(local_path(public,'/startup-gpu-'+manifest['sourceWorldHash']+'.json').read_text())
    if gpu.get('profile')!=PROFILE or gpu.get('algorithm')!=ALGORITHM or gpu.get('portableHash')!=manifest['sourcePortableHash']:
        raise ValueError('Matching quality-gated source GPU package required')
    reports=[]
    with tempfile.TemporaryDirectory(prefix='.drive-stream-gpu-',dir=public) as scratch:
        temp=Path(scratch)
        with (temp/'source-gpu.glb').open('w+b') as source_stream:
            unpack_portable(public,gpu,source_stream);source=Source(source_stream);roles=image_roles(source.model);lookup={}
            for i,image in enumerate(source.model.get('images',[])):
                extras=image.get('extras',{});original=extras.get('viewerCpuBufferView',image['bufferView'])
                key=(view_hash(source,original),tuple(sorted(roles[i])))
                lookup[key]=image['bufferView'] if 'viewerCpuBufferView' in extras else None
            print(json.dumps({'phase':'indexed-source','images':len(lookup),'seconds':round(time.monotonic()-started,2)}),flush=True)
            for entry in [manifest['base']]+manifest['cells']+manifest.get('pieces',[]):
                existing=entry.get('gpu')
                if existing and existing.get('portableHash')==entry['contentHash'] and existing.get('profile')==PROFILE and all(local_path(public,u).is_file() for u in existing['chunks']):
                    print(json.dumps({'phase':'kept-existing-gpu','id':entry.get('id','base')}),flush=True)
                    continue
                part_start=time.monotonic();part_id=entry.get('id','base')
                with (temp/'portable.glb').open('w+b') as portable_stream:
                    unpack_portable(public,entry,portable_stream);target=Source(portable_stream);model=copy.deepcopy(target.model);target_roles=image_roles(model)
                    stats={'images':0,'rgbaMipBytes':0,'compressedMipBytes':0,'formats':{},'reusedSourceGpuHash':gpu['contentHash']}
                    total_rgba=0;fallback_rgba=0;misses=0
                    with (temp/'binary').open('w+b') as binary:
                        target.stream.seek(target.offset)
                        remaining=target.size
                        while remaining:
                            data=target.stream.read(min(COPY_BYTES,remaining));binary.write(data);remaining-=len(data)
                        for i,image in enumerate(model.get('images',[])):
                            old=image['bufferView'];original=view_bytes(target,old)
                            header=Image.open(io.BytesIO(original));w,h=header.size
                            rgba=sum(max(1,w>>level)*max(1,h>>level)*4 for level in range(max(w,h).bit_length()))
                            total_rgba+=rgba
                            key=(hashlib.sha256(original).hexdigest(),tuple(sorted(target_roles[i])))
                            found=lookup.get(key)
                            if key not in lookup:misses+=1
                            if found is None:fallback_rgba+=rgba;continue
                            encoded=view_bytes(source,found)
                            if encoded[:4]!=b'DDS ':raise ValueError('Source GPU image is not DDS')
                            height,width=struct.unpack_from('<II',encoded,12);mips=struct.unpack_from('<I',encoded,28)[0]
                            if (width,height)!=(w,h):raise ValueError('Source DDS dimensions differ')
                            if mips!=max(w,h).bit_length():raise ValueError('Incomplete DDS mip chain')
                            fmt=encoded[84:88].decode('ascii');offset=binary.tell();binary.write(encoded);binary.write(bytes(-len(encoded)%4))
                            view=len(model['bufferViews']);model['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(encoded)})
                            image.update(bufferView=view,mimeType='image/vnd-ms.dds',extras={**image.get('extras',{}),'viewerCpuBufferView':old,'viewerCpuMimeType':image['mimeType'],'viewerCpuImageSha256':hashlib.sha256(original).hexdigest(),'viewerImageSha256':hashlib.sha256(encoded).hexdigest()})
                            stats['images']+=1;stats['rgbaMipBytes']+=rgba;stats['compressedMipBytes']+=len(encoded)-128
                            stats['formats'][fmt]=stats['formats'].get(fmt,0)+1
                        model['buffers'][0]['byteLength']=binary.tell();text=json_bytes(model).rstrip(b'\n');text+=b' '*(-len(text)%4)
                        packed=temp/'gpu.glb'
                        with packed.open('wb') as out:
                            out.write(struct.pack('<5I',0x46546c67,2,28+len(text)+binary.tell(),len(text),0x4e4f534a));out.write(text)
                            out.write(struct.pack('<II',binary.tell(),0x004e4942));binary.seek(0);shutil.copyfileobj(binary,out,COPY_BYTES)
                derivative=emit_sequential(packed,temp,public)
                derivative.update(version=3,profile=PROFILE,algorithm=ALGORITHM,sourceHash=manifest['sourceWorldHash'],portableHash=entry['contentHash'],textureStats=stats)
                entry['gpu']=derivative
                result={'id':part_id,'portableHash':entry['contentHash'],'gpuHash':derivative['contentHash'],'seconds':round(time.monotonic()-part_start,2),
                        'images':len(model.get('images',[])),'ddsImages':stats['images'],'missingSourceImageKeys':misses,
                        'portableTransferBytes':entry['compressedBytes'],'gpuTransferBytes':derivative['compressedBytes'],
                        'portableRGBAMipBytes':total_rgba,'gpuResidentMipBytes':stats['compressedMipBytes']+fallback_rgba,
                        'uncompressedImagesRGBAMipBytes':fallback_rgba}
                reports.append(result);print(json.dumps(result),flush=True)
        from scene_residency import annotate_manifest
        annotate_manifest(public, manifest)
        if path.read_bytes()!=before:raise RuntimeError('Portable stream manifest changed during GPU repack')
        immutable=write_json(public,stream,manifest)
        updated=temp/'manifest.json';updated.write_bytes(json_bytes(manifest));updated.replace(path)
    result={'manifest':'/compare/'+alias+'.json','immutableManifest':immutable,'sourceGpuHash':gpu['contentHash'],
            'quality':'Exact stored source DDS mip bytes; source profile s3tc-v1 algorithm3 quality gate reused. Original image companions retained.',
            'seconds':round(time.monotonic()-started,2),'parts':reports,
            'estimates':'Texture resident bytes include complete mip chains and uncompressed fallbacks; excludes renderer geometry, targets, driver overhead and transient image companions.'}
    # Residual appearance is bound to the finished portable piece identities.
    if stream=='drive-stream':
        from package_drive_residuals import package as package_residuals
        residuals = package_residuals(public)
        published_index = path.read_bytes()
        result['immutableManifest'] = '/compare/drive-stream-' + hashlib.sha256(published_index).hexdigest()[:16] + '.json'
        result['distantExteriors'] = {'manifest': json.loads(published_index)['distantExteriors'], 'bytes': residuals['byteLength'], 'stats': residuals['stats']}
    result['seconds'] = round(time.monotonic()-started, 2)
    Path(report).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('public',type=Path);cli.add_argument('--report',type=Path,required=True)
    cli.add_argument('--stream',choices=['drive-stream','neighborhood-stream'],default='drive-stream')
    cli.add_argument('--alias',help='Separate area package alias')
    args=cli.parse_args();print(json.dumps(package(args.public,args.report,args.stream,args.alias),indent=2))
