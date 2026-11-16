"""Sequential quality-gated BC7 derivatives; portable WebP assets remain fallback.

Build dependencies: ispc_texcomp==1.0.1, texture2ddecoder==1.0.6, Pillow>=11,
numpy. Only RGB contributes to these opaque parked materials; color and all
used surface RGB channels are gated at every mip.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import struct
import ispc_texcomp
import texture2ddecoder
import numpy as np
from PIL import Image



def bounded_mips(image, srgb):
    while True:
        yield image
        if image.size == (1,1):
            return
        size = tuple(max(1, dimension//2) for dimension in image.size)
        if not srgb:
            image = image.resize(size, Image.Resampling.BOX)
            continue
        # One channel at a time gives the same linear-light BOX filter without
        # simultaneously allocating float32 copies of the entire RGBA image.
        channels=[]
        for channel in range(3):
            encoded = np.asarray(image.getchannel(channel), dtype=np.float32)/255
            linear = np.where(encoded <= .04045, encoded/12.92, ((encoded+.055)/1.055)**2.4)
            reduced = np.asarray(Image.fromarray(linear).resize(size,Image.Resampling.BOX))
            result = np.where(reduced <= .0031308,reduced*12.92,1.055*reduced**(1/2.4)-.055)
            channels.append(Image.fromarray(np.clip(np.rint(result*255),0,255).astype(np.uint8)))
            del encoded,linear,reduced,result
        channels.append(image.getchannel('A').resize(size,Image.Resampling.BOX))
        image=Image.merge('RGBA',channels)


def quality(original, decoded, roles):
    # Same thresholds and percentile definition as startup_gpu_textures.quality;
    # a 256-bin histogram avoids several full-size float32 image copies.
    a,b=np.asarray(original),np.asarray(decoded)
    if np.abs(a[...,3].astype(np.int16)-b[...,3]).max()>1:
        return None
    histogram=np.zeros(256,dtype=np.int64)
    for channel in range(3):
        error=np.abs(a[...,channel].astype(np.int16)-b[...,channel]).astype(np.uint8)
        histogram+=np.bincount(error.reshape(-1),minlength=256)
    cumulative=histogram.cumsum();count=int(cumulative[-1]);target=(count-1)*.99
    lower=int(np.searchsorted(cumulative,int(np.floor(target))+1))
    upper=int(np.searchsorted(cumulative,int(np.ceil(target))+1))
    p99=lower+(upper-lower)*(target-np.floor(target))
    maximum=int(np.flatnonzero(histogram)[-1])
    rms=float(np.sqrt(np.sum(histogram*np.arange(256,dtype=np.int64)**2)/count))
    color='baseColorTexture' in roles;small=max(original.size)<=16
    limit=(4.5 if small else 3) if color else (3 if small else 1.5)
    if rms>limit or p99>(10 if color else 5) or maximum>(32 if color else 16):
        return None
    return {'rmsByte':round(rms,5),'p99Byte':round(float(p99),5),'maxByte':maximum}


def compile_atlas(path, srgb):
    image = Image.open(path).convert('RGBA')
    width, height = image.size
    if min(width,height)<4 or max(width,height)>8192 or width & (width-1) or height & (height-1):
        raise ValueError('Atlas must have power-of-two dimensions between4 and8192')
    levels, metrics = [], []
    for index, level in enumerate(bounded_mips(image, srgb)):
        w, h = level.size
        # BC blocks always cover four pixels per edge, including final 2/1 mips.
        padded = np.asarray(level)
        padded = np.pad(padded, ((0, max(0, 4-h)), (0, max(0, 4-w)), (0, 0)), mode='edge')
        encoded = ispc_texcomp.compress_blocks_bc7(ispc_texcomp.RGBASurface(padded.tobytes(), max(4,w), max(4,h)), ispc_texcomp.BC7EncSettings.from_profile('alpha_slow' if srgb and index < 7 else 'slow'))
        decoded = Image.frombytes('RGBA', (w,h), texture2ddecoder.decode_bc7(encoded,w,h), 'raw', 'BGRA')
        # Both maps serve the opaque parked paint material: no blending,
        # alphaTest, alphaHash, alpha-to-coverage or transmission. Three's OPAQUE
        # output forces alpha to1; only RGB affects display. The surface red
        # channel drives paint/clearcoat and is included in the stricter gate.
        decoded.putalpha(level.getchannel('A'))
        metric = quality(level, decoded, {'baseColorTexture'} if srgb else {'metallicRoughnessTexture','occlusionTexture'})
        if metric is None:
            a, b = np.asarray(level), np.asarray(decoded)
            diagnostics = []
            for channel in range(4 if srgb else 3):
                error = np.abs(a[...,channel].astype(np.float32)-b[...,channel])
                diagnostics.append({'rms':float(np.sqrt(np.mean(error*error))), 'p99':float(np.percentile(error,99)), 'max':float(error.max())})
            print('Rejected channel metrics RGB(A):', diagnostics, flush=True)
        print(path.name, index, metric, flush=True)
        if metric is None:
            return None
        metrics.append(metric); levels.append(encoded)
        # Darwin reports bytes. Keep this job comfortably below its 450 MiB budget.
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 440*1024*1024:
            raise MemoryError('BC7 encoder memory budget exceeded')
    payload = struct.pack('<6I', 0x37434250, 1, width, height, len(levels), int(srgb)) + b''.join(levels)
    return payload, metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    manifest={}
    for name,srgb in [('atlas-colour',True),('atlas-surface',False)]:
        result=compile_atlas(args.directory/(name+'.webp'),srgb)
        if result is None:
            print('Rejected',name,flush=True);continue
        payload,metrics=result
        output=name+'-'+hashlib.sha256(payload).hexdigest()[:16]+'.bc7'
        (args.directory/output).write_bytes(payload)
        manifest[name]={'file':output,'quality':metrics,'alphaPolicy':'unused-opaque-parked-material','sourceSha256':hashlib.sha256((args.directory/(name+'.webp')).read_bytes()).hexdigest()}
    (args.directory/'atlas-bc7.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Peak memory MiB:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, flush=True)

if __name__=='__main__':main()
