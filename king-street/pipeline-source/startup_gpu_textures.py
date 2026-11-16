"""Optional quality-gated S3TC derivative of a portable startup package.

The portable package remains the fallback. Original encoded imagery is retained
only as temporary CPU companions so the viewer's exact tint comparisons remain
unchanged; the viewer closes those bitmaps before its first render.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import mmap
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
from PIL import Image
from package_startup import CHUNK_SIZE, emit_chunks

PROFILE = 's3tc-v1'
ALGORITHM = 3


def image_roles(model):
    roles = [set() for _ in model.get('images', [])]
    for material in model.get('materials', []):
        for part in (material, material.get('pbrMetallicRoughness', {})):
            for slot, ref in part.items():
                if slot.endswith('Texture') and isinstance(ref, dict) and 'index' in ref:
                    roles[model['textures'][ref['index']]['source']].add(slot)
    return roles


def encode_level(image, pixel_format):
    output = io.BytesIO()
    image.save(output, format='DDS', pixel_format=pixel_format)
    return output.getvalue()


def mip_images(image, srgb=False, normal=False):
    yield image
    width, height = image.size
    while width > 1 or height > 1:
        width, height = max(1, width // 2), max(1, height // 2)
        # Filter color maps in linear light, as sRGB GPU textures require.
        # Normal/roughness data are already linear; do not renormalize normals.
        if srgb:
            rgba = np.asarray(image, dtype=np.float32) / 255
            rgb = rgba[..., :3]
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            reduced = np.stack([np.asarray(Image.fromarray(linear[..., c]).resize((width, height), Image.Resampling.BOX)) for c in range(3)], axis=2)
            encoded = np.where(reduced <= .0031308, reduced * 12.92, 1.055 * reduced ** (1 / 2.4) - .055)
            alpha = np.asarray(image.getchannel('A').resize((width, height), Image.Resampling.BOX))
            pixels = np.empty((height, width, 4), dtype=np.uint8)
            pixels[..., :3] = np.clip(np.rint(encoded * 255), 0, 255).astype(np.uint8)
            pixels[..., 3] = alpha
            image = Image.fromarray(pixels)
        else:
            image = image.resize((width, height), Image.Resampling.BOX)
        yield image


def quality(original, decoded, roles):
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(decoded, dtype=np.float32)
    error = np.abs(a - b)
    if float(error[..., 3].max()) > 1:
        return None
    color = bool(roles & {'baseColorTexture', 'emissiveTexture'})
    channels = [0, 1, 2]
    if roles and not color and 'normalTexture' not in roles:
        channels = sorted(({1, 2} if 'metallicRoughnessTexture' in roles else set()) |
                          ({0} if 'occlusionTexture' in roles else set())) or channels
    e = error[..., channels]
    rms = float(np.sqrt(np.mean(e * e)))
    p99 = float(np.percentile(e, 99))
    maximum = float(e.max())
    small = max(original.size) <= 16
    rms_limit = (4.5 if small else 3) if color else (3 if small else 1.5)
    if rms > rms_limit or p99 > (10 if color else 5) or maximum > (32 if color else 16):
        return None
    result = {'rmsByte': round(rms, 5), 'p99Byte': round(p99, 5), 'maxByte': maximum}
    if 'normalTexture' in roles:
        av, bv = a[..., :3] / 127.5 - 1, b[..., :3] / 127.5 - 1
        av /= np.maximum(np.linalg.norm(av, axis=2, keepdims=True), 1e-8)
        bv /= np.maximum(np.linalg.norm(bv, axis=2, keepdims=True), 1e-8)
        angles = np.degrees(np.arccos(np.clip(np.sum(av * bv, axis=2), -1, 1)))
        angular_rms, angular_p99, angular_max = float(np.sqrt(np.mean(angles * angles))), float(np.percentile(angles, 99)), float(angles.max())
        if angular_rms > (2 if small else .75) or angular_p99 > 2 or angular_max > 6:
            return None
        result.update(normalRmsDegrees=round(angular_rms, 5), normalP99Degrees=round(angular_p99, 5), normalMaxDegrees=round(angular_max, 5))
    return result


def compress_image(data, roles):
    image = Image.open(io.BytesIO(data)).convert('RGBA')
    width, height = image.size
    if min(width, height) < 4 or width & (width - 1) or height & (height - 1):
        return None
    pixel_format = 'DXT1' if image.getextrema()[3] == (255, 255) else 'DXT5'
    levels, metrics = [], []
    for level in mip_images(image, srgb=bool(roles & {'baseColorTexture', 'emissiveTexture'})):
        encoded = encode_level(level, pixel_format)
        decoded = Image.open(io.BytesIO(encoded)).convert('RGBA')
        measure = quality(level, decoded, roles)
        if measure is None:
            return None
        metrics.append(measure)
        levels.append(encoded)
    header = bytearray(levels[0][:128])
    flags = struct.unpack_from('<I', header, 8)[0] | 0x20000
    struct.pack_into('<I', header, 8, flags)
    struct.pack_into('<I', header, 28, len(levels))
    caps = struct.unpack_from('<I', header, 108)[0] | 0x8 | 0x400000
    struct.pack_into('<I', header, 108, caps)
    dds = bytes(header) + b''.join(level[128:] for level in levels)
    # Resident bytes, not encoded PNG size, determine whether compression helps.
    uncompressed = sum(max(1, width >> i) * max(1, height >> i) * 4 for i in range(len(levels)))
    return dds, {'format': pixel_format, 'width': width, 'height': height, 'mips': len(levels),
                 'rgbaMipBytes': uncompressed, 'compressedMipBytes': len(dds) - 128,
                 'quality': metrics}


def package_gpu(manifest_path, cache_dir=None):
    manifest_path = Path(manifest_path).resolve()
    public = manifest_path.parent.parent if manifest_path.parent.name == 'compare' else manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    portable = manifest.get('startup')
    if not portable or portable.get('format') != 'gzip-chunks':
        raise ValueError('Build the portable startup package first')
    existing = manifest.get('startupGpu')
    if existing and existing.get('portableHash') == portable['contentHash'] and existing.get('profile') == PROFILE and existing.get('algorithm') == ALGORITHM and all((public / u[1:]).is_file() for u in existing['chunks']):
        return existing
    cache = Path(cache_dir or tempfile.gettempdir()) / f'viewer-s3tc-v1-{ALGORITHM}'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gpu-startup-', dir=public) as scratch:
        temp = Path(scratch)
        source = temp / 'portable.glb'
        with source.open('wb') as out:
            for url in portable['chunks']:
                out.write(zlib.decompress((public / url.lstrip('/')).read_bytes(), 31))
        if source.stat().st_size != portable['uncompressedBytes']:
            raise ValueError('Incomplete portable startup package')
        with source.open('rb') as infile, mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as raw:
            magic, version, total, json_length, kind = struct.unpack_from('<5I', raw)
            if (magic, version, kind) != (0x46546c67, 2, 0x4e4f534a) or total != len(raw):
                raise ValueError('Invalid portable GLB')
            model = json.loads(raw[20:20 + json_length])
            binary = 28 + json_length
            roles = image_roles(model)
            def compile_one(pair):
                index, image = pair
                view = model['bufferViews'][image['bufferView']]
                start = binary + view.get('byteOffset', 0)
                data = raw[start:start + view['byteLength']]
                key = hashlib.sha256(PROFILE.encode() + json.dumps(sorted(roles[index])).encode() + data).hexdigest()
                meta, encoded = cache / (key + '.json'), cache / (key + '.dds')
                if meta.is_file():
                    measure = json.loads(meta.read_text())
                    if measure is None or encoded.is_file():
                        return index, encoded if measure else None, measure
                result = compress_image(data, roles[index])
                if result:
                    encoded.write_bytes(result[0]); meta.write_text(json.dumps(result[1]))
                    return index, encoded, result[1]
                meta.write_text('null')
                return index, None, None
            with ThreadPoolExecutor(max_workers=2) as pool:
                compiled = list(pool.map(compile_one, enumerate(model.get('images', []))))
            added, stats = {}, {'images': 0, 'rgbaMipBytes': 0, 'compressedMipBytes': 0, 'formats': {}}
            for index, path, measure in compiled:
                if path is None:
                    continue
                image = model['images'][index]
                original_view = image['bufferView']
                new_view = len(model['bufferViews'])
                model['bufferViews'].append({'buffer': 0, 'byteLength': path.stat().st_size})
                added[new_view] = path
                image['bufferView'] = new_view
                image['extras'] = {**image.get('extras', {}), 'viewerCpuBufferView': original_view,
                                   'viewerCpuMimeType': image['mimeType']}
                image['mimeType'] = 'image/vnd-ms.dds'
                stats['images'] += 1
                stats['rgbaMipBytes'] += measure['rgbaMipBytes']
                stats['compressedMipBytes'] += measure['compressedMipBytes']
                stats['formats'][measure['format']] = stats['formats'].get(measure['format'], 0) + 1
            references = model['accessors'] + model.get('images', [])
            used = {v['bufferView'] for v in references}
            used.update(image['extras']['viewerCpuBufferView'] for image in model.get('images', []) if 'viewerCpuBufferView' in image.get('extras', {}))
            image_views = {i['bufferView'] for i in model.get('images', [])}
            cpu_views = {i['extras']['viewerCpuBufferView'] for i in model.get('images', []) if 'viewerCpuBufferView' in i.get('extras', {})}
            order = sorted(used, key=lambda v: (v not in image_views | cpu_views, v))
            view_map, views, offset = {}, [], 0
            for old in order:
                view = model['bufferViews'][old]
                view_map[old] = len(views)
                views.append({**view, 'byteOffset': offset})
                offset += view['byteLength']; offset += -offset % 4
            for ref in references:
                ref['bufferView'] = view_map[ref['bufferView']]
            for image in model.get('images', []):
                if 'viewerCpuBufferView' in image.get('extras', {}):
                    image['extras']['viewerCpuBufferView'] = view_map[image['extras']['viewerCpuBufferView']]
            old_views = model['bufferViews']; model['bufferViews'] = views
            model['buffers'][0]['byteLength'] = offset
            text = json.dumps(model, separators=(',', ':'), ensure_ascii=False).encode(); text += b' ' * (-len(text) % 4)
            packed = temp / 'gpu.glb'
            with packed.open('wb') as out:
                out.write(struct.pack('<5I', magic, version, 28 + len(text) + offset, len(text), kind)); out.write(text)
                out.write(struct.pack('<II', offset, 0x004e4942))
                for old in order:
                    view = old_views[old]
                    if old in added:
                        out.write(added[old].read_bytes())
                    else:
                        start = binary + view.get('byteOffset', 0)
                        out.write(raw[start:start + view['byteLength']])
                    out.write(bytes(-out.tell() % 4))
        derivative = emit_chunks(packed, temp, public)
        derivative.update(version=3, profile=PROFILE, algorithm=ALGORITHM, sourceHash=manifest['contentHash'], portableHash=portable['contentHash'], textureStats=stats)
        descriptor = public / ('startup-gpu-' + manifest['contentHash'] + '.json')
        descriptor.write_text(json.dumps(derivative, separators=(',', ':')) + '\n')
        current = json.loads(manifest_path.read_text())
        if current.get('startup', {}).get('contentHash') != portable['contentHash']:
            raise RuntimeError('Portable package changed during GPU texture compilation')
        current['startupGpu'] = derivative
        manifest_path.write_text(json.dumps(current, indent=1) + '\n')
        return derivative


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('manifest', type=Path)
    cli.add_argument('--cache-dir', type=Path)
    args = cli.parse_args()
    print(json.dumps(package_gpu(args.manifest, args.cache_dir), indent=2))
