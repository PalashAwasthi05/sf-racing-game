"""Build a content-addressed, parallel-decodable display copy of a published world."""
import argparse
import concurrent.futures
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
import zlib

try:
    from startup_surfaces import compact_surfaces
    from startup_foliage import compact_foliage
    from startup_tints import compact_tints
except ModuleNotFoundError as exc:
    if exc.name not in ('numpy', 'PIL'):
        raise
    compact_foliage = compact_tints = None

VERSION = 3
CHUNK_SIZE = 8 * 1024 * 1024


def unpack_archive(public, manifest, output):
    decoder = zlib.decompressobj(31)
    for url in manifest['chunks']:
        part = (public / url.lstrip('/')).read_bytes()
        # Bound individual writes even when a gzip chunk expands enormously.
        while part:
            output.write(decoder.decompress(part, CHUNK_SIZE))
            part = decoder.unconsumed_tail
    output.write(decoder.flush())
    if not decoder.eof or output.tell() != manifest['uncompressedBytes']:
        raise ValueError('Incomplete source gzip')


def deduplicate_images(model, binary):
    """Share encoded imagery only across identical bytes and texture semantics."""
    original = model.get('images', [])
    textures = model.get('textures', [])
    roles = [set() for _ in textures]
    for material in model.get('materials', []):
        for part in (material, material.get('pbrMetallicRoughness', {})):
            for slot, ref in part.items():
                if slot.endswith('Texture') and isinstance(ref, dict) and 'index' in ref:
                    roles[ref['index']].add(slot)
    view_uses = {}
    for i, texture in enumerate(textures):
        view = original[texture['source']]['bufferView']
        state = {k: v for k, v in texture.items() if k != 'source'}
        key = (json.dumps(state, sort_keys=True), tuple(sorted(roles[i])))
        view_uses.setdefault(view, set()).add(key)

    old_views = model['bufferViews']
    data_by_view = {}
    def hashed(view):
        if view not in data_by_view:
            source = old_views[view]
            start = source.get('byteOffset', 0)
            data = binary[start:start + source['byteLength']]
            data_by_view[view] = (hashlib.sha256(data).digest(), data)
        return data_by_view[view]

    images, candidates, image_map = [], {}, []
    for image in original:
        view = image['bufferView']
        digest, data = hashed(view)
        metadata = {k: v for k, v in image.items() if k != 'bufferView'}
        view_metadata = {k: v for k, v in old_views[view].items() if k not in ('byteOffset', 'byteLength')}
        key = (digest, json.dumps([metadata, view_metadata], sort_keys=True), frozenset(view_uses.get(view, ())))
        # Keep a bytewise comparison even where the content digest collides.
        bucket = candidates.setdefault(key, [])
        match = next((i for i in bucket if images[i]['bufferView'] == view or
                      hashed(images[i]['bufferView'])[1] == data), None)
        if match is None:
            match = len(images)
            images.append(image)
            bucket.append(match)
        image_map.append(match)
    for texture in textures:
        texture['source'] = image_map[texture['source']]
    if 'images' in model:
        model['images'] = images
    return images


def prune_material_resources(model):
    """Remove dead material/texture/image dependencies after display transforms."""
    primitives = [p for mesh in model['meshes'] for p in mesh['primitives']]
    material_ids = sorted({p['material'] for p in primitives if 'material' in p})
    material_map = {old:new for new,old in enumerate(material_ids)}
    for primitive in primitives:
        if 'material' in primitive:
            primitive['material'] = material_map[primitive['material']]
    model['materials'] = [model.get('materials', [])[i] for i in material_ids]
    texture_refs = []
    def collect(value):
        for key, item in value.items():
            if key == 'extras':
                continue
            if key.endswith('Texture') and isinstance(item, dict) and 'index' in item:
                texture_refs.append(item)
            elif isinstance(item, dict):
                collect(item)
    for material in model['materials']:
        collect(material)
    texture_ids = sorted({r['index'] for r in texture_refs})
    texture_map = {old:new for new,old in enumerate(texture_ids)}
    for ref in texture_refs:
        ref['index'] = texture_map[ref['index']]
    model['textures'] = [model.get('textures', [])[i] for i in texture_ids]
    image_ids = sorted({t['source'] for t in model['textures']})
    image_map = {old:new for new,old in enumerate(image_ids)}
    for texture in model['textures']:
        texture['source'] = image_map[texture['source']]
    model['images'] = [model.get('images', [])[i] for i in image_ids]


def compact_geometry(model, binary):
    """Drop dependencies of replaced and unreachable meshes, preserving shared views."""
    added = compact_tints(model, binary)
    added.update(compact_foliage(model, binary))
    compact_surfaces(model, binary, added)
    used_meshes = sorted({n['mesh'] for n in model['nodes'] if 'mesh' in n})
    mesh_map = {old: new for new, old in enumerate(used_meshes)}
    for node in model['nodes']:
        if 'mesh' in node:
            node['mesh'] = mesh_map[node['mesh']]
    model['meshes'] = [model['meshes'][i] for i in used_meshes]
    primitives = [p for mesh in model['meshes'] for p in mesh['primitives']]
    indices = {a for p in primitives for a in p['attributes'].values()}
    indices.update(p['indices'] for p in primitives if 'indices' in p)
    accessor_map = {old: new for new, old in enumerate(sorted(indices))}
    for p in primitives:
        p['attributes'] = {key: accessor_map[i] for key, i in p['attributes'].items()}
        if 'indices' in p:
            p['indices'] = accessor_map[p['indices']]
    model['accessors'] = [model['accessors'][i] for i in sorted(indices)]

    prune_material_resources(model)
    images = deduplicate_images(model, binary)
    refs = model['accessors'] + images
    used_views = {r['bufferView'] for r in refs}
    # Put embedded photographs and maps first so Chrome can decode them
    # concurrently with the larger geometry transfers.
    image_views = {r['bufferView'] for r in images}
    order = sorted(used_views, key=lambda i: (i not in image_views, i))
    old_views = model['bufferViews']
    views, payload = [], []
    view_map = {}
    length = 0
    for old in order:
        view = old_views[old]
        offset, size = view.get('byteOffset', 0), view['byteLength']
        data = added[old] if old in added else binary[offset:offset + size]
        if len(data) != size:
            raise ValueError('Incomplete source buffer view')
        view_map[old] = len(views)
        views.append({**view, 'byteOffset': length})
        payload.append(data)
        length += size
        pad = -length % 4
        if pad:
            payload.append(bytes(pad))
            length += pad
    for r in refs:
        r['bufferView'] = view_map[r['bufferView']]
    model['bufferViews'] = views
    model['buffers'][0]['byteLength'] = length
    return payload, length


def has_extensions(value):
    if isinstance(value, dict):
        return bool(value.get('extensions')) or any(
            has_extensions(v) for k, v in value.items() if k != 'extras' and isinstance(v, (list, dict)))
    if isinstance(value, list):
        return any(has_extensions(v) for v in value if isinstance(v, (list, dict)))
    return False


def supported(model):
    # Other storage layouts keep the archival loading path. In particular,
    # animations, morphs and skins can reference accessors outside primitives.
    return (model.get('asset', {}).get('version') == '2.0'
            and not any(model.get(k) for k in ('extensionsUsed', 'extensionsRequired', 'animations', 'skins'))
            and not has_extensions(model)
            and len(model.get('buffers', [])) == 1 and 'uri' not in model['buffers'][0]
            and all(i.get('bufferView') is not None for i in model.get('images', []))
            and all('source' in t for t in model.get('textures', []))
            and all(v['buffer'] == 0 and 'extensions' not in v for v in model['bufferViews'])
            and all('bufferView' in a and 'sparse' not in a for a in model['accessors'])
            and all('targets' not in p and 'extensions' not in p for m in model['meshes'] for p in m['primitives']))


def pack_glb(raw, target):
    magic, version, total, json_len, chunk_type = struct.unpack_from('<5I', raw)
    if (magic, version, chunk_type) != (0x46546c67, 2, 0x4e4f534a) or total != len(raw):
        raise ValueError('Invalid source GLB header')
    model = json.loads(raw[20:20 + json_len])
    if not supported(model):
        return False
    size, chunk_type = struct.unpack_from('<II', raw, 20 + json_len)
    if chunk_type != 0x004e4942 or size != total - 28 - json_len:
        raise ValueError('Invalid source BIN')
    binary = memoryview(raw)[28 + json_len:]
    payload, length = compact_geometry(model, binary)
    text = json.dumps(model, ensure_ascii=False, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 4)
    target.write(struct.pack('<5I', magic, version, 28 + len(text) + length, len(text), 0x4e4f534a))
    target.write(text)
    target.write(struct.pack('<II', length, 0x004e4942))
    target.writelines(payload)
    return True


def gzip_chunk(source_path, offset, out):
    with source_path.open('rb') as src:
        src.seek(offset)
        data = src.read(CHUNK_SIZE)
    compressed = zlib.compressobj(3, wbits=31)
    out.write_bytes(compressed.compress(data) + compressed.flush())


def emit_chunks(packed, temp, public):
    size = packed.stat().st_size
    paths = [temp / f'{i}.gz' for i in range((size + CHUNK_SIZE - 1) // CHUNK_SIZE)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(gzip_chunk, packed, i * CHUNK_SIZE, f) for i, f in enumerate(paths)]
        for f in futures:
            f.result()
    digest = hashlib.sha256()
    sizes = []
    for path in paths:
        content = path.read_bytes()
        sizes.append(len(content))
        digest.update(content)
    hash_id = digest.hexdigest()[:12]
    names = [f'world-{hash_id}-{i:02}.bin' for i in range(len(paths))]
    for path, name in zip(paths, names):
        path.replace(public / name)
    return {'format': 'gzip-chunks', 'version': VERSION, 'contentHash': hash_id,
            'compressedBytes': sum(sizes), 'uncompressedBytes': size,
            'chunks': ['/' + name for name in names]}


def package(manifest_path, gpu_textures=False):
    manifest_path = Path(manifest_path).resolve()
    public = manifest_path.parent.parent if manifest_path.parent.name == 'compare' else manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    source_hash = manifest.get('contentHash', '')
    if not isinstance(source_hash, str) or not re.fullmatch('[a-f0-9]{12}', source_hash):
        return None
    descriptor = public / f'startup-v{VERSION}-{source_hash}.json'

    def complete(candidate):
        if not isinstance(candidate, dict) or candidate.get('version') != VERSION or candidate.get('sourceHash') != source_hash:
            return False
        hash_id = candidate.get('contentHash', '')
        paths = candidate.get('chunks', [])
        if not isinstance(hash_id, str) or not re.fullmatch('[a-f0-9]{12}', hash_id) or not paths:
            return False
        if paths != [f'/world-{hash_id}-{i:02}.bin' for i in range(len(paths))]:
            return False
        return all((public / url[1:]).is_file() for url in paths)

    derivative = manifest.get('startup')
    if not complete(derivative):
        derivative = json.loads(descriptor.read_text()) if descriptor.is_file() else None
    with tempfile.TemporaryDirectory(prefix='.world-startup-', dir=public) as scratch:
        temp = Path(scratch)
        if not complete(derivative):
            if compact_foliage is None:
                return None
            packed = temp / 'display.glb'
            with tempfile.TemporaryFile() as source:
                unpack_archive(public, manifest, source)
                with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as raw, packed.open('wb') as out:
                    if not pack_glb(raw, out):
                        return None
            derivative = emit_chunks(packed, temp, public)
            derivative['sourceHash'] = source_hash
        # Old immutable manifests can discover this version without rewriting
        # their bytes. Moving aliases also carry an inline copy for one less fetch.
        sidecar = temp / 'descriptor.json'
        sidecar.write_text(json.dumps(derivative, separators=(',', ':')) + '\n')
        sidecar.replace(descriptor)
        current = json.loads(manifest_path.read_text())
        if current.get('contentHash') != source_hash:
            raise RuntimeError('The source manifest changed during packaging')
        if current.get('startup') != derivative:
            current['startup'] = derivative
            updated = temp / 'manifest.json'
            updated.write_text(json.dumps(current, indent=1) + '\n')
            os.chmod(updated, manifest_path.stat().st_mode & 0o777)
            updated.replace(manifest_path)
        if gpu_textures:
            from startup_gpu_textures import package_gpu
            package_gpu(manifest_path)
        return derivative


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('manifest', type=Path)
    cli.add_argument('--gpu-textures', action='store_true', help='Also build the optional S3TC texture derivative; portable fallback remains available')
    args = cli.parse_args()
    result = package(args.manifest, gpu_textures=args.gpu_textures)
    print(f"{result['compressedBytes']:,} bytes in {len(result['chunks'])} parallel chunks"
          if result else 'Kept original package for this model.')
