"""Splice rebuilt buildings into a built area model, so a few buildings can be rebuilt and viewed in minutes instead of
rebuilding the area. The base model's nodes of the listed buildings (every part of each source outline) are dropped
and the rebuilt model's nodes of the same buildings are added with the meshes, accessors, materials, textures and
images they use. A material the base already has (names are unique per appearance) is reused, not copied. Both models
come out of the same materials step: one buffer each, every node a root node. The building lists of their world.json
are merged the same way.
Usage: splice.py <base candidate dir> <rebuilt candidate dir> <ids.json> <out candidate dir>
       (a candidate dir holds materials/candidate.glb and public/world.json)"""
import json, shutil, struct, sys
from pathlib import Path


def read(path):
    b = Path(path).read_bytes(); magic, _, _ = struct.unpack('<4sII', b[:12]); assert magic == b'glTF', path
    jl, _ = struct.unpack('<II', b[12:20]); j = json.loads(b[20:20 + jl]); o = 20 + jl
    bl, _ = struct.unpack('<II', b[o:o + 8]); return j, b[o + 8:o + 8 + bl]


def write(path, j, bin_):
    bin_ = bytes(bin_) + b'\0' * (-len(bin_) % 4); j['buffers'] = [{'byteLength': len(bin_)}]
    js = json.dumps(j, separators=(',', ':')).encode(); js += b' ' * (-len(js) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 28 + len(js) + len(bin_))); f.write(struct.pack('<II', len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack('<II', len(bin_), 0x004E4942)); f.write(bin_)


def source(name):
    """The source outline a node draws (`layer__<id>[_p<k>][__...]___material`), or None for a shared layer."""
    parts = (name or '').split('___')[0].split('__')
    return parts[1].split('_p')[0] if len(parts) > 1 else None


def splice(base, bbin, add, abin, ids):
    out = bytearray(bbin); out += b'\0' * (-len(out) % 4)
    for k in ('meshes', 'accessors', 'bufferViews', 'materials', 'textures', 'images', 'samplers', 'nodes'): base.setdefault(k, [])
    by_name = {m.get('name'): i for i, m in enumerate(base['materials']) if m.get('name')}
    memo = {k: {} for k in ('bufferViews', 'accessors', 'images', 'samplers', 'textures', 'materials', 'meshes')}

    def view(i):
        if i in memo['bufferViews']: return memo['bufferViews'][i]
        v = dict(add['bufferViews'][i]); s = v.get('byteOffset', 0); data = abin[s:s + v['byteLength']]
        out.extend(b'\0' * (-len(out) % 16)); v.update(buffer=0, byteOffset=len(out)); out.extend(data)
        base['bufferViews'].append(v); memo['bufferViews'][i] = len(base['bufferViews']) - 1; return memo['bufferViews'][i]

    def copy(kind, i, fix):
        if i in memo[kind]: return memo[kind][i]
        o = fix(json.loads(json.dumps(add[kind][i]))); base[kind].append(o); memo[kind][i] = len(base[kind]) - 1; return memo[kind][i]

    def accessor(i):
        def fix(a):
            if 'bufferView' in a: a['bufferView'] = view(a['bufferView'])
            assert 'sparse' not in a, 'sparse accessors are not spliced'
            return a
        return copy('accessors', i, fix)

    def image(i):
        def fix(m):
            if 'bufferView' in m: m['bufferView'] = view(m['bufferView'])
            return m
        return copy('images', i, fix)

    def texture(i):
        def fix(t):
            if 'source' in t: t['source'] = image(t['source'])
            if 'sampler' in t: t['sampler'] = copy('samplers', t['sampler'], lambda s: s)
            return t
        return copy('textures', i, fix)

    def textures_in(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith('Texture') and isinstance(v, dict) and 'index' in v: v['index'] = texture(v['index'])
                textures_in(v)
        elif isinstance(o, list):
            for v in o: textures_in(v)
        return o

    def material(i):
        name = add['materials'][i].get('name')
        if name in by_name: return by_name[name]
        k = copy('materials', i, textures_in); by_name[name] = k; return k

    def mesh(i):
        def fix(m):
            for p in m['primitives']:
                p['attributes'] = {a: accessor(x) for a, x in p['attributes'].items()}
                if 'indices' in p: p['indices'] = accessor(p['indices'])
                if 'material' in p: p['material'] = material(p['material'])
                assert 'targets' not in p, 'morph targets are not spliced'
            return m
        return copy('meshes', i, fix)

    scene = base['scenes'][base.get('scene', 0)]
    dropped = [i for i in scene['nodes'] if source(base['nodes'][i].get('name')) in ids]
    scene['nodes'] = [i for i in scene['nodes'] if source(base['nodes'][i].get('name')) not in ids]
    added = 0
    for i in add['scenes'][add.get('scene', 0)]['nodes']:
        n = add['nodes'][i]
        if source(n.get('name')) not in ids: continue
        assert 'children' not in n, 'nested nodes are not spliced'
        n = json.loads(json.dumps(n))
        if 'mesh' in n: n['mesh'] = mesh(n['mesh'])
        base['nodes'].append(n); scene['nodes'].append(len(base['nodes']) - 1); added += 1
    used = set(base.get('extensionsUsed', [])) | set(add.get('extensionsUsed', []))
    if used: base['extensionsUsed'] = sorted(used)
    return out, {'droppedNodes': len(dropped), 'addedNodes': added, 'newMaterials': len(memo['materials']), 'newImages': len(memo['images'])}


def main():
    base_dir, add_dir, ids_path, out_dir = map(Path, sys.argv[1:5])
    ids = {i.split('_p')[0] for i in json.load(open(ids_path))}
    base, bbin = read(base_dir / 'materials/candidate.glb'); add, abin = read(add_dir / 'materials/candidate.glb')
    out, stats = splice(base, bbin, add, abin, ids)
    (out_dir / 'materials').mkdir(parents=True, exist_ok=True); write(out_dir / 'materials/candidate.glb', base, out)
    # world data: the base's building list with the rebuilt buildings' entries in place of theirs
    (out_dir / 'public').mkdir(parents=True, exist_ok=True)
    w = json.loads((base_dir / 'public/world.json').read_text()); r = json.loads((add_dir / 'public/world.json').read_text())
    fresh = {b['id']: b for b in r['buildings'] if b['id'].split('_p')[0] in ids}
    w['buildings'] = [b for b in w['buildings'] if b['id'].split('_p')[0] not in ids] + list(fresh.values())
    (out_dir / 'public/world.json').write_text(json.dumps(w, separators=(',', ':')))
    for f in ('facade-evidence.json',):
        if (base_dir / 'public' / f).exists(): shutil.copyfile(base_dir / 'public' / f, out_dir / 'public' / f)
    print(json.dumps({**stats, 'worldBuildings': len(w['buildings']), 'rebuiltEntries': len(fresh), 'mb': round((out_dir / 'materials/candidate.glb').stat().st_size / 1e6)}))


if __name__ == '__main__':
    main()
