"""Which surfaces kept a real material. Wall-scale families (masonry, stone, concrete, render, panels, timber, roofing)
must end as a scanned or synthesized surface (colour, normal and roughness maps from the texture stages); ending as the
generator's procedural placeholder or a flat colour means the material was dropped (a colour variant nobody mapped, a
recipe no stage knows). Glass, frames, trims and lamps may stay flat: the viewer gives them their finish.
Reads only the GLB's JSON chunk. Usage: audit_materials_v30.py <glb> [--json out.json]; exits 1 when anything dropped."""
import argparse, collections, json, struct, sys

WALL_FAMILIES = ('brick', 'stone', 'limestone', 'cast_stone', 'concrete', 'venue_concrete', 'render', 'stucco', 'panel_', 'terracotta', 'timber', 'roofing')


def audit(path):
    f = open(path, 'rb'); f.seek(12); ln = struct.unpack('<I', f.read(4))[0]; f.seek(20); j = json.loads(f.read(ln))
    dropped = collections.Counter(); real = collections.Counter(); where = collections.defaultdict(set)
    for m in j['meshes']:
        name = m.get('name', ''); layer = name.split('__')[0]
        if layer not in ('buildings', 'facades', 'roofs'): continue
        src = name.split('___')[-1]; base = src.split('#')[0]
        if not base.startswith(WALL_FAMILIES): continue
        for p in m['primitives']:
            mt = j['materials'][p['material']] if 'material' in p else {}
            n = j['accessors'][p['indices']]['count'] // 3 if 'indices' in p else 0
            ok = mt.get('name', '').startswith(('auto_', 'scan_', 'surface_'))
            (real if ok else dropped)[src] += n
            if not ok: where[src].add(name.split('___')[0].split('__', 1)[-1][:40])
    return real, dropped, where


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('glb'); ap.add_argument('--json'); a = ap.parse_args()
    real, dropped, where = audit(a.glb)
    print(f'wall-scale surfaces with a real material: {sum(real.values())} triangles in {len(real)} source materials')
    print(f'dropped (placeholder or flat): {sum(dropped.values())} triangles in {len(dropped)} source materials')
    for k, v in dropped.most_common(20): print(f'   {k:34s} {v:8d}  e.g. {sorted(where[k])[:3]}')
    if a.json: json.dump({'real': dict(real), 'dropped': dict(dropped)}, open(a.json, 'w'), indent=1)
    sys.exit(1 if dropped else 0)


if __name__ == '__main__':
    main()
