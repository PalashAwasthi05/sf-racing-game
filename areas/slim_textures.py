"""Shrink an area build's embedded colour textures for the local viewer: every image used only as a base colour is
resized to at most --max pixels a side and stored as JPEG (as PNG when it carries real transparency); normal,
roughness and other maps, geometry and materials are copied unchanged. A large area recolours hundreds of facade
textures, each a full-size PNG (Pacific Heights v2: 1,065 images, 947 MB), which the browser would hold as several GB
of GPU textures and the later steps decode in full.
Usage: slim_textures.py <in.glb> <out.glb> [--max 512] [--quality 88]"""
import argparse, io, json, struct
from PIL import Image

GLB, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--max', type=int, default=512); ap.add_argument('--quality', type=int, default=88)
a = ap.parse_args()
raw = open(a.src, 'rb').read()
magic, version, _ = struct.unpack_from('<III', raw, 0); assert magic == GLB and version == 2, 'not a glTF 2 binary'
jl, jt = struct.unpack_from('<II', raw, 12); assert jt == JSON_CHUNK
j = json.loads(raw[20:20 + jl]); bl, bt = struct.unpack_from('<II', raw, 20 + jl); assert bt == BIN_CHUNK; b0 = 28 + jl
tex, img = j.get('textures', []), j.get('images', [])
def view(ref): return img[tex[ref['index']]['source']].get('bufferView') if ref is not None and tex[ref['index']].get('source') is not None else None
color, other = set(), set()
for m in j.get('materials', []):
    color.add(view((m.get('pbrMetallicRoughness') or {}).get('baseColorTexture')))
    other |= {view((m.get('pbrMetallicRoughness') or {}).get('metallicRoughnessTexture'))} | {view(m.get(k)) for k in ('normalTexture', 'occlusionTexture', 'emissiveTexture')}
slim = (color - other) - {None}
assert all(v.get('buffer', 0) == 0 for v in j['bufferViews'])
out, kind, before, after = bytearray(), {}, 0, 0
for k, v in enumerate(j['bufferViews']):
    s = b0 + v.get('byteOffset', 0); data = raw[s:s + v['byteLength']]
    if k in slim:
        im = Image.open(io.BytesIO(data)); im.load(); before += len(data)
        alpha = 'A' in im.getbands() and im.getchannel('A').getextrema() != (255, 255)
        im = im.convert('RGBA' if alpha else 'RGB'); f = a.max / max(im.size)
        if f < 1: im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, 'PNG' if alpha else 'JPEG', **({} if alpha else {'quality': a.quality})); data = buf.getvalue()
        kind[k] = 'image/png' if alpha else 'image/jpeg'; after += len(data)
    out += b'\0' * (-len(out) % 4); v['byteOffset'] = len(out); v['byteLength'] = len(data); out += data
for i in img:
    if i.get('bufferView') in kind: i['mimeType'] = kind[i['bufferView']]
out += b'\0' * (-len(out) % 4); j['buffers'][0]['byteLength'] = len(out)
js = json.dumps(j, separators=(',', ':')).encode(); js += b' ' * (-len(js) % 4)
with open(a.dst, 'wb') as f:
    f.write(struct.pack('<III', GLB, 2, 28 + len(js) + len(out))); f.write(struct.pack('<II', len(js), JSON_CHUNK)); f.write(js)
    f.write(struct.pack('<II', len(out), BIN_CHUNK)); f.write(out)
print(json.dumps({'colourImages': len(slim), 'jpeg': sum(t == 'image/jpeg' for t in kind.values()), 'keptPngForAlpha': sum(t == 'image/png' for t in kind.values()),
                  'colourMB': [round(before / 1e6), round(after / 1e6)], 'fileMB': [round(len(raw) / 1e6), round((28 + len(js) + len(out)) / 1e6)]}))
