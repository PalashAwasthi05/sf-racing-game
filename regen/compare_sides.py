"""Each side's photograph next to the build drawn square-on in the same frame (judge_facades_v27.render_wall), so a
reading can be checked against what the generator made of it. One build in memory."""
import argparse, json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import judge_facades_v27 as J, auto_read_frontages_v24 as ar
ap = argparse.ArgumentParser(); ap.add_argument('--build', required=True); ap.add_argument('--ids', nargs='+'); ap.add_argument('--out', required=True); a = ap.parse_args()
root = pathlib.Path('regen/v1'); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
B = J.load_build(a.build); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
for i in a.ids:
    p = byid[i]; v = p['views'][0]
    r = json.load(open(root / 'final' / f'{i}.json')); v = p['views'][r.get('use', 0) if isinstance(r.get('use'), int) else 0]
    wall = {'midpoint': (np.asarray(p['normal']) * p['planeV'] + np.array([-p['normal'][1], p['normal'][0]]) * sum(p['frameU']) / 2).tolist(), 'normal': p['normal'], 'lengthM': p['lengthM'], 'heightM': p['heightM']}
    rend, _, _ = J.render_wall(B, wall, p['groundZ'], p['heightM'], p['lengthM'])
    ph = Image.open(root / 'sides/img' / v['rect']).convert('RGB'); pad = v['rectPad']; ph = ph.crop((pad, pad, ph.width - pad, ph.height - pad))
    rd = Image.fromarray(rend).resize(ph.size)
    s = Image.new('RGB', (ph.width * 2 + 10, ph.height + 24), (20, 20, 22)); s.paste(ph, (0, 24)); s.paste(rd, (ph.width + 10, 24))
    d = ImageDraw.Draw(s); d.text((6, 5), 'photo  ' + i, fill=(230, 230, 230)); d.text((ph.width + 16, 5), 'model, same frame', fill=(230, 230, 230))
    s.save(out / f'{i}.jpg', quality=85)
print('ok')
