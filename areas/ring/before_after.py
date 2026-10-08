"""Photograph | before | after, per ring building, from the viewer renders saved for two builds at the poses of
poses.py, plus contact sheets of them all.
Usage: before_after.py <poses.json> <before prefix> <after prefix> <out dir> [--views 0]   (run from work/)"""
import argparse, json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument('poses'); ap.add_argument('before'); ap.add_argument('after'); ap.add_argument('out'); ap.add_argument('--views', default='0'); a = ap.parse_args()
P = [p for p in json.load(open(a.poses))['poses'] if str(p['view']) in a.views.split(',')]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 22); small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
H = 640; made = []
for p in P:
    name = f"{p['side']}__v{p['view']}"; b = Path('areas/captures') / f"{a.before}__{name}.png"; f = Path('areas/captures') / f"{a.after}__{name}.png"
    if not (b.exists() and f.exists() and os.path.exists(p['persp'])): continue
    tiles = [Image.open(x).convert('RGB') for x in (p['persp'], b, f)]
    tiles = [t.resize((round(t.width * H / t.height), H), Image.LANCZOS) for t in tiles]
    W = sum(t.width for t in tiles) + 24; im = Image.new('RGB', (W, H + 40), (20, 22, 24)); d = ImageDraw.Draw(im); x = 0
    for t, label in zip(tiles, ('photograph', 'before (v7)', 'after (ring rebuild)')):
        im.paste(t, (x, 40)); d.text((x + 8, 9), label, font=font, fill=(236, 236, 236)); x += t.width + 12
    d.text((W - 360, 12), p['building'], font=small, fill=(150, 154, 158))
    im.save(out / f"{p['building']}.jpg", quality=86); made.append(out / f"{p['building']}.jpg")
# contact sheets: four buildings a sheet, stacked
for k in range(0, len(made), 4):
    ims = [Image.open(m) for m in made[k:k + 4]]; W = max(i.width for i in ims)
    sheet = Image.new('RGB', (W, sum(i.height for i in ims) + 8 * (len(ims) - 1)), (10, 10, 10)); y = 0
    for i in ims: sheet.paste(i, (0, y)); y += i.height + 8
    s = 1600 / W; sheet = sheet.resize((1600, int(sheet.height * s)), Image.LANCZOS); sheet.save(out / f'sheet-{k // 4 + 1:02}.jpg', quality=84)
print(json.dumps({'buildings': len(made), 'sheets': (len(made) + 3) // 4, 'out': str(out)}))
