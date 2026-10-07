"""Photograph | model pairs for the ring poses (poses.py), from renders saved by the viewer's capture receiver.
Usage: pairs.py <poses.json> <render prefix> <out dir> [--views 0]   (run from work/)"""
import argparse, json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument('poses'); ap.add_argument('prefix'); ap.add_argument('out'); ap.add_argument('--views', default='0'); a = ap.parse_args()
P = [p for p in json.load(open(a.poses))['poses'] if str(p['view']) in a.views.split(',')]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
made = []
for p in P:
    name = f"{p['side']}__v{p['view']}"; r = Path('areas/captures') / f"{a.prefix}__{name}.png"
    if not r.exists() or not os.path.exists(p['persp']): continue
    ph, md = Image.open(p['persp']).convert('RGB'), Image.open(r).convert('RGB')
    md = md.resize((round(md.width * ph.height / md.height), ph.height), Image.LANCZOS)
    im = Image.new('RGB', (ph.width + md.width + 12, ph.height + 34), (22, 24, 26)); im.paste(ph, (0, 34)); im.paste(md, (ph.width + 12, 34))
    d = ImageDraw.Draw(im); d.text((6, 7), f"{p['building']}  photo {p['captureDate']}", font=font, fill=(235, 235, 235)); d.text((ph.width + 18, 7), 'model', font=font, fill=(235, 235, 235))
    im.save(out / f'{name}.jpg', quality=88); made.append(name)
print(len(made), 'pairs in', out)
