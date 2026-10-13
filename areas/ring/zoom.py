"""A close look at one part of a photograph | model pair: a region cut from the street photograph and a region cut from
the model rendered from that camera, side by side and enlarged, so small parts (a railing, a cornice, a second bay
behind a branch) can be compared at full resolution. The recorded camera can stand a few metres or degrees off, so the
same part may sit at a different place in the model: give the model its own region with --model.
Usage (run from work/): zoom.py <side> <view 0|1> <render prefix> <fx0> <fy0> <fx1> <fy1> <out.jpg> [--model mx0 my0 mx1 my1]
  fx/fy (and mx/my) are fractions of the image (0,0 top left; 1,1 bottom right), e.g. 0 0 0.5 0.5 for the top left
  quarter."""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(usage=__doc__); ap.add_argument('side'); ap.add_argument('view', type=int); ap.add_argument('prefix')
ap.add_argument('box', type=float, nargs=4); ap.add_argument('out'); ap.add_argument('--model', type=float, nargs=4)
a = ap.parse_args(); boxes = [a.box, a.model or a.box]
for b in boxes:
    if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1): ap.error('a region needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1')
pose = next((p for p in json.load(open('areas/ring/poses-all.json'))['poses'] if p['side'] == a.side and p['view'] == a.view), None)
if pose is None: ap.error(f'no pose for {a.side} v{a.view}')
render = Path('areas/captures') / f'{a.prefix}__{a.side}__v{a.view}.png'
if not render.exists(): ap.error(f'no render {render}')
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
cut = []
for path, (x0, y0, x1, y1) in zip((pose['persp'], render), boxes):
    im = Image.open(path).convert('RGB'); W, H = im.size
    c = im.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)))
    s = min(4., 760 / c.width, 900 / c.height); cut.append(c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS))
h = max(c.height for c in cut); can = Image.new('RGB', (cut[0].width + cut[1].width + 12, h + 28), (22, 24, 26))
can.paste(cut[0], (0, 28)); can.paste(cut[1], (cut[0].width + 12, 28)); d = ImageDraw.Draw(can)
d.text((6, 5), f'photo {a.side} v{a.view}  [{boxes[0][0]:.2f},{boxes[0][1]:.2f} - {boxes[0][2]:.2f},{boxes[0][3]:.2f}]', font=font, fill=(235, 235, 235))
d.text((cut[0].width + 18, 5), f'model [{boxes[1][0]:.2f},{boxes[1][1]:.2f} - {boxes[1][2]:.2f},{boxes[1][3]:.2f}]', font=font, fill=(235, 235, 235))
can.save(a.out, quality=90); print(json.dumps({'image': a.out, 'side': a.side, 'view': a.view}))
