"""Source photographs beside real viewer captures, retaining each aspect ratio.

Capture files are 00.png, 01.png, ... in job order. This only assembles evidence;
it does not grade resemblance or replace the actual navigable world.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, required=True)
    ap.add_argument('--capture', type=Path, action='append', required=True)
    ap.add_argument('--label', action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--cell-width', type=int, default=640)
    args = ap.parse_args()
    if len(args.capture) != len(args.label) or args.rows < 1 or args.cell_width < 100:
        ap.error('each capture directory needs a label; dimensions must be positive')
    work = Path(__file__).resolve().parents[1]
    jobs = json.loads(args.jobs.read_text())
    if isinstance(jobs, dict): jobs = jobs['poses']
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    width = args.cell_width * (1 + len(args.capture))
    audit = []
    for start in range(0, len(jobs), args.rows):
        rows = []
        for i in range(start, min(start + args.rows, len(jobs))):
            job = jobs[i]
            source = Path(job['persp'])
            if not source.is_absolute(): source = work / source
            paths = [source, *[root / f'{i:02d}.png' for root in args.capture]]
            images = [Image.open(p).convert('RGB') for p in paths]
            expected = (job['width'], job['height'])
            if any(im.size != expected for im in images):
                raise ValueError(f'Job {i}: camera aspect mismatch, expected {expected}, got {[im.size for im in images]}')
            h = min(540, round(args.cell_width * expected[1] / expected[0]))
            row = Image.new('RGB', (width, h + 34), '#eeeeee')
            draw = ImageDraw.Draw(row)
            for col, (im, label) in enumerate(zip(images, ['Source', *args.label])):
                draw.text((col*args.cell_width + 7, 8), f'{i:02d} {label} | {job["side"]}' if col == 0 else label, fill='#222222', font=font)
                im.thumbnail((args.cell_width - 8, h), Image.Resampling.LANCZOS)
                row.paste(im, (col*args.cell_width + (args.cell_width-im.width)//2, 34 + (h-im.height)//2))
            rows.append(row)
            audit.append({'job': i, 'side': job['side'], 'view': job['view'],
                          'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]})
        sheet = Image.new('RGB', (width, sum(r.height for r in rows)), '#eeeeee')
        y = 0
        for row in rows: sheet.paste(row, (0, y)); y += row.height
        sheet.save(args.output / f'sheet-{start//args.rows+1}.jpg', quality=92)
    (args.output / 'sheets-manifest.json').write_text(json.dumps({'jobs': str(args.jobs), 'labels': args.label,
                      'visualAcceptance': 'not_assessed_by_sheet_generator', 'captures': audit}, indent=2))
    print(f'{len(jobs)} source/capture pairs in {args.output}')


if __name__ == '__main__': main()
