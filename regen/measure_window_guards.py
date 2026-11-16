"""Find discrete window guards hidden by an inherited continuous-balcony reading.

Notes only nominate a reinspection. Actual measurements come from independently
queried source crops. Output is external evidence, never an identity-specific rule.
This command does not mutate profiles or generated geometry.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from regen.reading_provenance import load_registered_sides
from regen.specs import bank_cells


def candidates(reading, indices):
    for i in indices:
        railing = reading['balconies'][i]
        lo, hi, base = railing['x0'], railing['x1'], railing['zM']
        top = base + railing.get('railingHeightM', 1.)
        cells = []
        for bank_id, bank in enumerate(reading.get('banks', [])):
            if bank.get('kind', 'window') != 'window':
                continue
            for cell_id, (x0, z0, x1, z1) in enumerate(bank_cells(bank)):
                if lo <= (x0+x1)/2 <= hi and abs(z0-top) <= .45:
                    cells.append({'balcony': i, 'bank': bank_id, 'cell': cell_id,
                                  'rectM': [x0, base-.25, x1, top+.25]})
        # A single already bounded balcony is not evidence of a repeated-span
        # encoding error. Distinct window rows must never be joined implicitly.
        if len(cells) > 1:
            yield from cells


def crop_region(image, rect, view, height):
    x0, z0, x1, z1 = rect
    margin = (x1-x0)*.45
    pad, ppm = view['rectPad'], view['ppm']
    box = [int(pad+(x0-margin)*ppm), int(pad+(height-z1)*ppm),
           int(pad+(x1+margin)*ppm), int(pad+(height-z0)*ppm)]
    if box[0] < pad or box[1] < pad or box[2] > image.width-pad or box[3] > image.height-pad:
        return None
    if min(box[2]-box[0], box[3]-box[1]) < 12:
        return None
    # Enlargement changes presentation only, not the retained pixel resolution.
    crop = image.crop(box)
    scale = min(448/crop.width, 448/crop.height)
    size = (round(crop.width*scale), round(crop.height*scale))
    supplied = Image.new('RGB', (448, 448), '#808080')
    offset = ((448-size[0])//2, (448-size[1])//2)
    supplied.paste(crop.resize(size, Image.Resampling.LANCZOS), offset)
    return supplied, {'sourcePixels': box, 'suppliedSize': size, 'offset': offset,
                      'scaleXY': [size[0]/crop.width, size[1]/crop.height]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--sides', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registered = load_registered_sides(args.sides)
    results = []
    for hypothesis in json.loads(args.proposals.read_text())['rows']:
        if hypothesis['answer']['probabilities'].get('individual_window_guards', 0) < .8:
            continue
        reading = json.loads(Path(hypothesis['reading']).read_text())
        side = registered.get(reading['id'])
        if not side or side.get('status') != 'ok' or reading.get('calibratedCrops'):
            continue
        piece = side['piece']
        for candidate in candidates(reading, hypothesis['recordIndices']):
            for vi, view in enumerate(piece['views']):
                source = side['path'].parent/'img'/view['rect']
                if not source.is_file():
                    continue
                packed = crop_region(Image.open(source).convert('RGB'), candidate['rectM'], view, piece['heightM'])
                if packed is None:
                    continue
                crop, mapping = packed
                buf = io.BytesIO(); crop.save(buf, 'PNG')
                questions = {
                    'visible': {'type': 'noul', 'instructions': 'Is a complete, separate small guard/railing panel at the bottom of a window clearly visible in `photo`, without foliage hiding its outer bounds? A continuous facade molding is not a guard.'},
                    'bounds': {'type': 'box', 'instructions': 'Tight rectangle around the complete separate window guard/railing panel in `photo`, including its own perimeter frame. Exclude the window above, the wall behind, and horizontal building-wide moldings.'},
                    'pattern': {'type': 'choice', 'instructions': 'What is the infill of that separate guard panel in `photo`?', 'criteria': {
                        'vertical_bars': 'Only straight vertical rods/pickets between horizontal rails.',
                        'cross_lattice': 'Visible crossed diagonal rods, diamonds, or X motifs, possibly with vertical separators.',
                        'turned_balusters': 'Thick sculpted or turned vertical balusters with varying width.',
                        'solid': 'A solid opaque panel without holes.',
                        'unclear': 'The guard pattern is not sufficiently visible, or none of these patterns fits.'}}}
                state = {'photo': 'window-guard-source'}
                key = hashlib.sha256(buf.getvalue()+json.dumps(questions, sort_keys=True).encode()).hexdigest()
                crop.save(args.output/(key+'.png'))
                result_path = args.output/(key+'.json')
                answers = json.loads(result_path.read_text()) if result_path.exists() else None
                if answers is None and args.infer:
                    from regen.jev import ask
                    answers = ask(state, questions, tag='window-guard-measure-'+key[:12], retries=1,
                                  assets={'window-guard-source': {'type': 'image', 'format': 'png',
                                          'data_base64': base64.b64encode(buf.getvalue()).decode()}})
                    result_path.write_text(json.dumps(answers, indent=2)+'\n')
                result = {'side': reading['id'], 'reading': hypothesis['reading'], **candidate,
                          'view': vi, 'image': view['image'], 'source': str(source),
                          'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                          **mapping, 'key': key, 'answers': answers, 'applied': False}
                if answers:
                    b = answers['bounds']['box']; ox, oy = mapping['offset']; sx, sy = mapping['scaleXY']; px = mapping['sourcePixels']
                    left, right = px[0]+(448*b['left']-ox)/sx, px[0]+(448*b['right']-ox)/sx
                    top, bottom = px[1]+(448*b['top']-oy)/sy, px[1]+(448*b['bottom']-oy)/sy
                    valid = px[0] <= left < right <= px[2] and px[1] <= top < bottom <= px[3]
                    result['boundsInsideSource'] = valid
                    pad, ppm, H = view['rectPad'], view['ppm'], piece['heightM']
                    result['measuredRectM'] = [(left-pad)/ppm, H-(bottom-pad)/ppm,
                                                (right-pad)/ppm, H-(top-pad)/ppm] if valid else None
                    proof = crop.copy(); ImageDraw.Draw(proof).rectangle([448*b['left'],448*b['top'],448*b['right'],448*b['bottom']],outline='red',width=2)
                    proof.save(args.output/(key+'-proof.png'))
                results.append(result)
    (args.output/'measurements.json').write_text(json.dumps({'status':'proposals requiring cross-view consistency and image review','applied':0,'rows':results}, indent=2)+'\n')
    print(json.dumps({'crops':len(results),'answers':sum(r['answers'] is not None for r in results),'applied':0}))


if __name__ == '__main__':
    main()
