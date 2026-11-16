"""Blind, bounded Jev verification of image-line surface semantics only.

Prepare crops first, record independent human labels, then run the API batch.
No roof class or expected answer is sent. No metric roof observations are made.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np
from PIL import Image, ImageDraw


CRITERIA = {
    'pitched_roof': 'Sloping roof covering such as shingles, slate or tiles. Exclude vertical triangular gable walls, cornice trim and balcony floors.',
    'flat_deck': 'A horizontal flat roof or deck top surface. Exclude vertical trim bands and balcony/railing assemblies.',
    'wall': 'Vertical facade or gable wall/cladding, or architectural cornice/trim. Windows may be present. Not sloping roof covering.',
    'sky': 'Open sky or clouds with no architectural surface.',
    'railing_balcony': 'Railing, balcony edge/slab, guardrail or terrace balcony assembly rather than main roof covering.',
    'foliage': 'Tree leaves, branches or plants concealing the architectural surface.',
    'unclear': 'The surface is cropped, hidden, mixed along this segment, or not recognizable reliably.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(records, maximum=24):
    """One visible long line per source first; then diverse second lines.

    Enumerates every job, including explicit abstentions. Ranking uses only
    line support/visibility and geometric diversity, never IDs or roof labels.
    """
    selected, second, coverage = [], [], []
    for record in records:
        choices = []
        for index, candidate in enumerate(record.get('junctionCandidates', [])):
            points = np.asarray(candidate['pointsPx'], float)
            span = float(np.linalg.norm(points[1] - points[0]))
            width = max(1., np.ptp(record['projection']['xRangePx']))
            quality = span / width * (1 - candidate.get('occluderProbability', 0)) * candidate.get('structureBelowProbability', 0)
            choices.append({'row': record['row'], 'candidateIndex': index, 'quality': quality,
                            'candidate': candidate, 'source': record})
        choices.sort(key=lambda c: (-c['quality'], c['candidateIndex']))
        coverage.append({'row': record['row'], 'candidatesAvailable': len(choices),
                         'reason': None if choices else 'no junction candidate survived the shared source algorithm'})
        if not choices:
            continue
        first = choices[0]; selected.append(first)
        if len(choices) > 1:
            center = np.mean(first['candidate']['pointsPx'], axis=0)
            for other in choices[1:]:
                separation = np.linalg.norm(center - np.mean(other['candidate']['pointsPx'], axis=0)) / record['sourceSizePx'][1]
                other['diversityQuality'] = other['quality'] * min(1., 4 * separation)
            second.append(max(choices[1:], key=lambda c: (c['diversityQuality'], -c['candidateIndex'])))
    second.sort(key=lambda c: (-c['diversityQuality'], c['row']))
    return selected[:maximum] + second[:max(0, maximum-len(selected))], coverage


def prepare(candidates, output):
    data = json.loads(candidates.read_text())
    chosen, coverage = select(data['records'])
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(chosen):
        source, candidate = item['source'], item['candidate']
        if sha(source['sourcePath']) != source['sourceSha256']:
            raise ValueError('source image changed after candidate extraction')
        image = Image.open(source['sourcePath']).convert('RGB')
        line = np.asarray(candidate['pointsPx'], float)
        midpoint = line.mean(0); span = np.linalg.norm(line[1]-line[0])
        width, height = min(image.width, max(400, min(720, int(2.2*span)))), min(image.height, max(320, min(540, int(1.5*span))))
        left = int(np.clip(midpoint[0]-width/2, 0, image.width-width)); top = int(np.clip(midpoint[1]-height/2, 0, image.height-height))
        crop = image.crop((left, top, left+width, top+height))
        scale = min(1., 640/max(crop.size))
        crop = crop.resize((int(crop.width*scale), int(crop.height*scale)), Image.Resampling.LANCZOS)
        if min(crop.size)<280:
            raise ValueError('source crop too small for current image API')
        target = (line-np.array([left,top]))*scale
        clean = output/f'{index:02d}-clean.jpg'; crop.save(clean, quality=92)
        marked = crop.copy(); draw = ImageDraw.Draw(marked)
        a,b=target; direction=b-a; length=np.linalg.norm(direction); normal=np.array([-direction[1],direction[0]])/max(length,1e-8)
        # Central70% remains untouched, so classification pixels are available.
        for point in (a,b):
            draw.line([tuple(point-normal*6),tuple(point+normal*6)], fill=(255,0,170), width=2)
        for t in np.arange(0, length, 12):
            u=t/max(length,1)
            if .15<u<.85: continue
            draw.line([tuple(a+direction*u),tuple(a+direction*min(1,(t+3)/length))],fill=(0,230,255),width=1)
        path=output/f'{index:02d}-marked.jpg';marked.save(path,quality=92)
        records.append({'id':f'c{index:02d}','row':item['row'],'candidateIndex':item['candidateIndex'],
            'sourcePath':source['sourcePath'],'sourceSha256':source['sourceSha256'],
            'sourceLinePx':candidate['pointsPx'],'cropBoundsPx':[left,top,left+width,top+height],
            'cropSizePx':list(crop.size),'targetSegmentPx':target.tolist(),
            'markedPath':str(path),'markedSha256':sha(path),'cleanPath':str(clean),
            'selectionQuality':item['quality'],'selectionUsesRoofClass':False,
            'availableSourceAbovePx':float(midpoint[1]-top),'availableSourceBelowPx':float(top+height-midpoint[1])})
    manifest={'status':'prepared-before-API-inference','inputCandidatesPath':str(candidates),'inputCandidatesSha256':sha(candidates),
              'selectionRule':'one longest visible spatially supported candidate per available source, then best spatially separated second candidates; no IDs/manual classes',
              'all22Coverage':coverage,'records':records,'maximumCandidates':24,'classes':CRITERIA}
    (output/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for page in range((len(records)+7)//8):
        sheet=Image.new('RGB',(1440,680),'white');draw=ImageDraw.Draw(sheet)
        for j,record in enumerate(records[page*8:page*8+8]):
            im=Image.open(record['markedPath']);im.thumbnail((355,300));x=j%4*360;y=j//4*340
            sheet.paste(im,(x,y+30));draw.text((x+5,y+5),f'{record["id"]} row{record["row"]:02d} line{record["candidateIndex"]}',fill='black')
        sheet.save(output/f'blind-sheet-{page+1}.jpg')
    print(json.dumps({'selected':len(records),'sources':len({r['row'] for r in records}),'jobsConsidered':len(coverage)}))


def run(output, batch_size=4):
    from regen import jev
    manifest=json.loads((output/'selection.json').read_text())
    blind_path=output/'blind-labels.json'
    if not blind_path.exists(): raise ValueError('Record independent blind labels before API inference')
    blind=json.loads(blind_path.read_text())
    if set(blind['labels']) != {r['id'] for r in manifest['records']}: raise ValueError('blind labels must cover every selected crop')
    records=manifest['records'];batches=[];results=[]
    for offset in range(0,len(records),batch_size):
        chunk=records[offset:offset+batch_size];assets={};state={'instruction':'Classify only visible material. A proposed image line may lie inside one surface and is not assumed to be a boundary. No roof/eave answer is expected.','crops':{}}
        questions={}
        for record in chunk:
            ref=str(uuid.uuid4());path=Path(record['markedPath'])
            if sha(path)!=record['markedSha256']: raise ValueError('marked crop changed after blind labeling')
            assets[ref]={'type':'image','format':'jpg','data_base64':base64.b64encode(path.read_bytes()).decode()}
            state['crops'][record['id']]={'image':ref,'targetSegmentPx':record['targetSegmentPx'],
                'marking':'The segment connects the two magenta endpoint ticks. Only end portions have thin cyan dashes; the central70% is deliberately unpainted.'}
            for side in ('above','below'):
                questions[f'{record["id"]}_{side}']={'type':'choice','instructions':
                    f'In `crops.{record["id"]}.image`, what visible surface lies immediately {side.upper()} the target segment between the magenta endpoint ticks? Judge the central60% of that segment and the nearby strip roughly5–20pixels {side}, not the colored marks. Use the surrounding photograph as context. Do not assume this is a roof boundary; if mixed, hidden or cropped choose unclear.',
                    'criteria':CRITERIA}
        tag=f'roof-line-semantics-v1-batch-{offset//batch_size}'
        request_summary={'tag':tag,'model':'jev-latest','cropIds':[r['id'] for r in chunk],
            'blindLabelsSha256':sha(blind_path),'questions':questions,'imagePaths':[r['markedPath'] for r in chunk]}
        (output/f'request-{offset//batch_size:02d}.json').write_text(json.dumps(request_summary,indent=2)+'\n')
        started=perf_counter();answers=jev.ask(state,questions,tag=tag,assets=assets);elapsed=perf_counter()-started
        usage=[json.loads(line) for line in jev.LOG.read_text().splitlines() if json.loads(line).get('tag')==tag]
        batch={'tag':tag,'elapsedSeconds':elapsed,'answers':answers,'usageLogRecords':usage,
               'modelAlias':'jev-latest','resolvedModelNotExposedByExistingClient':True}
        batches.append(batch);(output/f'response-{offset//batch_size:02d}.json').write_text(json.dumps(batch,indent=2)+'\n')
        for record in chunk:
            result={**record,'above':answers[f'{record["id"]}_above'],'below':answers[f'{record["id"]}_below'],
                'blindLabel':blind['labels'][record['id']],'metricDepthDerived':False,'activeRoofConstraintPromoted':False}
            results.append(result)
        print(json.dumps({'batch':offset//batch_size,'crops':len(chunk),'seconds':round(elapsed,2),'inputTokens':sum(u.get('input',0) for u in usage)}),flush=True)
        (output/'results.json').write_text(json.dumps({'selection':manifest,'blindLabelsSha256':sha(blind_path),
            'batches':batches,'records':results,'scope':'above/below surface semantics only'},indent=2)+'\n')


def audit(output):
    """Compare with the unchanged preregistered labels; make no geometry."""
    data = json.loads((output/'results.json').read_text())
    if sha(output/'blind-labels.json') != data['blindLabelsSha256']:
        raise ValueError('blind labels changed after inference')
    records = data['records']
    assessments = []
    confusion = {side: Counter() for side in ('above', 'below')}
    for record in records:
        label = record['blindLabel']
        matches = {side: record[side]['choice'] == label[side] for side in ('above','below')}
        if not label['ambiguous']:
            for side in matches:
                confusion[side][(label[side], record[side]['choice'])] += 1
        above = record['above']['probabilities']
        below = record['below']['probabilities']
        # Illustrative operating point, not a fitted or production threshold.
        pair_gate = above.get('pitched_roof',0) + above.get('flat_deck',0) >= .8 and below.get('wall',0) >= .8
        assessments.append({'id':record['id'],'row':record['row'], 'ambiguousBlindLabel':label['ambiguous'],
            'matchesBlind':matches, 'bothMatchBlind':all(matches.values()),
            'illustrativeRoofOverWallGate':pair_gate,
            'blindRoofOverWall':label['above'] in ('pitched_roof','flat_deck') and label['below']=='wall',
            'aboveRoofProbability':above.get('pitched_roof',0)+above.get('flat_deck',0),
            'belowWallProbability':below.get('wall',0), 'metricEvidenceAdmitted':False})
    clear = [a for a in assessments if not a['ambiguousBlindLabel']]
    usage = [u for batch in data['batches'] for u in batch['usageLogRecords']]
    summary = {'status':'completed-semantic-experiment-not-production-admission',
        'approval':'User explicitly approved exactly these24 prepared public street-photo crops: Yes, send the24crops.',
        'selectionSha256':sha(output/'selection.json'), 'blindLabelsSha256':data['blindLabelsSha256'],
        'sourceJobsConsidered':len(data['selection']['all22Coverage']), 'sourceJobsSelected':len({r['row'] for r in records}),
        'crops':len(records),'independentQuestions':2*len(records), 'unambiguousCrops':len(clear),
        'ambiguousCrops':len(records)-len(clear), 'unambiguousQuestionMatches':sum(sum(a['matchesBlind'].values()) for a in clear),
        'unambiguousQuestionCount':2*len(clear), 'unambiguousPairMatches':sum(a['bothMatchBlind'] for a in clear),
        'allQuestionMatchesIncludingAmbiguous':sum(sum(a['matchesBlind'].values()) for a in assessments),
        'unambiguousConfusion':{s:[{'blind':k[0],'prediction':k[1],'count':v} for k,v in sorted(c.items())] for s,c in confusion.items()},
        'illustrativeGate':{'aboveRoofProbabilityAtLeast':.8,'belowWallProbabilityAtLeast':.8,
            'thresholdStatus':'post-inference illustrative only; no threshold fit or admission',
            'unambiguousTruePositiveIds':[a['id'] for a in clear if a['illustrativeRoofOverWallGate'] and a['blindRoofOverWall']],
            'unambiguousFalsePositiveIds':[a['id'] for a in clear if a['illustrativeRoofOverWallGate'] and not a['blindRoofOverWall']],
            'unambiguousFalseNegativeIds':[a['id'] for a in clear if not a['illustrativeRoofOverWallGate'] and a['blindRoofOverWall']],
            'ambiguousSelectedIds':[a['id'] for a in assessments if a['illustrativeRoofOverWallGate'] and a['ambiguousBlindLabel']]},
        'usage':{'successfulCalls':len(data['batches']),'inputTokens':sum(u['input'] for u in usage),
            'outputTokens':sum(u['output'] for u in usage),
            'totalMeasuredRequestSeconds':sum(b['elapsedSeconds'] for b in data['batches']),
            'requestSeconds':[b['elapsedSeconds'] for b in data['batches']],
            'modelAlias':'jev-latest','resolvedModelNotExposedByExistingClient':True},
        'interpretation':'Useful semantic screening, insufficient for automatic roof-edge admission: narrow trim/roof strips and balcony soffits remain confused. Human labels are one prior visual review, not independent surveyed truth. Six mixed/occluded crops are reported separately. No depth, eave height, pitch or active roof observation was derived.',
        'metricDepthDerived':False,'activeRoofObservationsChanged':False,'assessments':assessments}
    (output/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    short = {'pitched_roof':'roof','flat_deck':'deck','railing_balcony':'balcony','foliage':'foliage','wall':'wall','sky':'sky','unclear':'unclear'}
    for page in range((len(records)+7)//8):
        sheet = Image.new('RGB',(1440,880),'white'); draw=ImageDraw.Draw(sheet)
        for index,record in enumerate(records[page*8:page*8+8]):
            x=index%4*360; y=index//4*440; label=record['blindLabel']
            assess=assessments[page*8+index]
            color='#996600' if label['ambiguous'] else '#18732b' if assess['bothMatchBlind'] else '#bb2525'
            draw.text((x+5,y+5),f'{record["id"]} row{record["row"]:02d} line{record["candidateIndex"]}',fill=color)
            im=Image.open(record['markedPath']); im.thumbnail((350,290));sheet.paste(im,(x+5,y+28))
            for j,side in enumerate(('above','below')):
                prediction=record[side]
                draw.text((x+5,y+325+16*j),f'{side}: {short[prediction["choice"]]}  conf {prediction["confidence"]:.2f}',fill='black')
            draw.text((x+5,y+359),f'blind: {short[label["above"]]} / {short[label["below"]]}',fill=color)
            draw.text((x+5,y+376),'blind ambiguous' if label['ambiguous'] else 'blind agreement' if assess['bothMatchBlind'] else 'blind disagreement',fill=color)
            draw.text((x+5,y+393),'semantic check only; no metric evidence',fill='#555555')
        sheet.save(output/f'predictions-sheet-{page+1}.jpg',quality=92)
    blocked=output/'blocked-audit.json'
    if blocked.exists():
        history=json.loads(blocked.read_text())
        history['supersededBy']='audit.json'
        history['historicalStatusOnly']=True
        history['subsequentExplicitApproval']='Yes, send the24crops'
        blocked.write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('crops','unambiguousQuestionMatches','unambiguousQuestionCount','unambiguousPairMatches','usage','illustrativeGate')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidates',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit: audit(a.output)
    elif a.run: run(a.output)
    else:
        if not a.candidates:p.error('--candidates required for preparation')
        prepare(a.candidates,a.output)


if __name__=='__main__':main()
