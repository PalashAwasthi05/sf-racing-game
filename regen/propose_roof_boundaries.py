"""Offline source-only roof/wall/sky contour proposals for every review job.

Cached ADE20K SegFormer supplies sky/building/occlusion masks. Cached Grounding
DINO supplies explicit roof/eave/cornice semantic boxes; ADE's building class
alone never becomes a roof or eave label. OpenCV line segments refine actual
pixel edges. Outputs are review candidates, never metric roof constraints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OfflineModels:
    def __init__(self, root, device='auto'):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = torch.device('mps' if device == 'auto' and torch.backends.mps.is_available() else ('cpu' if device == 'auto' else device))
        segpath = sorted((root / '.models/models--nvidia--segformer-b2-finetuned-ade-512-512/snapshots').glob('*'))[-1]
        detpath = sorted((root / '.models/models--IDEA-Research--grounding-dino-tiny/snapshots').glob('*'))[-1]
        self.segprocessor = AutoImageProcessor.from_pretrained(segpath, local_files_only=True)
        self.seg = AutoModelForSemanticSegmentation.from_pretrained(segpath, local_files_only=True, use_safetensors=False).to(self.device).eval()
        self.detprocessor = AutoProcessor.from_pretrained(detpath, local_files_only=True)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detpath, local_files_only=True).to(self.device).eval()
        labels = self.seg.config.id2label
        self.codes = {name: int(i) for i, name in labels.items()}
        self.identities = {'segmentationConfigPath': str(segpath / 'config.json'), 'segmentationConfigSha256': sha(segpath / 'config.json'),
                           'detectorConfigPath': str(detpath / 'config.json'), 'detectorConfigSha256': sha(detpath / 'config.json'),
                           'device': str(self.device), 'networkUsed': False,
                           'roofLabelInADE20K': False,
                           'semanticPrompt': 'a roof. a sloped roof. an eave. a cornice. a building facade.'}

    def infer(self, image):
        torch = self.torch
        im = Image.fromarray(image)
        inputs = self.segprocessor(images=im, return_tensors='pt').to(self.device)
        with torch.no_grad():
            logits = self.seg(**inputs).logits
            probabilities = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode='bilinear', align_corners=False).softmax(1)[0]
            groups = {}
            for group, names in {'sky': ['sky'], 'structure': ['building', 'wall', 'house', 'skyscraper'],
                                 'opening': ['windowpane', 'door', 'screen door'],
                                 'occluder': ['tree', 'palm', 'plant', 'pole', 'streetlight', 'person', 'car', 'fence', 'traffic light']}.items():
                indices = [self.codes[n] for n in names if n in self.codes]
                groups[group] = probabilities[indices].sum(0).float().cpu().numpy()
        del inputs, logits, probabilities
        # Whole source keeps context; detector boxes remain coarse semantic
        # supports. They are not silently treated as measured edge positions.
        small = im.copy(); small.thumbnail((768, 768))
        prompt = self.identities['semanticPrompt']
        inputs = self.detprocessor(images=small, text=prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.det(**inputs)
        result = self.detprocessor.post_process_grounded_object_detection(output, inputs.input_ids,
            threshold=.20, text_threshold=.16, target_sizes=[image.shape[:2]])[0]
        boxes = [{'boxPx': b.tolist(), 'score': float(s), 'text': str(t)}
                 for b, s, t in zip(result['boxes'].cpu().numpy(), result['scores'].cpu().numpy(), result.get('text_labels', result.get('labels')))]
        del inputs, output, result
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        return groups, boxes


def project_target(job, piece, size):
    camera, at = np.asarray(job['from'], float), np.asarray(job['at'], float)
    forward = at - camera; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    focal = size[1] / 2 / math.tan(math.radians(float(job['vfovDeg'])) / 2)
    normal = np.asarray(piece['normal'], float); normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    points = np.array([[*(normal * float(piece['planeV']) + tangent * float(u)), float(piece['groundZ']) + z]
                       for u in piece['frameU'] for z in (0., float(piece['heightM']))])
    directions = points - camera
    depth = directions @ forward
    pixels = np.c_[directions @ right, directions @ down] / depth[:, None] * focal + np.asarray(size) / 2 - .5
    if not np.isfinite(pixels).all() or np.any(depth <= 0):
        raise ValueError('registered facade frame is behind camera or malformed')
    return {'facadeCornersPx': pixels.tolist(), 'xRangePx': [float(max(0, pixels[:, 0].min())), float(min(size[0] - 1, pixels[:, 0].max()))],
            'cameraXYZ': camera.tolist(), 'basisRightDownForward': np.array([right, down, forward]).tolist(),
            'focalPx': float(focal), 'cameraXYValidated': False}


def _runs(points, gap=5):
    runs, current = [], []
    for point in points:
        if current and point[0] - current[-1][0] > gap:
            if len(current) >= 12:
                runs.append(current)
            current = []
        current.append(point)
    if len(current) >= 12:
        runs.append(current)
    return runs


def skyline_candidates(image, groups, x_range, boxes):
    h, w = image.shape[:2]
    sky, building, occluder = (groups[k] for k in ('sky', 'structure', 'occluder'))
    points, hidden, unclear = [], [], []
    low, high = map(int, x_range)
    for x in range(max(2, low), min(w - 2, high) + 1, 2):
        found = None
        for y in np.flatnonzero((building[:, x] > .55) & (sky[:, x] < .25)):
            if y < 5 or y > h - 8:
                continue
            above = slice(max(0, y - 8), y - 1)
            below = slice(y + 1, min(h, y + 9))
            if np.mean(sky[above, x]) > .65 and np.mean(building[below, x]) > .6:
                if np.max(occluder[max(0, y-5):y+6, x]) > .35:
                    hidden.append(x)
                    continue
                found = [x, int(y)]
                break
        if found is not None:
            points.append(found)
        elif x not in hidden:
            if np.max(occluder[:max(1, int(h*.6)), x]) > .5:
                hidden.append(x)
            else:
                unclear.append(x)
    candidates = []
    quantization = max(2., max(h, w) / 128.)  # logits are coarser than source pixels
    for run in _runs(points):
        pts = np.asarray(run, float)
        if np.ptp(pts[:, 0]) < max(28, .04 * (high-low)):
            continue
        # Preserve curves/dormers. This is a pixel contour, not a fitted ridge.
        contour = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), quantization / 2, False).reshape(-1, 2)
        roof_overlap = any('roof' in b['text'] and b['score'] >= .25 and b['boxPx'][0] <= pts[:,0].mean() <= b['boxPx'][2]
                           and b['boxPx'][1] - quantization <= pts[:,1].mean() <= b['boxPx'][3] + quantization for b in boxes)
        candidates.append({'role': 'roof-sky-silhouette-candidate' if roof_overlap else 'building-sky-silhouette',
            'pointsPx': contour.tolist(), 'uncertaintyPx': quantization,
            'semanticConfidence': float(np.mean([min(sky[max(0,int(y)-5),int(x)], building[min(h-1,int(y)+5),int(x)]) for x,y in pts])),
            'visibility': 'unoccluded-semantic-sky/structure-transition',
            'use': 'conditional upper-silhouette check only; does not establish eave, ridge height or depth',
            'metricCoordinatesDerived': False, 'automaticHardConstraint': False})
    return candidates, {'occludedColumns': len(set(hidden)), 'unclearColumns': len(set(unclear)),
                        'candidateColumns': len(points), 'targetColumnsSampled': max(0, (high-low)//2)}


def junction_candidates(image, groups, x_range, boxes):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    roofs = [b for b in boxes if 'roof' in b['text'] and b['score'] >= .25]
    trim = [b for b in boxes if any(t in b['text'] for t in ('cornice', 'eave')) and b['score'] >= .25]
    h, w = gray.shape
    candidates, rejected = [], Counter()
    if detected is None:
        return [], {'no-line-segments': 1}
    x0, x1 = x_range
    for raw in np.asarray(detected).reshape(-1, 4):
        a, b = np.array(raw[:2]), np.array(raw[2:])
        if a[0] > b[0]:
            a, b = b, a
        span = b[0] - a[0]
        if span < max(35., .15 * (x1-x0)) or abs(b[1]-a[1]) > .35 * span:
            continue
        if b[0] < x0 or a[0] > x1 or (a[1]+b[1])/2 > h*.68:
            continue
        xs = np.linspace(max(x0,a[0]), min(x1,b[0]), max(12,int(span/5)))
        ys = np.interp(xs, [a[0],b[0]], [a[1],b[1]])
        iy, ix = np.clip(ys.astype(int),0,h-1), np.clip(xs.astype(int),0,w-1)
        occlusion = float(np.mean(groups['occluder'][iy,ix]))
        if occlusion > .35:
            rejected['foliage-or-occluder-on-line'] += 1; continue
        above_y, below_y = np.clip(iy-10,0,h-1),np.clip(iy+12,0,h-1)
        structure_below = float(groups['structure'][below_y,ix].mean())
        if structure_below < .5:
            rejected['no-structural-wall-below'] += 1; continue
        roof_votes = [r for r in roofs if r['boxPx'][0] <= xs.mean() <= r['boxPx'][2]
                      and r['boxPx'][1] <= ys.mean() <= r['boxPx'][3] + .18 * (r['boxPx'][3]-r['boxPx'][1])
                      and ys.mean() >= r['boxPx'][1] + .35 * (r['boxPx'][3]-r['boxPx'][1])]
        trim_votes = [r for r in trim if r['boxPx'][0] <= xs.mean() <= r['boxPx'][2]
                      and r['boxPx'][1]-8 <= ys.mean() <= r['boxPx'][3]+8]
        if not roof_votes:
            rejected['no-explicit-roof-semantic-support'] += 1; continue
        # Coarse zero-shot boxes sometimes call a whole glazed facade a roof.
        # A box full of repeated openings cannot establish roof/wall identity.
        supported_roofs=[]
        for roof in roof_votes:
            rx0,ry0,rx1,ry1=roof['boxPx']
            left,right=max(0,int(rx0)),min(w,int(rx1)+1)
            top,bottom=max(0,int(ry0)),min(h,int(ry1)+1)
            share=float(groups['opening'][top:bottom,left:right].mean()) if right>left and bottom>top else 1.
            if share>.12 or ((ry1-ry0)>.55*h and share>.03):
                continue
            supported_roofs.append({**roof,'openingShare':share})
        if not supported_roofs:
            rejected['roof-box-is-ambiguous-facade-with-openings'] += 1; continue
        roof_votes=supported_roofs
        roof_score = max(r['score'] for r in roof_votes)
        opening_above = float(groups['opening'][above_y,ix].mean())
        sky_above = float(groups['sky'][above_y,ix].mean())
        role = 'roof-wall-junction-candidate'
        ambiguity = ['Roof detector is a coarse box; cladding/trim identity still requires source review.']
        if sky_above>.55:
            role='roof-sky-silhouette-line'
            ambiguity.append('Sky above line makes this an upper silhouette, not a wall/roof junction or a measured host ridge.')
        elif opening_above > .08 or max(r['openingShare'] for r in roof_votes)>.035:
            role = 'gable-wall-or-dormer-junction-unresolved'
            ambiguity.append('Openings above line may belong to a gable facade or dormer; not an eave measurement.')
        elif trim_votes:
            role = 'roof-cornice-junction-candidate'
            ambiguity.append('Visible cornice edge may conceal structural wall/roof junction.')
        candidates.append({'role':role,'pointsPx':[[float(xs[0]),float(ys[0])],[float(xs[-1]),float(ys[-1])]],
            'uncertaintyPx': max(2.,max(w,h)/256), 'roofSemanticScore':roof_score,
            'structureBelowProbability':structure_below,'occluderProbability':occlusion,
            'openingAboveProbability':opening_above,'explicitTrimSemanticSupport':bool(trim_votes),
            'skyAboveProbability':sky_above,
            'roofBoxOpeningShare':max(r['openingShare'] for r in roof_votes),
            'confidence':'candidate-medium' if trim_votes and opening_above<=.08 else 'candidate-low',
            'ambiguity':ambiguity,'automaticHardConstraint':False,'metricCoordinatesDerived':False,
            'score':float(span/max(1,x1-x0)*roof_score*(1-occlusion))})
    # Parallel trim edges are uncertainty alternatives, not separate eaves.
    candidates.sort(key=lambda c:c['score'],reverse=True)
    selected=[]
    for candidate in candidates:
        mid=np.mean(candidate['pointsPx'],axis=0)
        if any(np.linalg.norm(mid-np.mean(other['pointsPx'],axis=0))<12 for other in selected):
            continue
        selected.append(candidate)
        if len(selected)>=8: break
    # A roof cornice has several parallel trim edges. Keep a visible band as
    # an uncertainty bracket instead of promoting each trim edge to an eave.
    grouped=[]
    for candidate in selected:
        a,b=np.asarray(candidate['pointsPx']);match=None
        for group in grouped:
            if ('sky-silhouette' in candidate['role'])!=('sky-silhouette' in group['role']):
                continue
            c,d=np.asarray(group['pointsPx'])
            overlap=max(0,min(b[0],d[0])-max(a[0],c[0]))
            if overlap>=.6*min(b[0]-a[0],d[0]-c[0]) and abs(np.mean([a[1],b[1]])-np.mean([c[1],d[1]]))<max(12,.025*h):
                match=group;break
        if match is None:
            grouped.append(candidate)
        else:
            match.setdefault('parallelAlternativeSegmentsPx',[match['pointsPx']]).append(candidate['pointsPx'])
            match['role']='visible-roof-trim-band-unresolved-eave'
            match['uncertaintyPx']=max(match['uncertaintyPx'],abs(np.mean(np.array(candidate['pointsPx'])[:,1])-np.mean(np.array(match['pointsPx'])[:,1]))+2)
            match['ambiguity'].append('Parallel trim alternatives bracket the visible band; the concealed eave is not located.')
    return grouped,dict(rejected)


def overlay(image, record):
    out=Image.fromarray(image).convert('RGB');d=ImageDraw.Draw(out)
    for candidate in record.get('skylineCandidates',[]):
        d.line([tuple(p) for p in candidate['pointsPx']],fill=(0,225,255),width=3)
    for candidate in record.get('junctionCandidates',[]):
        color=(255,190,0) if 'unresolved' not in candidate['role'] else (255,0,180)
        d.line([tuple(p) for p in candidate['pointsPx']],fill=color,width=3)
    for box in record.get('semanticBoxes',[]):
        if 'roof' in box['text']:
            d.rectangle(box['boxPx'],outline=(80,255,80),width=2)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--reuse',action='store_true',help='Reprocess verified cached semantic probabilities/boxes without loading models.')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    jobs=json.loads(args.jobs.read_text());metadata=json.loads(args.metadata.read_text());pieces=metadata.get('pieces',[]) if isinstance(metadata,dict) else metadata
    args.output.mkdir(parents=True,exist_ok=True)
    cached=json.loads((args.output/'candidates.json').read_text()) if args.reuse else None
    cache_records={(r.get('side'),r.get('image')):r for r in (cached or {}).get('records',[])}
    models=None if args.reuse else OfflineModels(root,args.device)
    model_identities=cached['models'] if cached else models.identities
    results=[]
    for row,job in enumerate(jobs):
        record={'row':row,'side':job.get('side'),'image':job.get('image'),'view':job.get('view'),
                'status':'unresolved','metricCoordinatesDerived':False,'cameraXYValidated':False}
        try:
            matched=[p for p in pieces if p.get('id')==job.get('side')]
            if len(matched)!=1: raise ValueError('missing or ambiguous source side metadata')
            piece=matched[0];view=piece['views'][job['view']]
            if view.get('image')!=job.get('image'): raise ValueError('source view/image identity mismatch')
            path=Path(job['persp'])
            if path.name!=Path(view['persp']).name: raise ValueError('source perspective filename mismatch')
            image=np.asarray(Image.open(path).convert('RGB'))
            if [image.shape[1],image.shape[0]]!=[job['width'],job['height']]: raise ValueError('job/source crop dimensions differ')
            record.update(sourcePath=str(path),sourceSha256=sha(path),sourceSizePx=[image.shape[1],image.shape[0]],
                metadataPath=str(args.metadata),metadataSha256=sha(args.metadata),calibrationPath=str(args.calibration),
                calibrationSha256=sha(args.calibration),captureDate=view.get('captureDate'),registration=view.get('registered'))
            projection=project_target(job,piece,(image.shape[1],image.shape[0]));record['projection']=projection
            if args.reuse:
                previous=cache_records.get((job['side'],job['image']))
                if previous is None or previous.get('sourceSha256')!=record['sourceSha256']:
                    raise ValueError('cached source identity/hash mismatch')
                with np.load(args.output/f'{row:02d}-semantic-probabilities.npz') as archive:
                    groups={k:archive[k].astype(np.float32) for k in ('sky','structure','opening','occluder')}
                boxes=previous['semanticBoxes']
                record['semanticCacheReused']=True
            else:
                groups,boxes=models.infer(image)
            record['semanticBoxes']=boxes
            sky,visibility=skyline_candidates(image,groups,projection['xRangePx'],boxes)
            junctions,rejected=junction_candidates(image,groups,projection['xRangePx'],boxes)
            sky_lines=[c for c in junctions if c['role']=='roof-sky-silhouette-line']
            for line in sky_lines:
                line.update(use='conditional upper-silhouette check only; not an eave/ridge measurement',
                            visibility='explicit-roof-box and sky-above-line; review segmentation boundary')
            sky+=sky_lines
            junctions=[c for c in junctions if c['role']!='roof-sky-silhouette-line']
            record.update(skylineCandidates=sky,junctionCandidates=junctions,visibility=visibility,rejectedLineReasons=rejected,
                status='pixel-candidates-only' if sky or junctions else 'abstained',
                abstentionReasons=[] if sky or junctions else ['No unoccluded semantically supported source transition survived.'])
            np.savez_compressed(args.output/f'{row:02d}-semantic-probabilities.npz',**{k:v.astype(np.float16) for k,v in groups.items()})
            overlay(image,record).save(args.output/f'{row:02d}-proposals.jpg')
        except (ValueError,KeyError,IndexError,OSError) as error:
            record['reason']=str(error)
        results.append(record)
        print(json.dumps({'row':row,'status':record['status'],'skyline':len(record.get('skylineCandidates',[])),
                          'junction':len(record.get('junctionCandidates',[])),'reason':record.get('reason')}),flush=True)
        (args.output/'candidates.json').write_text(json.dumps({'algorithm':'offline-semantic-and-line-roof-proposals-v1',
            'models':model_identities,'algorithmSha256':sha(__file__),'jobsPath':str(args.jobs),'jobsSha256':sha(args.jobs),'records':results,
            'noMetricDepthClaim':True,'automaticConstraintPromotion':False},indent=2)+'\n')
    summary={'jobs':len(jobs),'processed':len(results),'statusCounts':dict(Counter(r['status'] for r in results)),
             'withSkyline':sum(bool(r.get('skylineCandidates')) for r in results),
             'withJunctionCandidates':sum(bool(r.get('junctionCandidates')) for r in results),
             'semanticRoofEvidenceMissing':sum(not any('roof' in b['text'] for b in r.get('semanticBoxes',[])) for r in results),
             'automaticMetricConstraints':0,'reviewRequired':True}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    ray_constraints=[]
    for record in results:
        if 'projection' not in record: continue
        projection=record['projection'];basis=np.asarray(projection['basisRightDownForward'])
        size=np.asarray(record['sourceSizePx']);focal=projection['focalPx']
        for category,items in [('upper-silhouette',record.get('skylineCandidates',[])),('junction-review-candidate',record.get('junctionCandidates',[]))]:
            for candidate in items:
                pixels=np.asarray(candidate['pointsPx'],float)
                directions=np.c_[(pixels+.5-size/2)/focal,np.ones(len(pixels))]@basis
                directions/=np.linalg.norm(directions,axis=1)[:,None]
                ray_constraints.append({'side':record['side'],'image':record['image'],'view':record['view'],
                    'sourcePath':record['sourcePath'],'sourceSha256':record['sourceSha256'],
                    'metadataSha256':record['metadataSha256'],'calibrationSha256':record['calibrationSha256'],
                    'role':candidate['role'],'category':category,'pointsPx':candidate['pointsPx'],
                    'cameraXYZ':projection['cameraXYZ'],'rayDirectionsWorld':directions.tolist(),
                    'pixelUncertaintyPx':candidate['uncertaintyPx'],
                    'angularReadingRadiusDeg':math.degrees(math.atan(candidate['uncertaintyPx']/focal)),
                    'sourceSemanticsVerified':False,'cameraXYValidated':False,
                    'surfaceIdentity':'unresolved host roof versus component/trim',
                    'metricDepthDerived':False,'automaticHardConstraint':False})
    (args.output/'source-ray-constraints.json').write_text(json.dumps({'status':'model-proposed-pixel-rays-only',
        'constraints':ray_constraints,'qualification':'No paired correspondence or depth was inferred. Human source review/visibility and exact surface identity must precede constraint admission.'},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
