"""Publish one selected, fitted corridor to the existing local driving viewer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from shutil import copyfile
from statistics import median
from tempfile import NamedTemporaryFile


def publish(source, directory):
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    name = f'{source.stem}-{digest[:16]}{source.suffix}'
    target = directory / name
    if not target.exists():
        with NamedTemporaryFile(dir=directory, delete=False) as temp:
            temp_name = temp.name
        copyfile(source, temp_name)
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, target)
    return '/block-trial/' + name


def observed_lane(area, route, train):
    # Map the original camera trajectory onto the OSM carriageway. One common offset keeps
    # the route on a consistent lane and rejects isolated GPS excursions at junctions.
    line = [p[:2] for p in route['enu']]
    times, offsets = [], []
    for cam in train:
        pos = cam['eyeEnu'][:2]
        candidates = []
        for p, q in zip(line, line[1:]):
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = math.hypot(dx,dy)
            if not length: continue
            along = max(0, min(1, ((pos[0]-p[0])*dx + (pos[1]-p[1])*dy)/length**2))
            ex, ey = pos[0]-p[0]-along*dx, pos[1]-p[1]-along*dy
            candidates.append((math.hypot(ex,ey),(ex*dy-ey*dx)/length))
        offsets.append(min(candidates)[1])
        entity = json.loads((area/'imagery/entities'/(str(cam['id'])+'.json')).read_text())
        if entity.get('captured_at') is not None:
            times.append((entity['captured_at'],cam['alongM']))
    offset = median(offsets)
    shifted = []
    for i,p in enumerate(line):
        a,b = line[max(0,i-1)],line[min(i+1,len(line)-1)]
        dx,dy = b[0]-a[0],b[1]-a[1]
        length = math.hypot(dx,dy)
        shifted.append([round(p[0]+offset*dy/length,4),round(p[1]-offset*dx/length,4)])
    times.sort()
    trends = [b[1]-a[1] for i,a in enumerate(times) for b in times[i+1:] if a[0]!=b[0]]
    if trends and median(trends)<0:
        shifted.reverse()
    return shifted,offset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--area',required=True,type=Path)
    ap.add_argument('--public',required=True,type=Path)
    ap.add_argument('--baseline',required=True,type=Path)
    ap.add_argument('--comparison',type=Path,help='Finished same-camera comparison sheet for these assets')
    ap.add_argument('--splat-dir',type=Path,help='Directory holding the chosen frozen Gaussian fit')
    args=ap.parse_args()
    trial=args.area/'block-splat-trial'
    output=args.public/'block-trial'
    output.mkdir(parents=True,exist_ok=True)
    cameras=json.loads((trial/'cameras.json').read_text())
    train,held=cameras['train'],cameras['heldout']
    if {r['driveId'] for r in train} & {r['driveId'] for r in held}:
        raise ValueError('Training and test drives overlap')
    photo=trial/'photo-mesh.glb'
    splats=(args.splat_dir or trial)/'block-splats.bin'
    splat_report=json.loads(splats.with_suffix('.json').read_text())
    photo_report=json.loads((trial/'photo-mesh-report.json').read_text())
    for asset,report in ((photo,photo_report),(splats,splat_report)):
        with asset.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if report['sha256']!=digest:
            raise ValueError(f'Fit receipt does not match {asset.name}')
    size=splats.stat().st_size
    if size<40 or size%40 or photo.stat().st_size<1024:
        raise ValueError('Incomplete fitted representations')
    route,offset=observed_lane(args.area,cameras['route'],train)
    manifest=json.loads(args.baseline.read_text())
    if not manifest.get('world') or not manifest.get('chunks'):
        raise ValueError('Missing base world or geometry')
    attribution=publish(trial/'source-attributions.json',output)
    config = dict(label=cameras['route']['label'],routeEN=route,lengthM=cameras['route']['lengthM'],
                  observedLaneOffsetM=round(offset,3),manifest=publish(args.baseline,output),
                  photo=publish(photo,output),splats=publish(splats,output),count=size//40,
                  trainFrames=len(train),heldoutFrames=len(held),sourceUrl=train[0]['sourceUrl'],attribution=attribution)
    if args.comparison:
        metrics=json.loads((args.comparison.parent/'metrics.json').read_text())
        frozen=metrics['frozenTrial']
        if any(frozen[k]!=config[k] for k in ('manifest','photo','splats','routeEN')):
            raise ValueError('Comparison pictures refer to different assets or route')
        config['comparisonUrl']=publish(args.comparison,output)
    with NamedTemporaryFile(dir=output,mode='w',delete=False) as temp:
        json.dump(config,temp,indent=2)
        temp.write('\n')
    os.chmod(temp.name,0o644)
    os.replace(temp.name,output/'trial.json')
    print(json.dumps(config,indent=2))


if __name__=='__main__':
    main()
