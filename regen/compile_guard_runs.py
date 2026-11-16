"""Resolve repeated window guards from a semantic reading and source-image boxes.

The notes classifier can nominate a run but cannot change geometry by itself.
At least two independently detected panels in the exact reading photograph must
support the repeated layout. Other views can disagree because of registration;
they are retained as evidence but never mixed into this metric frame.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def infer_guards(reading, proposal, measurements, source_image):
    probability=(proposal.get('answer',{}).get('probabilities') or {}).get('individual_window_guards',0.)
    if probability<.85:
        return None, {'status':'unresolved','reason':'reading does not establish individual guards'}
    supported=[]
    for m in measurements:
        if str(m.get('image'))!=str(source_image) or not m.get('boundsInsideSource'):
            continue
        answers=m.get('answers',{})
        if answers.get('visible',{}).get('noul',0.)<.65:
            continue
        bounds=m.get('measuredRectM')
        if not bounds or len(bounds)!=4 or not np.isfinite(bounds).all():continue
        bank_index=m.get('bank');cell=m.get('cell');run=m.get('balcony')
        banks=reading.get('banks',[]);runs=reading.get('balconies',[])
        if not all(isinstance(i,int) and not isinstance(i,bool) for i in (bank_index,cell,run)):continue
        if not 0<=bank_index<len(banks) or not 0<=run<len(runs):continue
        bank=banks[bank_index];balcony=runs[run]
        count=int(bank.get('columns',1))
        if count<2 or not 0<=cell<count or int(bank.get('rows',1))!=1:continue
        centers=bank.get('xCentersM') or [bank['x0']+(i+.5)*(bank['x1']-bank['x0'])/count for i in range(count)]
        opening_width=bank.get('openingWidthM')
        if not isinstance(opening_width,(int,float)) or opening_width<=0 or len(centers)!=count:continue
        x0,z0,x1,z1=map(float,bounds);width=x1-x0;height=z1-z0
        if not .7*opening_width<=width<=1.6*opening_width or not .15<=height<=1.1:continue
        if abs((x0+x1)/2-centers[cell])>.35*opening_width:continue
        if abs(z0-float(balcony['zM']))>.3 or float(balcony.get('depthM',.8))>.25:continue
        supported.append({**m,'widthRatio':width/opening_width,'heightM':height})
    independent={(m['balcony'],m['cell']) for m in supported}
    if len(independent)<2:
        return None,{'status':'unresolved','reason':'fewer than two visible source panels','supportCount':len(independent)}
    out=copy.deepcopy(reading);replacements={};proof=[]
    for run in proposal.get('recordIndices',[]):
        support=[m for m in supported if m['balcony']==run]
        if not support:continue
        bank_ids={m['bank'] for m in support}
        if len(bank_ids)!=1:continue
        bank=reading['banks'][next(iter(bank_ids))];balcony=reading['balconies'][run]
        count=int(bank['columns']);centers=bank.get('xCentersM') or [bank['x0']+(i+.5)*(bank['x1']-bank['x0'])/count for i in range(count)]
        ratios=[m['widthRatio'] for m in support]
        if max(ratios)-min(ratios)>.25:continue
        width=float(np.median(ratios))*bank['openingWidthM']
        # Preserve the original measured vertical course. Box observations
        # verify the layout, not a license to average different camera heights.
        panels=[]
        for i,center in enumerate(centers):
            if not balcony['x0']-.2<=center<=balcony['x1']+.2:continue
            panels.append({**balcony,'x0':float(center-width/2),'x1':float(center+width/2),
                           'hasDeck':False,'layoutEvidence':{
                               'method':'source-confirmed repeated window guards',
                               'image':str(source_image),'observedCell':i in {m['cell'] for m in support},
                               'inferredRepeat':i not in {m['cell'] for m in support}}})
        if len(panels)<2:continue
        replacements[run]=panels
        proof.append({'run':run,'bank':next(iter(bank_ids)),'panels':len(panels),
                      'sourceSupportCells':sorted({m['cell'] for m in support}),
                      'measuredWidthM':width,'sourceBoxes':[m['measuredRectM'] for m in support]})
    if not replacements:return None,{'status':'unresolved','reason':'no consistent repeated run'}
    out['balconies']=[q for i,b in enumerate(reading.get('balconies',[])) for q in replacements.get(i,[b])]
    return out,{'status':'compiled','runs':proof,'unresolvedPattern':True,
                'claim':'layout correction only; decorative aperture pattern not reconstructed'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('proposals','measurements','sides','output'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args()
    from regen.reading_provenance import load_registered_sides
    sides=load_registered_sides(args.sides)
    measurements=json.loads(args.measurements.read_text())['rows']
    proposals=json.loads(args.proposals.read_text())['rows']
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for proposal in proposals:
        path=Path(proposal['reading']);reading=json.loads(path.read_text());sid=reading['id']
        source=sides.get(sid)
        if not source or source.get('status')!='ok':continue
        if reading.get('calibratedCrops'):continue # requires a matching calibrated crop catalog
        piece=source['piece'];use=reading.get('use',0)
        if not isinstance(use,int) or not 0<=use<len(piece.get('views',[])):continue
        view=piece['views'][use];selected=[]
        for m in measurements:
            if Path(m['reading']).resolve()!=path.resolve():continue
            source_path=Path(m['source'])
            if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest()!=m['sourceSha256']:continue
            if source_path.name!=view.get('rect') or str(m['image'])!=str(view['image']):continue
            selected.append(m)
        updated,report=infer_guards(reading,proposal,selected,view['image'])
        report.update(side=sid,sourceReading=str(path),sourceReadingSha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if updated:
            target=args.output/path.name;target.write_text(json.dumps(updated,indent=2)+'\n')
            report['output']=str(target)
        reports.append(report)
    (args.output/'guard-run-manifest.json').write_text(json.dumps({'status':'candidate, pending regenerated visual review','decisions':reports},indent=2)+'\n')
    print(json.dumps({'considered':len(proposals),'compiled':sum(r['status']=='compiled' for r in reports)}))


if __name__=='__main__':main()
