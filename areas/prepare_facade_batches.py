"""Sequential, resumable facade photo preparation; never launches the viewer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area',required=True,type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--max-batches',type=int,default=0,help='0 continues until complete or a safety stop')
    args=parser.parse_args()
    if args.max_batches<0:parser.error('--max-batches cannot be negative')
    root=Path(__file__).resolve().parents[2]
    area=args.area.resolve();out=(args.out or area/'regen-small/sides').resolve()
    reports=out.parent/'memory';reports.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'sides.partial.json'
    env={**os.environ,'SFW_AREA':str(area),'PYTHONPATH':str(root/'work'),
         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for iteration in range(args.max_batches or 10**9):
        before=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if before.get('complete'):
            print('All facade photo batches complete',flush=True);return 0
        count=len(before.get('completedBuildings',[]))
        report=reports/f'batch-{count:04d}-{time.time_ns()}.json'
        cmd=[sys.executable,str(root/'work/viewer-bench/guard_memory.py'),
             '--small-cpu-batch','--mib','384','--reserve-mib','3584',
             '--wait-for-green','20','--launch-headroom-mib','4096','--timeout','180',
             '--report',str(report),sys.executable,str(root/'work/regen/sides.py'),
             '--evidence',str(area/'evidence.json'),'--out',str(out),
             '--nominal','--resume','--max-buildings','1']
        with report.with_suffix('.log').open('w') as log:
            result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        status=json.loads(report.read_text()) if report.exists() else {}
        print(json.dumps({'completedBefore':count,'exit':result.returncode,
                          'peakMiB':status.get('peakMiB'),'reason':status.get('reason'),
                          'report':str(report)}),flush=True)
        if result.returncode:return result.returncode
        after=json.loads(checkpoint.read_text())
        if not after.get('complete') and len(after.get('completedBuildings',[]))<=count:
            raise RuntimeError('Batch succeeded without progress; refusing to loop')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
