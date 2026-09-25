"""Publish a full generated candidate as-is: no layers are merged from an earlier build. Copies the material stage's
GLB to final/candidate.glb, chunks it into the viewer's public folder and writes world/model/generation manifests.
Usage: publish_full_candidate_v24.py --candidate <dir> [--label TEXT]"""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--label',default='0.24 / CLEAN REGENERATION')
p.add_argument('--public',type=Path,default=Path('work/king-street/public'))
p.add_argument('--flagged-views-waived',default=None,help='publish without the flagged-view sheets; the reason is recorded in release.json')
a=p.parse_args()
cand,public=a.candidate,a.public;final=cand/'final';final.mkdir(exist_ok=True)
# Every view the user has flagged must have been rendered for THIS build (check_flagged_views.py) before it goes live.
_idx=cand/'flagged-views/index.json';_reg=json.loads(Path('work/review/flagged-views.json').read_text())
if not a.flagged_views_waived:
    if not _idx.exists():raise SystemExit(f'refused: run work/check_flagged_views.py --candidate {cand} and look at the sheets first')
    _ix=json.loads(_idx.read_text());_h=hashlib.sha256()
    with open(cand/'materials/candidate.glb','rb') as _f:
        for _b in iter(lambda:_f.read(1<<24),b''):_h.update(_b)
    if _ix.get('candidateGlbSha256')!=_h.hexdigest():raise SystemExit('refused: the flagged-view sheets are for a different build of this candidate; rerun check_flagged_views.py')
    _missing={v['id'] for v in _reg['views']}-{v['id'] for v in _ix.get('views',[])}
    if _missing:raise SystemExit(f'refused: flagged views without sheets: {sorted(_missing)}')
shutil.copy2(cand/'materials/candidate.glb',final/'candidate.glb')
raw=(final/'candidate.glb').read_bytes();sha=hashlib.sha256(raw).hexdigest();packed=gzip.compress(raw,compresslevel=6,mtime=0);chunks=[]
for i,s in enumerate(range(0,len(packed),4400000)):
    n=f'world-{sha[:12]}-{i:02}.bin';(public/n).write_bytes(packed[s:s+4400000]);chunks.append('/'+n)
world=json.loads((cand/'public/world.json').read_text());world['previewRevision']=a.label
world['updateSummary']='Every layer regenerated from the current general rules in one run; no geometry carried over from earlier builds. Facade source coverage remains incomplete.'
def put(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,separators=(',',':')));tmp.replace(path)
for name in ['height-audit.json','roof-envelope-audit.json','street-space-audit.json','building-parts-audit.json']:
    if (cand/name).exists():shutil.copy2(cand/name,public/name)
if (cand/'public/facade-evidence.json').exists():shutil.copy2(cand/'public/facade-evidence.json',public/'facade-evidence.json')
put(public/'world.json',world)
put(public/'generation.json',{'pipelineVersion':'0.24','fullRegeneration':True,'layersCarriedOver':[],'outputSha256':sha,'candidate':str(final/'candidate.glb'),'independentRealismGate':'NOT PASSED'})
put(public/'model.json',dict(compression='gzip',contentHash=sha[:12],chunks=chunks,compressedBytes=len(packed),uncompressedBytes=len(raw)))
res={'sha256':sha,'localPreview':'http://127.0.0.1:3001/?revision='+sha[:12],'flaggedViews':str(_idx) if not a.flagged_views_waived else {'waived':a.flagged_views_waived}};(cand/'release.json').write_text(json.dumps(res));print(json.dumps(res))
