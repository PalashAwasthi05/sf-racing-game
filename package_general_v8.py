from pathlib import Path
import shutil,json,zipfile,hashlib,ast
out=Path('outputs');work=Path('work');candidate=work/'candidate-v8'
r=json.load(open(candidate/'report.json'));g=json.load(open(candidate/'generation-manifest.json'));a=json.load(open(out/'general-world-area-validation.json'))
assert r['outputHash']==g['outputHash'];assert g['manualProfileDependencies']==0 and not g['releaseAllowed'];assert a['wholeAreaIdNameAndOrderPermutation']
for src,dst in [('candidate.glb','general-world-geometry-diagnostic.glb'),('report.json','general-world-generation-report.json'),('generation-manifest.json','general-world-generation-manifest.json'),('acquisition-queue.json','general-world-acquisition-queue.json')]:shutil.copy2(candidate/src,out/dst)
# Keep the previous area-specific bundle historical; the current source deliverable is pure general code.
old=out/'king-street-pilot-bundle.zip';history=work/'history/v7'
if old.exists() and not (history/old.name).exists():shutil.move(old,history/old.name)
if (out/'king-street-pilot-notes.md').exists():
 shutil.copy2(out/'king-street-pilot-notes.md',history/'king-street-pilot-notes.md')
 (out/'king-street-pilot-notes.md').write_text('# Current reconstruction status\n\nThe hosted scene is the prior visual prototype. The active heuristic has been replaced by the location-independent `world-generation-blueprint.md`. The clean geometry diagnostic and its failed appearance gate are documented in `reconstruction-root-cause-report.md`. It is not published as an accuracy improvement.\n')
files=[p for p in (work/'general_world').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files += [work/n for n in ['build_world.py','rebuild_world.py','normalize_area_inputs.py','test_general_world.py','general-world-requirements.txt']]
# Static import check: no earlier authored-data pipeline is part of the active core.
imports=[]
for p in files:
 if p.suffix!='.py':continue
 tree=ast.parse(p.read_text())
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imports.extend(a.name for a in n.names)
  elif isinstance(n,ast.ImportFrom):imports.append(n.module or '')
assert not any(n=='pipeline' or n.startswith('pipeline.') for n in imports)
files += [out/'world-generation-blueprint.md',out/'reconstruction-implementation-notes.md']
with zipfile.ZipFile(out/'general-world-heuristic-and-source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,str(p.relative_to(work)) if p.is_relative_to(work) else p.name)
with zipfile.ZipFile(out/'general-world-heuristic-and-source.zip') as z:
 assert not any(any(x in n for x in ['normalized-v8','observations.json','facade-profiles','area-source-job']) for n in z.namelist())
summary={'generalSourceFiles':len(files),'legacyPipelineImports':0,'manualProfileDependencies':0,'sourceArchiveContainsAreaData':False,'buildingsAttempted':r['buildingsAttempted'],'closedSolids':r['closedBuildings'],'shapeGeometriesComparedAfterRelabel':a['geometriesCompared'],'appearanceGate':r['appearanceGate'],'published':False,'sourceArchiveSha256':hashlib.sha256((out/'general-world-heuristic-and-source.zip').read_bytes()).hexdigest()}
(out/'general-world-release-status.json').write_text(json.dumps(summary,indent=2));print(summary)
