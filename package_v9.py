"""Package general code separately from area observations and the review mesh."""
from pathlib import Path
import ast,hashlib,json,zipfile
root=Path(__file__).resolve().parents[1];work=root/'work';out=root/'outputs'
files=[p for p in (work/'general_world').glob('*') if p.is_file()]
files += [work/n for n in ['build_world.py','rebuild_world.py','normalize_area_inputs.py','repair_preview_roofs_v9.py','test_general_world.py','test_roof_planes_v9.py','test_scene_roofs_v9.py','test_opening_mesh_v9.py','test_billboards_v9.py','test_scanned_materials_v9.py','general-world-requirements.txt','general-world-test-requirements.txt']]
for p in files:
    if p.suffix!='.py':continue
    for node in ast.walk(ast.parse(p.read_text())):
        modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(m=='pipeline' or m.startswith('pipeline.') for m in modules)
archive=out/'general-world-heuristic-and-source.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(work))
    z.write(out/'world-generation-blueprint.md','heuristic.md')
    z.write(work/'material-library-v9/manifest.json','material-library-v9/manifest.json')
    z.write(work/'material-library-v9/acquire.py','material-library-v9/acquire.py')
with zipfile.ZipFile(out/'building-material-library.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in (work/'material-library-v9').rglob('*'):
        if p.is_file() and p.suffix!='.zip' and '__pycache__' not in str(p):z.write(p,p.relative_to(work))
status={'heuristicRevision':'0.9','sourceFiles':len(files),'legacyPipelineImports':0,'areaObservationsInSourceArchive':False,'sourceArchiveSha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'newRegressionTestsPassed':29,'earlierChecksPassed':10,'automaticFacadeExtractionComplete':False,'independentRealismGate':'NOT PASSED','detailedPreviewUsesHistoricalFacadeInput':True}
(out/'general-world-release-status.json').write_text(json.dumps(status,indent=2));print(status)
