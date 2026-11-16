"""External launcher for a pure staged generator/roof implementation.

No production rule is edited. The staged generator passes explicit source priors
and topology points; runtime generation contains no frame introspection/runpy.
"""
import argparse,json,sys,types,hashlib,importlib.util
from pathlib import Path
import pipeline
from regen.staged_gable_ownership import roof_gable_ownership
from regen.check_roof_sky_projection import sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--helper',type=Path);p.add_argument('--generator',type=Path);p.add_argument('--roof-module',type=Path);p.add_argument('generator_args',nargs=argparse.REMAINDER);a=p.parse_args();argv=a.generator_args[1:] if a.generator_args[:1]==['--'] else a.generator_args;root=Path(__file__).resolve().parent.parent;stage=root/'regen/staged_gable_ownership';hashes={str(q.relative_to(root)):sha(q) for q in [root/'generate_detailed_v14.py',*sorted((root/'pipeline').glob('*.py'))]};staged_hashes={q.name:sha(q) for q in stage.glob('*.py')};a.audit.parent.mkdir(parents=True,exist_ok=True)
 helper_path=a.helper or stage/'roof_gable_ownership.py';generator_path=a.generator or stage/'generate_detailed_v14.py';roof_path=a.roof_module or stage/'roof_shell.py'
 if a.helper:
  spec=importlib.util.spec_from_file_location('isolated_source_gable_helper',helper_path);chosen_helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(chosen_helper)
 else:chosen_helper=roof_gable_ownership
 audit={'status':'EXTERNAL-PROVISIONAL-NOT-ACCEPTED','productionFilesEdited':False,'observationConfidenceChanged':False,'generatorArgs':argv,'supportMode':'source-wide-lint-gable-outline-unknown-only','productionRuleHashes':hashes,'stagedHashes':staged_hashes,'records':[],'massingOverridesUsed':'--massing' in argv,'actualStagedSources':{str(q):sha(q) for q in [helper_path,generator_path,roof_path]}}
 a.audit.write_text(json.dumps(audit,indent=2)+'\n');module=types.ModuleType('pipeline.roof_shell');module.__package__='pipeline';module.__file__=str(roof_path);exec(compile(roof_path.read_text(),module.__file__,'exec'),module.__dict__);pipeline.roof_shell=module;pipeline.roof_gable_ownership=chosen_helper;sys.modules['pipeline.roof_shell']=module;sys.modules['pipeline.roof_gable_ownership']=chosen_helper;sys.argv=[str(root/'generate_detailed_v14.py'),*argv]
 try:
  context={'__name__':'__main__','__file__':str(root/'generate_detailed_v14.py')};exec(compile(generator_path.read_text(),str(generator_path),'exec'),context)
 finally:
  output=Path(argv[argv.index('--output')+1]);records=output/'roof-envelope-audit.json'
  if records.exists():
   for row in json.loads(records.read_text()):
    shell=row.get('roofShell') or{};proposal=shell.get('sourceGableOwnership')or{}
    if shell.get('sourceGablePriorValidation'):audit['records'].append({'building':row['buildingId'],'changed':bool(proposal.get('appliedBoundaryVertices',0)),'proposal':proposal,'preparation':shell['sourceGablePriorValidation']})
  a.audit.write_text(json.dumps(audit,indent=2)+'\n');changed=[name for name,h in hashes.items() if sha(root/name)!=h]
  if changed:raise RuntimeError('Production changed during isolated stage:'+','.join(changed))
if __name__=='__main__':main()
