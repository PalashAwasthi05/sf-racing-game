from pathlib import Path
import json,subprocess
P=Path('work/pipeline');repo=Path('work/king-street')
recipes={'brick_modular':('brick','default'),'brick_dark':('brick','dark'),'terracotta_panel':('terracotta','default'),'terracotta_ribbed':('terracotta','ribbed'),'stucco':('render','default'),'stucco_red':('render','red'),'panel_dark':('metal-panel','dark'),'panel_silver':('metal-panel','default'),'stone':('pale-trim','default')}
for name in ['facade-profiles.json','envelope-observations.json']:
 d=json.load(open(P/name));old=json.loads(subprocess.check_output(['git','show','7d360140fc66f6eaf00264bdba4ced0c8b1e055f:pipeline-source/pipeline/'+name],cwd=repo))
 oldby={p['id']:p for p in old['profiles']}
 for p in d['profiles']:
  def visit(node,prior):
   m=node.get('material')
   if m in recipes:
    family,finish=recipes[m];node['materialEvidence']={'family':family,'finish':finish,'sourceIds':p.get('sourceIds',[]),'status':'visually observed family; appearance parameters estimated'}
    if prior and 'material' in prior:node['material']=prior['material']
   oldchildren={c['id']:c for c in (prior or {}).get('children',[])}
   for child in node.get('children',[]):visit(child,oldchildren.get(child['id']))
  if 'nodes' in p:
   oldnodes={n['id']:n for n in oldby[p['id']]['nodes']}
   for n in p['nodes']:visit(n,oldnodes.get(n['id']))
  else:visit(p,oldby[p['id']])
 if name=='facade-profiles.json':
  loft=d['profiles'][0];loft['openings'][0]['id']='garage';loft['openings'][1]['id']='residential-entry'
 (P/name).write_text(json.dumps(d,indent=2))
p=P/'lettering-observations.json';d=json.load(open(p))
for a in d['signs']:
 if a['id']=='loft-address':a['attachment']={'kind':'opening-canopy','profileId':'deep-framed-loft','openingId':'residential-entry'}
 elif a.get('canopy'):a['attachment']={'kind':'generated-canopy'}
 elif a['id']=='glassworks-third-flagship':a['attachment']={'kind':'envelope-surface','nodeId':'recessed-commercial-base'}
 elif a['id'].startswith('glassworks'):a['attachment']={'kind':'glazing','nodeId':'king-commercial-base'}
 elif a['profileId']=='clock-tower-side':a['attachment']={'kind':'envelope-surface','nodeId':'clock-return'}
 else:a['attachment']={'kind':'envelope-surface','nodeId':'plaza-clock-tower'}
p.write_text(json.dumps(d,indent=2))
p=Path('work/build_world.py');s=p.read_text().replace('signs,envelopes,lettering','signs,envelopes,lettering,surface_evidence')
s=s.replace("glb=scene.export(file_type='glb');", "audit={'rules':surface_evidence.AUDIT,'signAttachments':[r['attachmentCheck'] for r in lettering_records],'unverifiedFacadeSections':sum(r['status']=='inferred' for r in facade_records),'independentAccuracyGate':'NOT PASSED'}\n(OUT/'king-street-rule-audit.json').write_text(json.dumps(audit,indent=2))\nglb=scene.export(file_type='glb');")
s=s.replace("'surfaceSources':LETTERING_DATA['sources']", "'surfaceSources':LETTERING_DATA['sources'],'ruleAudit':audit")
p.write_text(s)
