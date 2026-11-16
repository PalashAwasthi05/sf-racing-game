"""Discover every provenance-joined lint gable source, without scene ID rules."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from pipeline.facade_lint import clean
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source
from regen.staged_gable_ownership.roof_gable_ownership import prepare_gable_prior


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('area','profiles','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);profiles=json.loads(a.profiles.read_text())['profiles'];sides=load_registered_sides(a.area/'regen/sides-reg');rows=[];selected=[]
 for original in profiles:
  profile,lint=clean(original)
  if not profile.get('gableOutline'):continue
  prior,audit=prepare_gable_prior(profile,0.);source=resolve_exact_source(original,a.area,sides) if prior else None;match=profile.get('match') or {};identity=match.get('partId') or match.get('featureId')
  status='eligible-provisional-source-gable' if prior and source.get('status')=='ok' and identity else 'unresolved'
  row={'profileId':profile['id'],'featureId':match.get('featureId'),'partId':match.get('partId'),'rebuildIdentity':identity,'status':status,'preparation':audit,'exactSource':source,'lintIssues':lint,'confidenceUpgraded':False,'imageVerified':False,'cameraXYValidated':False}
  if status=='eligible-provisional-source-gable':
   selected.append(identity);row['localOutlineM']=[[float(u-profile['frameU'][0]),float(z)] for u,z in prior['traceUZ']];row['sourceProfileSha256']=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  rows.append(row)
 ids=sorted(set(selected));(a.output/'all-eligible-feature-ids.json').write_text(json.dumps(ids,indent=2)+'\n');(a.output/'audit.json').write_text(json.dumps({'status':'source-wide-provisional-discovery-not-acceptance','profilesPath':str(a.profiles),'profilesSha256':hashlib.sha256(a.profiles.read_bytes()).hexdigest(),'allProfileCount':len(profiles),'lintPreservedGableCount':len(rows),'eligibleProfileCount':len(selected),'eligibleIdentityCount':len(ids),'reasonCounts':dict(Counter(r['preparation'].get('reason') or (r.get('exactSource') or {}).get('reason') or r['status'] for r in rows)),'records':rows,'qualification':'All eligible joins are discovered; no native support or photo shape is accepted by this list. Boundary-domain/anchor/protected-part checks still run per feature.'},indent=2)+'\n');print(json.dumps({'profiles':len(profiles),'gableOutlines':len(rows),'eligibleProfiles':len(selected),'eligibleIdentities':len(ids)}))
if __name__=='__main__':main()
