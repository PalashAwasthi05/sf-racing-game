"""Attach exact registered source-pixel scale to provisional gable profiles."""
import argparse,json,hashlib
from pathlib import Path
from pipeline.facade_lint import clean
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source
from regen.staged_gable_ownership.roof_gable_ownership import prepare_gable_prior

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('area','profiles','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();data=json.loads(a.profiles.read_text());sides=load_registered_sides(a.area/'regen/sides-reg');records=[]
 for profile in data['profiles']:
  q,_=clean(profile);prior,validation=prepare_gable_prior(q,0.)
  if prior is None:continue
  source=resolve_exact_source(profile,a.area,sides)
  if source['status']!='ok':records.append({'profileId':profile['id'],'status':'unresolved','reason':source['reason']});continue
  entry=sides[source['sideId']];piece=entry['piece'];view=piece['views'][source['selectedViewIndex']];record={'status':'exact-source-joined','side':source['sideId'],'image':str(view['image']),'sourceReadingSha256':source['sourceReadingSha256'],'metadataPath':str(entry['path']),'metadataSha256':hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest(),'pixelsPerMetre':view['ppm'],'normal':profile['normal'],'frameU':profile['frameU'],'planeV':profile['planeV'],'heightM':profile['heightRefM'],'qualification':'Half a source pixel bounds source-frame/corner registration only; source annotation and camera errors remain unquantified.'};profile['sourceFrameRegistration']=record;records.append({'profileId':profile['id'],'status':'joined','halfPixelRegistrationM':.5/view['ppm']})
 data['gableRegistrationCompilation']={'sourceProfilesPath':str(a.profiles),'sourceProfilesSha256':hashlib.sha256(a.profiles.read_bytes()).hexdigest(),'records':records,'confidenceUpgraded':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n');print('joined',sum(r['status']=='joined' for r in records))
if __name__=='__main__':main()
