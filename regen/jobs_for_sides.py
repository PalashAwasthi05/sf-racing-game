"""Resolve side ids from evidence into reproducible, source-photo comparison cameras."""
import argparse,json,os
from pathlib import Path
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source,build_source_camera_job
def main():
 a=argparse.ArgumentParser()
 for k in ('area','profiles','sides','output'):a.add_argument('--'+k,required=True,type=Path)
 a=a.parse_args();area=a.area.resolve();os.environ['SFW_AREA']=str(area)
 import frontage_crops_v15 as fc
 from regen.register_cameras import apply_registration
 _,fr=fc.load_world();reg=area/'imagery/registered-cameras.json';fc.REGISTRATIONS={k:v for k,v in json.loads(reg.read_text()).items() if v.get('accepted')};fc.CAMS=apply_registration(fc.load_cameras(fr),reg);sides=load_registered_sides(area/'regen/sides-reg')
 ps={p['id'].removeprefix('regen-'):p for p in json.loads(a.profiles.read_text())['profiles']};jobs=[]
 for k in json.loads(a.sides.read_text()):
  k=k.removeprefix('regen-');p=ps[k];src=resolve_exact_source(p,area,sides)
  if src['status']!='ok':continue
  j,e=build_source_camera_job(p,src,area,sides)
  if not e:jobs.append({**j,'side':k,'building':p['match']['featureId'],'persp':src['perspPath'],'photoSha256':src['perspSha256']})
 a.output.write_text(json.dumps(jobs,indent=1));print(len(jobs))
if __name__=='__main__':main()
