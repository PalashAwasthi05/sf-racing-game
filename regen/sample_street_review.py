"""Choose repeatable, geographically spread street-photo viewpoints outside a tuned cohort.
Visits actual source profiles across distinct material and frontage conditions; selection is
fixed before rendering, never based on whether a generated building looks good.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from regen.audit_aperture_repair_targets import resolve_exact_source,build_source_camera_job
from regen.reading_provenance import load_registered_sides


def main():
    ap=argparse.ArgumentParser()
    for key in ('area','profiles','exclude','garden-audit','output'):ap.add_argument('--'+key,required=True,type=Path)
    ap.add_argument('--count',type=int,default=8);a=ap.parse_args();area=a.area.resolve();os.environ['SFW_AREA']=str(area)
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    _,frame=fc.load_world();reg=area/'imagery/registered-cameras.json';fc.REGISTRATIONS={k:v for k,v in json.loads(reg.read_text()).items() if v.get('accepted')};fc.CAMS=apply_registration(fc.load_cameras(frame),reg)
    excluded=set(json.loads(a.exclude.read_text()));garden={x['side'] for x in json.loads(a.garden_audit.read_text())['changed']};sides=load_registered_sides(area/'regen/sides-reg');opts=[]
    for p in json.loads(a.profiles.read_text())['profiles']:
        fid=p['match']['featureId'];sid=p['id'].removeprefix('regen-')
        if fid in excluded:continue
        s=resolve_exact_source(p,area,sides)
        if s['status']!='ok':continue
        j,err=build_source_camera_job(p,s,area,sides)
        if err:continue
        j.update(side=sid,building=fid,persp=s['perspPath'],photoSha256=s['perspSha256'],material=p['material'].split('#')[0].split('~')[0],changedGarden=p['id'] in garden,registered=s['registeredState'] if 'registeredState' in s else j['registeredState']);opts.append(j)
    chosen=[];by={};ids=set()
    # Each stratum's first view is independently hash-ordered. Subsequent choices
    # favour spatial spread and an unused stratum; no source or scene ID whitelist.
    opts.sort(key=lambda j:hashlib.sha256(j['side'].encode()).digest())
    while opts and len(chosen)<a.count:
        def score(j):
            stratum=(j['material'],j['changedGarden']);return (-by.get(stratum,0),min([180]+[np.linalg.norm(np.array(k['from'][:2])-j['from'][:2]) for k in chosen]))
        j=max(opts,key=score);chosen.append(j);q=(j['material'],j['changedGarden']);by[q]=by.get(q,0)+1;opts=[k for k in opts if k['building']!=j['building'] and k['image']!=j['image']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(chosen,indent=1));print('selected',len(chosen),'street views',by)
if __name__=='__main__':main()
