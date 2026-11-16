"""Read-only source-camera receiver visibility audit for measured profile groups.

Expands eligible repeat banks with profile_apertures.audit_profile (which uses
pipeline.facade_grammar.openings), then casts the original registered camera
rays through each generated group aperture. No geometry or profile is modified.
"""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from pathlib import Path
import numpy as np
from regen.audit_aperture_repair_targets import (
    resolve_exact_source, build_source_camera_job, _feature_scene,
    _receiver_ray_coverage,
)
from regen.glb_stream import GltfStream, bounds_zbase
from regen.profile_apertures import audit_profile
from regen.reading_provenance import load_registered_sides, sha256_file
from regen.host_shell_apertures import TriangleIndex


def run(glb, profiles_path, area_root, output):
    area_root=Path(area_root).resolve()
    doc=json.loads(Path(profiles_path).read_text())
    profiles=doc['profiles']
    sides=load_registered_sides(area_root/'regen/sides-reg')
    os.environ['SFW_AREA']=str(area_root)
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    _,frame=fc.load_world()
    reg_path=area_root/'imagery/registered-cameras.json'
    fc.REGISTRATIONS={k:v for k,v in json.loads(reg_path.read_text()).items() if v.get('accepted')}
    fc.CAMS=apply_registration(fc.load_cameras(frame),reg_path)
    stream=GltfStream(glb)
    records=[]; abstentions=Counter()
    for source_profile in profiles:
        source=resolve_exact_source(source_profile,area_root,sides)
        if source.get('status')!='ok':
            abstentions[source.get('reason','unknown')]+=1
            records.append({'profileId':source_profile.get('id'),'status':'abstain','reason':source.get('reason')})
            continue
        profile,meta=audit_profile(source_profile,source_hash_verified=True,fits=[])
        group_indices=sorted({i for indices in meta['groupOpeningIndices'].values() for i in indices})
        if not group_indices:
            records.append({'profileId':profile.get('id'),'featureId':profile['match']['featureId'],
                'status':'no-eligible-groups','explicitOpeningCount':meta['explicitOpeningCount'],
                'skippedGroups':meta['skippedGroups']})
            continue
        feature_id=profile['match']['featureId']
        job,error=build_source_camera_job(profile,source,area_root,sides)
        if error:
            abstentions[error]+=1;records.append({'profileId':profile['id'],'status':'abstain','reason':error});continue
        rows=stream.rows_for_feature(feature_id);zbase=bounds_zbase(rows,feature_id)
        if zbase is None:
            records.append({'profileId':profile['id'],'status':'abstain','reason':'feature-ground-missing'});continue
        # Include the complete owning feature (base shell and all measured
        # facade groups) but omit unrelated, oversized city geometry. Full
        # world occlusion is handled separately by audit_aperture_repair_targets.
        feature_scene=_feature_scene(stream,feature_id)
        index=TriangleIndex.from_scene(feature_scene)
        group_records=[]
        for gi,indices in meta['groupOpeningIndices'].items():
            per=[]
            for oi in indices:
                op=profile['openings'][oi]
                counts=_receiver_ray_coverage(feature_scene,profile,op,job,zbase,oi,triangle_index=index)
                ratio=counts['expectedReceiverFirst']/max(1,counts['sampled'])
                rec={'openingIndex':oi,'cell':op['_auditCell'],'coverage':counts,'expectedReceiverFraction':round(ratio,5)}
                per.append(rec)
            group_records.append({'groupIndex':int(gi),'openingCount':len(per),'openings':per,
                'aggregate':{k:sum(r['coverage'][k] for r in per) for k in ('sampled','expectedReceiverFirst','baseShellFirst','otherGeometryFirst','skyOrNoHit')}})
        records.append({'profileId':profile['id'],'featureId':feature_id,'status':'audited',
            'sourceImage':job['image'],'registeredState':job['registeredState'],
            'camera':{'from':job['from'],'at':job['at'],'vfovDeg':job['vfovDeg'],'sizePx':[job['width'],job['height']]},
            'explicitOpeningCount':meta['explicitOpeningCount'],'groupOpeningCount':meta['groupOpeningCount'],
            'groupReceiverPolicy':meta['groupReceiverPolicy'],'skippedGroups':meta['skippedGroups'],
            'groups':group_records})
    stream.close()
    result={'schema':'group-source-camera-receiver-rays-v1','readOnly':True,
        'glb':{'path':str(Path(glb).resolve()),'sha256':sha256_file(glb)},
        'profiles':{'path':str(Path(profiles_path).resolve()),'sha256':sha256_file(profiles_path)},
        'profileCount':len(profiles),'statusCounts':dict(Counter(r['status'] for r in records)),
        'abstentions':dict(abstentions),'records':records}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--glb',required=True,type=Path);p.add_argument('--profiles',required=True,type=Path)
    p.add_argument('--area',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();r=run(a.glb,a.profiles,a.area,a.output)
    print(json.dumps({'profiles':r['profileCount'],'statusCounts':r['statusCounts'],'abstentions':r['abstentions']}))
