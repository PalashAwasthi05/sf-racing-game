"""Transactionally patch source-supported closed apertures in one GLB.

The input BIN chunk is copied byte-for-byte; replacement meshes and measured
receiver parts are appended as new accessors. Each affected feature is staged
and source-camera ray checked before any of its nodes are committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from regen.glb_stream import GltfStream,GlbAppendWriter,SceneView,bounds_zbase
from regen.host_shell_apertures import (
    GLB_TO_SOURCE,SOURCE_TO_GLB,TriangleIndex,clip_shell_mesh,
    opening_host_conflicts,opening_slab,reveal_mesh,
)
from regen.prototype_host_shell_apertures import (
    _receiver_node_expected,_receiver_ray_coverage,build_measured_receiver,
    opening_center_source,
)
from regen.audit_aperture_repair_targets import (
    CompositeRayIndex,_feature_scene,_ray_neighborhood_scene,
    _segment_clears_ground,build_source_camera_job,resolve_exact_source,
)
from regen.reading_provenance import load_registered_sides,sha256_file,relative_path
from regen.profile_apertures import audit_profile


def _clone_scene(scene):
    return SceneView(dict(scene.graph._entries),dict(scene.geometry))


def _node_mesh(scene,node):
    _,geom=scene.graph.get(node)
    return scene.geometry[geom]


def _set_node_mesh(scene,node,mesh):
    _,geom=scene.graph.get(node)
    scene.geometry[geom]=mesh


def _add_scene_mesh(scene,node,mesh,transform):
    geom=f'patch_{len(scene.geometry)}'
    scene.geometry[geom]=mesh
    scene.graph._entries[node]=(geom,np.asarray(transform,float))
    scene.graph.nodes_geometry=list(scene.graph._entries)


def _overlay_feature(ray_scene,working_feature,feature_id):
    out=_clone_scene(ray_scene)
    for node in working_feature.graph.nodes_geometry:
        if not (node.startswith(f'buildings__{feature_id}__') or
                node.startswith(f'facades__{feature_id}__')):continue
        if node in out.graph._entries:
            _,old_geom=out.graph.get(node)
            _,work_geom=working_feature.graph.get(node)
            out.geometry[old_geom]=working_feature.geometry[work_geom]
        else:
            _,work_geom=working_feature.graph.get(node)
            _add_scene_mesh(out,node,working_feature.geometry[work_geom],
                            working_feature.graph.get(node)[0])
    return out


def _mesh_material_index(stream,node):
    row=stream.by_node.get(node)
    return None if row is None else row.get('material')


def _prepare_feature(stream,profiles_by_id,audit_records,area_root,sides,fc,writer,
                     progress=None):
    """Stage one feature. Return geometry edits only if every receiver passes."""
    records=[r for r in audit_records if r.get('status')=='supported-host-aperture-conflict' and r.get('conflicts')]
    feature_ids=sorted({r['featureId'] for r in records})
    outcomes=[];all_patches=[]
    for feature_index,feature_id in enumerate(feature_ids,1):
        if progress:progress(f'patch staging feature {feature_index}/{len(feature_ids)} {feature_id}')
        feature_records=[r for r in records if r['featureId']==feature_id]
        original=_feature_scene(stream,feature_id)
        if not original.graph.nodes_geometry:
            outcomes.append({'featureId':feature_id,'status':'rollback','reason':'feature-geometry-missing'});continue
        working=_clone_scene(original)
        zbase=bounds_zbase(stream.rows_for_feature(feature_id),feature_id)
        if zbase is None:
            outcomes.append({'featureId':feature_id,'status':'rollback','reason':'feature-base-elevation-missing'});continue
        replacements={};added=[];feature_checks=[];failure=None
        for record in feature_records:
            profile=profiles_by_id.get(record['profileId'])
            if profile is None:
                failure='compiled-profile-missing';break
            source=resolve_exact_source(profile,area_root,sides)
            if source['status']!='ok' or source['image']!=record['source']['image']:
                failure='source-provenance-changed';break
            profile, _ = audit_profile(profile, source_hash_verified=True, fits=[])
            job,error=build_source_camera_job(profile,source,area_root,sides)
            if error:
                failure=error;break
            expected={int(c['openingIndex']):c for c in record['conflicts']}
            clear_indices=[]
            for index in expected:
                target=opening_center_source(profile,profile['openings'][index],zbase)
                if _segment_clears_ground(job['from'],target,fc):clear_indices.append(index)
                else:
                    failure=f'terrain-ray-not-clear-opening-{index}';break
            if failure:break
            ray_scene,large_rows=_ray_neighborhood_scene(
                stream,feature_id,profile,job,zbase,stream.node_bounds,list(expected),interiors=True)
            index=CompositeRayIndex(TriangleIndex.from_scene(ray_scene),stream,large_rows)
            conflicts=opening_host_conflicts(ray_scene,profile,job,zbase,
                index=clear_indices,triangle_index=index)
            by_opening={c['openingIndex']:c for c in conflicts}
            if set(by_opening)!=set(expected):
                failure='owner-proof-changed-since-audit';break
            for opening_index,expected_conflict in expected.items():
                conflict=by_opening[opening_index]
                if conflict['firstHit']['node']!=expected_conflict['hostNode']:
                    failure=f'host-node-changed-opening-{opening_index}';break
                host_node=conflict['firstHit']['node']
                opening=profile['openings'][opening_index]
                before=_receiver_ray_coverage(ray_scene,profile,opening,job,zbase,
                    opening_index,triangle_index=index)
                host_row=stream.by_node.get(host_node)
                if host_row is None:
                    failure=f'host-primitive-unavailable-opening-{opening_index}';break
                _,host_geom=working.graph.get(host_node)
                current_host=working.geometry[host_geom]
                # Host helpers operate in source metric coordinates. GLB node
                # meshes are usually local-origin packed, so always transform
                # first instead of heuristically assuming they share a frame.
                source_from_local=GLB_TO_SOURCE@host_row['transform']
                world=current_host.vertices@host_row['transform'][:3,:3].T+host_row['transform'][:3,3]
                source_vertices=world@GLB_TO_SOURCE[:3,:3].T+GLB_TO_SOURCE[:3,3]
                source_host=current_host.copy();source_host.vertices=source_vertices
                source_normals=np.asarray(current_host.vertex_normals,dtype=float) @ np.linalg.inv(source_from_local[:3,:3])
                source_normals/=np.maximum(np.linalg.norm(source_normals,axis=1,keepdims=True),1e-12)
                source_host.vertex_normals=source_normals
                slab=opening_slab(source_host,profile,conflict,zbase)
                if slab is None:
                    failure=f'host-slab-no-longer-supported-opening-{opening_index}';break
                clipped_source,stats=clip_shell_mesh(
                    source_host,profile,conflict['outlineUV'],slab,zbase)
                if stats['affectedTriangles']<=0 or stats['removedProjectedAreaM2']<=1e-8:
                    failure=f'host-cut-empty-opening-{opening_index}';break
                local_from_source=np.linalg.inv(source_from_local)
                clipped=clipped_source.copy()
                clipped.vertices=np.asarray(clipped_source.vertices)@local_from_source[:3,:3].T+local_from_source[:3,3]
                local_normals=np.asarray(clipped_source.vertex_normals,dtype=float)@source_from_local[:3,:3]
                local_normals/=np.maximum(np.linalg.norm(local_normals,axis=1,keepdims=True),1e-12)
                clipped.vertex_normals=local_normals
                _set_node_mesh(working,host_node,clipped)
                replacements[host_node]=clipped
                host_material_index=host_row.get('material')
                reveal_material=None
                reveal_name=f"facades__{feature_id}__observed_{profile['id']}_aperture_{opening_index}_reveal___wall"
                reveal=reveal_mesh(profile,conflict['outlineUV'],slab,zbase)
                reveal_transform=SOURCE_TO_GLB
                _add_scene_mesh(working,reveal_name,reveal,reveal_transform);added.append((reveal_name,reveal,host_material_index,None))
                action=expected_conflict['proposedAction']
                if (action=='cut-host-rebuild-measured-receiver' and
                        opening.get('_receiverPolicy') != 'preserve-existing-only'):
                    parts=build_measured_receiver(profile,profile['openings'][opening_index],
                                                  zbase,opening_index=opening_index)
                    if not parts:
                        failure=f'measured-receiver-empty-opening-{opening_index}';break
                    for part_index,part in enumerate(parts):
                        material=part['mesh'].visual.material
                        mat_name=str(getattr(material,'name','material'))
                        node=(f"facades__{feature_id}__observed_{profile['id']}_"
                              f"aperture_{opening_index}_part_{part_index}_"
                              f"{part['role'].replace('-','_')}___{mat_name}")
                        _add_scene_mesh(working,node,part['mesh'],SOURCE_TO_GLB)
                        # The streaming reader leaves images and materials on
                        # disk. Reuse the authored GLB material by ownership;
                        # its decoded mesh carries only a placeholder visual.
                        suffix='___'+mat_name
                        matches=[row for row in stream.rows_for_feature(feature_id)
                                 if row['node'].startswith(f'facades__{feature_id}__observed_{profile["id"]}')
                                 and row['node'].endswith(suffix)]
                        slot=matches[0]['material'] if matches else None
                        added.append((node,part['mesh'],slot,material))
                # Validate this opening against the actual source camera and
                # nearby world geometry after all feature changes staged so far.
                candidate_ray=_overlay_feature(ray_scene,working,feature_id)
                candidate_index=CompositeRayIndex(TriangleIndex.from_scene(candidate_ray),
                                                  stream,large_rows)
                coverage=_receiver_ray_coverage(candidate_ray,profile,opening,job,zbase,
                    opening_index,triangle_index=candidate_index)
                from regen.prototype_host_shell_apertures import _center_expected_receiver_hit
                center_hit=_center_expected_receiver_hit(candidate_ray,profile,opening,job,
                    zbase,opening_index,triangle_index=candidate_index)
                expected_ratio=coverage['expectedReceiverFirst']/max(1,coverage['sampled'])
                floor=.55 if action=='cut-host-rebuild-measured-receiver' else .30
                improved=(coverage['expectedReceiverFirst']>=before['expectedReceiverFirst']
                          and coverage['baseShellFirst']<before['baseShellFirst'])
                if (center_hit is None or expected_ratio<floor or coverage['skyOrNoHit']>0 or not improved):
                    failure=f'receiver-ray-guard-failed-opening-{opening_index}'
                    feature_checks.append({'profileId':profile['id'],'openingIndex':opening_index,'coverage':coverage,'before':before,
                        'centerExpected':center_hit is not None,'guard':'failed'})
                    break
                feature_checks.append({'profileId':profile['id'],'openingIndex':opening_index,'coverage':coverage,'before':before,
                    'centerExpected':True,'centerNode':center_hit['node'],
                    'guard':'passed','removedProjectedAreaM2':stats['removedProjectedAreaM2'],
                    'affectedTriangles':stats['affectedTriangles']})
            if failure:break
        if failure:
            outcomes.append({'featureId':feature_id,'status':'rollback','reason':failure,
                             'openingChecks':feature_checks});continue
        # Convert staged scene edits into append-only GLB writer operations only
        # after every opening on this feature passed its source-camera guards.
        feature_patches=[]
        for node,mesh in replacements.items():
            material_index=_mesh_material_index(stream,node)
            feature_patches.append(('replace',node,mesh,material_index,
                {'featureId':feature_id,'layer':'building-facade'}))
        for node,mesh,material_index,material in added:
            feature_patches.append(('add',node,mesh,material_index,material,
                                    {'featureId':feature_id,'layer':'facade',
                                     'sourceProfiles':[r['profileId'] for r in feature_records]}))
        all_patches.extend(feature_patches)
        outcomes.append({'featureId':feature_id,'status':'staged',
                         'patchCount':len(feature_patches),'openingChecks':feature_checks})
    for patch in all_patches:
        if patch[0]=='replace':
            _,node,mesh,material_index,extras=patch
            writer.replace_node_mesh(node,mesh,material_index,extras)
        else:
            _,node,mesh,material_index,material,extras=patch
            if material_index is None:material_index=writer.material_for_name(
                getattr(material,'name','material'),material)
            writer.add_node(node,mesh,material_index,SOURCE_TO_GLB,extras)
    return outcomes


def patch_world(input_glb,profiles_path,area_root,audit_path,output_glb,report_path,
                progress=None):
    input_glb=Path(input_glb);profiles_path=Path(profiles_path);area_root=Path(area_root).resolve()
    output_glb=Path(output_glb)
    if output_glb.resolve()==input_glb.resolve():
        raise ValueError('patch output must not overwrite its input GLB')
    if output_glb.exists():
        raise FileExistsError(f'refusing to overwrite existing release/candidate: {output_glb}')
    audit=json.loads(Path(audit_path).read_text())
    if audit.get('worldGlb',{}).get('sha256')!=sha256_file(input_glb):
        raise ValueError('audit world hash does not match patch input')
    if audit.get('profileInput',{}).get('sha256')!=sha256_file(profiles_path):
        raise ValueError('audit profile hash does not match patch input')
    pdoc=json.loads(profiles_path.read_text());profiles_by_id={p['id']:p for p in pdoc['profiles']}
    sides=load_registered_sides(area_root/'regen/sides-reg')
    os.environ['SFW_AREA']=str(area_root)
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    _,frame=fc.load_world();fc.CAMS=apply_registration(fc.load_cameras(frame),area_root/'imagery/registered-cameras.json')
    fc.REGISTRATIONS={k:r for k,r in json.loads((area_root/'imagery/registered-cameras.json').read_text()).items() if r.get('accepted')}
    with GltfStream(input_glb) as stream:
        writer=GlbAppendWriter(stream)
        outcomes=_prepare_feature(stream,profiles_by_id,audit['records'],area_root,
                                  sides,fc,writer,progress=progress)
        passed=[x for x in outcomes if x['status']=='staged']
        if passed:
            writer.write(output_glb)
        else:
            output_glb=None
        summary={'schema':'source-verified-aperture-glb-patch-v1',
            'inputGlb':{'path':str(input_glb.resolve()),'sha256':sha256_file(input_glb)},
            'profiles':{'path':str(profiles_path.resolve()),'sha256':sha256_file(profiles_path)},
            'audit':{'path':str(Path(audit_path).resolve()),'sha256':sha256_file(audit_path)},
            'readOnlyInput':True,'candidateOutput':None if output_glb is None else str(output_glb.resolve()),
            'candidateSha256':None if output_glb is None else sha256_file(output_glb),
            'candidateBinaryPassThrough':True,
            'inputBinaryByteLength':stream.bin_length,
            'writerAddedBinaryBytes':writer.binary_offset-stream.bin_length,
            'featureCount':len(outcomes),'stagedFeatureCount':len(passed),
            'rolledBackFeatureCount':sum(x['status']=='rollback' for x in outcomes),
            'outcomes':outcomes}
        Path(report_path).parent.mkdir(parents=True,exist_ok=True)
        Path(report_path).write_text(json.dumps(summary,indent=2)+'\n')
        writer.close()
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--glb',required=True,type=Path)
    ap.add_argument('--profiles',required=True,type=Path)
    ap.add_argument('--area',required=True,type=Path)
    ap.add_argument('--audit',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--report',required=True,type=Path)
    a=ap.parse_args()
    result=patch_world(a.glb,a.profiles,a.area,a.audit,a.output,a.report,
                       progress=lambda message:print(message,flush=True))
    print(json.dumps({'stagedFeatures':result['stagedFeatureCount'],
                      'rolledBackFeatures':result['rolledBackFeatureCount'],
                      'candidate':result['candidateOutput']}))


if __name__=='__main__':main()
