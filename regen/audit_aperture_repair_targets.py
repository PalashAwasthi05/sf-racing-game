"""Find source-provenanced closed apertures whose own shell blocks the view.

This is a read-only post-generator audit. It emits feature IDs for incremental
section regeneration; it does not edit a world GLB. Every camera is reconstructed
from the exact registered crop named by the compiled profile's provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import copy
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from regen.reading_provenance import (
    load_registered_sides,
    relative_path,
    select_source_view,
    sha256_file,
    reading_sha256,
)
from regen.host_shell_apertures import (
    TriangleIndex,
    opening_host_conflicts,
    opening_center_source,
    opening_slab,
    opening_outline_uv,
    segment_boxes,
    GLB_TO_SOURCE,
)
from regen.prototype_host_shell_apertures import _receiver_ray_coverage
from regen.glb_stream import GltfStream, bounds_zbase
from regen.profile_apertures import audit_profile


def _profile_nominal_frame(profile):
    frame_u=profile.get('frameU') or []
    height=profile.get('heightRefM')
    plane=profile.get('planeV')
    normal=profile.get('normal')
    if (len(frame_u)!=2 or not isinstance(height,(int,float)) or
            not isinstance(plane,(int,float)) or not isinstance(normal,(list,tuple)) or
            len(normal)!=2):
        return None
    return {'id':profile.get('id','').removeprefix('regen-'),
            'frameU':frame_u,'normal':normal,'planeV':plane,
            'lengthM':float(frame_u[1])-float(frame_u[0]),
            'heightM':float(height)}


def resolve_exact_source(profile, area_root, registered_sides):
    """Validate reading hash, source image, registered side, and metric frame."""
    prov=profile.get('readingProvenance') or {}
    if prov.get('status')!='ok' or not prov.get('accepted'):
        return {'status':'abstain','reason':'reading-provenance-not-accepted'}
    if not prov.get('registeredCropSource'):
        return {'status':'abstain','reason':'source-was-not-registered-crop'}
    src=prov.get('sourceReading') or {};path=Path(src.get('path',''))
    if not path.is_absolute():
        path=(Path(area_root)/path) if prov.get('pathBase')=='area' else (Path.cwd()/path)
    try:
        actual_hash=sha256_file(path)
    except OSError:
        return {'status':'abstain','reason':'source-reading-file-missing'}
    if actual_hash!=src.get('sha256'):
        return {'status':'abstain','reason':'source-reading-hash-mismatch'}
    try:
        reading=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return {'status':'abstain','reason':'source-reading-invalid'}
    # resultingReadingSha256 covers the calibrated/z-fixed reading passed to
    # the compiler, whereas sourceReading.sha256 intentionally covers the raw
    # source file above. They are different stages and are both recorded.
    side_id=profile.get('id','').removeprefix('regen-')
    if reading.get('id')!=side_id:
        return {'status':'abstain','reason':'source-reading-side-id-mismatch'}
    use=reading.get('use')
    if isinstance(use,bool) or not isinstance(use,int):
        return {'status':'abstain','reason':'source-reading-view-index-missing'}
    nominal=_profile_nominal_frame(profile)
    if nominal is None:
        return {'status':'abstain','reason':'compiled-profile-frame-invalid'}
    side_id=profile.get('id','').removeprefix('regen-')
    registered=registered_sides.get(side_id)
    if registered is None:
        return {'status':'abstain','reason':'registered-side-record-missing'}
    if registered.get('status')!='ok':
        return {'status':'abstain','reason':'registered-side-record-ambiguous'}
    piece=registered['piece']
    raw_views=piece.get('views') or []
    if not 0<=use<len(raw_views):
        return {'status':'abstain','reason':'registered-side-view-index-missing'}
    ppm=raw_views[use].get('ppm')
    if not isinstance(ppm,(int,float)) or ppm<=0:
        return {'status':'abstain','reason':'registered-view-ppm-missing'}
    length_delta=nominal['lengthM']-float(piece.get('lengthM',0.))
    # Side metadata stores lengthM to centimetres while profile endpoints are
    # preserved to millimetres. Accept only the subpixel rounding discrepancy;
    # exact frameU endpoints, plane, normal, and height are still checked below.
    pixel_tolerance=.5/float(ppm)
    if abs(length_delta)>pixel_tolerance:
        return {'status':'abstain','reason':'registered-frame-length-exceeds-half-pixel',
                'lengthDeltaM':round(length_delta,6),'halfPixelToleranceM':round(pixel_tolerance,6)}
    nominal_for_join=copy.copy(nominal)
    nominal_for_join['lengthM']=piece['lengthM']
    selection=select_source_view(nominal_for_join,registered_sides,use,registered_required=True)
    if selection.get('status')!='ok':
        return {'status':'abstain','reason':'registered-source-'+selection.get('reason','unresolved').replace(' ','-')}
    view=selection['view']
    if view.get('image')!=prov.get('image'):
        return {'status':'abstain','reason':'registered-image-does-not-match-provenance'}
    if view.get('rect')!=prov.get('rect') or view.get('persp')!=prov.get('persp'):
        return {'status':'abstain','reason':'registered-crop-files-do-not-match-provenance'}
    source_id='mapillary:'+str(view['image'])
    if source_id not in (profile.get('sourceIds') or []):
        return {'status':'abstain','reason':'profile-source-id-does-not-match-image'}
    nominal_source=relative_path(selection['side_path'],area_root)
    expected_side=Path(str(prov.get('sourceSide') or '')).as_posix()
    if expected_side and nominal_source!=expected_side:
        return {'status':'abstain','reason':'registered-side-file-does-not-match-provenance'}
    image_dir=selection['side_path'].parent/'img'
    crop_paths={k:image_dir/view[k] for k in ('rect','persp')}
    missing=[k for k,p in crop_paths.items() if not p.is_file()]
    if missing:
        return {'status':'abstain','reason':'registered-crop-file-missing-'+','.join(missing)}
    return {'status':'ok','sideId':side_id,'image':str(view['image']),
            'registeredState':view.get('registered'),'sourceReadingPath':relative_path(path,Path.cwd()),
            'sourceReadingSha256':actual_hash,'sourceSidePath':nominal_source,
            'resultingReadingSha256':prov.get('resultingReadingSha256'),
            'rectPath':relative_path(crop_paths['rect'],Path.cwd()),
            'rectSha256':sha256_file(crop_paths['rect']),
            'perspPath':relative_path(crop_paths['persp'],Path.cwd()),
            'perspSha256':sha256_file(crop_paths['persp']),
            'selectedViewIndex':use,'lengthDeltaM':round(length_delta,6),
            'halfPixelToleranceM':round(pixel_tolerance,6),
            'frameCheck':'exact registered side/frame/image/crops; length within half-pixel metadata rounding'}


def build_source_camera_job(profile, source, area_root, registered_sides):
    """Reconstruct the exact camera state used to create its registered crop."""
    import frontage_crops_v15 as fc
    from areas.ring.poses import side_camera_geometry, vfov_for_image
    side=registered_sides[profile['id'].removeprefix('regen-')]['piece']
    image=source['image'];camera=fc.CAMS.get(image)
    if camera is None:
        return None,'registered-camera-record-missing'
    view=next((v for v in side.get('views',[]) if str(v.get('image'))==image),None)
    if view is None or view.get('registered')!=camera.get('registered'):
        return None,'camera-placement-state-does-not-match-crop'
    if camera.get('registered') not in ('skyline','bias',False):
        return None,'camera-placement-state-unsupported'
    if camera.get('registered')=='skyline':
        # An accepted skyline registration must be traceable to its exact camera id.
        registration=fc.REGISTRATIONS.get(image)
        if not registration or not registration.get('accepted'):
            return None,'accepted-camera-registration-missing'
        camera_height=float(registration['cameraHeightM'])
    else:
        camera_height=fc.EYE_HEIGHT_M
    camera_xy=np.asarray(camera['xy'],float)
    ground=fc.ground_at(camera_xy)
    if ground is None:
        return None,'registered-camera-ground-unavailable'
    eye=np.array([camera_xy[0],camera_xy[1],float(ground)+camera_height])
    wall=side_camera_geometry(side)
    wall['groundZ']=float(side.get('groundZ',0.))
    vp=fc.view_parameters(camera_xy,wall,wall['groundZ'])
    hfov=min(100.,float(vp['hfovDeg'])*1.25)
    from PIL import Image
    persp=Path(area_root)/'regen/sides-reg/img'/source['perspPath'].split('/')[-1]
    try:
        with Image.open(persp) as im:width,height=im.size
    except (OSError,ValueError):
        return None,'registered-perspective-image-unreadable'
    vfov=vfov_for_image(hfov,width,height)
    forward=np.asarray(vp['forward'],float)
    return {'from':eye.tolist(),'at':(eye+20.*forward).tolist(),
            'width':int(width),'height':int(height),'vfovDeg':float(vfov),
            'image':image,'registeredState':camera.get('registered')},None


def _feature_scene(stream, feature_id):
    """Decode only one feature's shell and its facade detail primitives."""
    return stream.scene(stream.rows_for_feature(feature_id))


def _ray_box_intersects(origin,target,lo,hi,pad=.03):
    """Exact segment-vs-AABB slab test with a small numeric tolerance."""
    return bool(segment_boxes(origin,target,np.asarray([lo])-pad,np.asarray([hi])+pad)[0])


def _ray_neighborhood_scene(stream,feature_id,profile,job,zbase,bounds_rows,opening_indices=None,interiors=False):
    """Reference only nodes whose bounds intersect a measured opening ray."""
    from regen.host_shell_apertures import opening_center_source
    openings=profile.get('openings',[])
    chosen=list(range(len(openings)) if opening_indices is None else opening_indices)
    targets=[opening_center_source(profile,openings[i],zbase) for i in chosen]
    if not targets:return _feature_scene(stream,feature_id),[]
    camera=np.asarray(job['from'],dtype=float)
    mins=np.asarray([row[4] for row in bounds_rows]);maxs=np.asarray([row[5] for row in bounds_rows])
    node_names=[row[0] for row in bounds_rows];mask=np.zeros(len(bounds_rows),dtype=bool)
    own=np.fromiter((name.startswith(f'buildings__{feature_id}__') or
                     name.startswith(f'facades__{feature_id}__') for name in node_names),
                    dtype=bool,count=len(node_names))
    mask|=own
    for opening_index,target in zip(chosen,targets):
        delta=np.asarray(target,dtype=float)-camera;length=float(np.linalg.norm(delta))
        if length<1e-8:continue
        pad=.03
        if interiors:
            # Cover every interior sample out to the same distance as the
            # receiver visibility rays, including objects beside the centre ray.
            u,z,U,Z=opening_outline_uv(profile,openings[opening_index],zbase).bounds
            radius=np.hypot(U-u,Z-z)
            pad+=60.*min(2.,2.*radius/length)
            target=camera+delta/length*60.
        mask|=segment_boxes(camera,target,mins-pad,maxs+pad)
    selected=[];large=[]
    for i,(node,mesh_index,primitive_index,transform,lo,hi) in enumerate(bounds_rows):
        # The city terrain mesh has a huge bounding box and is handled by the
        # local ground-grid line check below, rather than copied into each ray
        # neighborhood.
        if node.startswith('terrain'):
            continue
        if not mask[i]:continue
        row=stream.by_node[node]
        vertex_count=int(stream.document['accessors'][row['primitive']['attributes']['POSITION']]['count'])
        if not (node.startswith(f'buildings__{feature_id}___') or
                node.startswith(f'facades__{feature_id}__')) and vertex_count>250000:
            large.append(row);continue
        selected.append(row)
    return stream.scene(selected),large


class CompositeRayIndex:
    """Small decoded neighbors plus lazy exact tests for large mesh chunks."""
    def __init__(self,small,stream,large_rows):
        self.small=small;self.stream=stream;self.large_rows=large_rows
    def ray_hits(self,origin,target,limit=None):
        hits=self.small.ray_hits(origin,target,limit=limit)
        for row in self.large_rows:
            hits.extend(self.stream.segment_mesh_hits(row,origin,target))
        hits.sort(key=lambda hit:hit['distanceM'])
        unique=[]
        for hit in hits:
            if any(hit['node']==old['node'] and abs(hit['distanceM']-old['distanceM'])<.015
                   for old in unique):continue
            unique.append(hit)
        return unique


def _source_mesh_for_node(stream,node):
    import trimesh
    row=stream.by_node.get(node)
    if row is None:return None
    mesh=stream.mesh(row);transform=row['transform']
    world=np.asarray(mesh.vertices)@transform[:3,:3].T+transform[:3,3]
    source=world@GLB_TO_SOURCE[:3,:3].T+GLB_TO_SOURCE[:3,3]
    return trimesh.Trimesh(vertices=source,faces=np.asarray(mesh.faces),process=False)


def _segment_clears_ground(camera,target,fc,samples=32):
    """Reject rays that meet the available terrain grid before the opening."""
    camera=np.asarray(camera,dtype=float);target=np.asarray(target,dtype=float)
    for t in np.linspace(.02,.98,samples):
        point=camera+(target-camera)*t
        ground=fc.ground_at(point[:2])
        if ground is None:return False
        if point[2]<=float(ground)+.025:return False
    return True


def audit_world(glb_path, profiles_path, area_root, output_path, id_path=None):
    area_root=Path(area_root).resolve()
    profile_doc=json.loads(Path(profiles_path).read_text())
    profiles=profile_doc['profiles']
    sides=load_registered_sides(area_root/'regen/sides-reg')
    # Camera registry and panorama metadata are loaded once, and registration is
    # applied once exactly as the crop generator does.
    os.environ['SFW_AREA']=str(area_root)
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    world,frame=fc.load_world()
    nominal_cameras=fc.load_cameras(frame)
    reg_path=area_root/'imagery/registered-cameras.json'
    fc.REGISTRATIONS={k:r for k,r in json.loads(reg_path.read_text()).items() if r.get('accepted')}
    fc.CAMS=apply_registration(nominal_cameras,reg_path)

    # One GLB load. Only one feature-sized scene and triangle index are built at
    # a time, avoiding whole-world triangle arrays and repeated GLB copies.
    stream=GltfStream(glb_path)
    bounds_rows=stream.node_bounds
    grouped=defaultdict(list)
    records=[];abstentions=Counter();accepted_profiles=0;camera_states=Counter()
    source_by_profile={}
    for p in profiles:
        source=resolve_exact_source(p,area_root,sides)
        if source['status']!='ok':
            rec={'profileId':p.get('id'),'featureId':(p.get('match') or {}).get('featureId'),
                 'status':'abstain','reason':source['reason'],
                 'openingCount':len(p.get('openings') or [])}
            abstentions[source['reason']]+=1;records.append(rec);continue
        accepted_profiles+=1;camera_states[str(source['registeredState'])]+=1
        p, aperture_meta = audit_profile(p, source_hash_verified=True, fits=[])
        p['_apertureAuditMeta'] = aperture_meta
        source_by_profile[p['id']]=source
        if not p.get('openings'):
            records.append({'profileId':p.get('id'),'featureId':(p.get('match') or {}).get('featureId'),
                'status':'clear-no-closed-apertures','openingCount':0,
                'sourceImage':source['image'],'registeredState':source['registeredState']})
            continue
        fid=str((p.get('match') or {}).get('featureId') or '')
        grouped[fid].append(p)
    total_features=len(grouped)
    affected={};feature_mesh_nodes=Counter()
    for feature_index,(feature_id,feature_profiles) in enumerate(grouped.items()):
        if not feature_id:
            for p in feature_profiles:
                abstentions['profile-feature-id-missing']+=1
                records.append({'profileId':p.get('id'),'status':'abstain','reason':'profile-feature-id-missing'})
            continue
        feature=_feature_scene(stream,feature_id)
        base_nodes=[n for n in feature.graph.nodes_geometry
                    if n.startswith(f'buildings__{feature_id}___') and '__observed_' not in n]
        feature_mesh_nodes[feature_id]=len(base_nodes)
        if not base_nodes:
            for p in feature_profiles:
                abstentions['feature-base-shell-missing']+=1
                records.append({'profileId':p['id'],'featureId':feature_id,
                                'status':'abstain','reason':'feature-base-shell-missing'})
            continue
        zbase=bounds_zbase(stream.rows_for_feature(feature_id),feature_id)
        if zbase is None:
            for p in feature_profiles:
                abstentions['feature-base-elevation-missing']+=1
                records.append({'profileId':p['id'],'featureId':feature_id,
                                'status':'abstain','reason':'feature-base-elevation-missing'})
            continue
        feature_triangle_index=TriangleIndex.from_scene(feature)
        for profile_index,profile in enumerate(feature_profiles):
            rec={'profileId':profile.get('id'),'featureId':feature_id,
                 'openingCount':len(profile.get('openings') or [])}
            source=source_by_profile[profile['id']]
            rec['source']={k:source[k] for k in ('image','registeredState','sourceReadingPath',
                       'sourceReadingSha256','sourceSidePath','rectPath','rectSha256',
                       'perspPath','perspSha256','selectedViewIndex','lengthDeltaM',
                       'halfPixelToleranceM','resultingReadingSha256')}
            job,error=build_source_camera_job(profile,source,area_root,sides)
            if error:
                rec.update(status='abstain',reason=error)
                abstentions[error]+=1;records.append(rec);continue
            clear_indices=[];terrain_uncertain=[]
            for oi,opening in enumerate(profile.get('openings') or []):
                target=opening_center_source(profile,opening,zbase)
                if _segment_clears_ground(job['from'],target,fc):clear_indices.append(oi)
                else:terrain_uncertain.append(oi)
            rec['terrainAbstainedOpeningIndices']=terrain_uncertain
            potential=opening_host_conflicts(feature,profile,job,zbase,index=clear_indices,
                                               triangle_index=feature_triangle_index)
            if not potential:
                rec.update(status=('abstain-terrain-ray' if terrain_uncertain else 'clear'),camera={'image':job['image'],
                           'registeredState':job['registeredState'],'fromSource':job['from'],
                           'atSource':job['at'],'vfovDeg':job['vfovDeg'],
                           'imageSizePx':[job['width'],job['height']]})
                if terrain_uncertain:abstentions['terrain-may-occlude-opening-ray']+=len(terrain_uncertain)
                records.append(rec);continue
            potential_indices=[c['openingIndex'] for c in potential]
            ray_scene,large_neighbor_rows=_ray_neighborhood_scene(
                stream,feature_id,profile,job,zbase,bounds_rows,potential_indices)
            triangle_index=CompositeRayIndex(TriangleIndex.from_scene(ray_scene),
                                             stream,large_neighbor_rows)
            conflicts=opening_host_conflicts(ray_scene,profile,job,zbase,
                index=potential_indices,triangle_index=triangle_index)
            if not conflicts:
                rec.update({'status':'clear-blocked-by-other-world-geometry','camera':{
                    'image':job['image'],'registeredState':job['registeredState'],
                    'fromSource':job['from'],'atSource':job['at'],
                    'vfovDeg':job['vfovDeg'],'imageSizePx':[job['width'],job['height']]},
                    'potentialFeatureHostOpeningIndices':potential_indices})
                records.append(rec);del triangle_index,ray_scene;continue
            conflict_rows=[]
            unbounded_indices=[];unproven_receivers=[]
            for conflict in conflicts:
                op=profile['openings'][conflict['openingIndex']]
                host_mesh=_source_mesh_for_node(stream,conflict['firstHit']['node'])
                slab=(opening_slab(host_mesh,profile,conflict,zbase)
                      if host_mesh is not None else None)
                if slab is None:
                    unbounded_indices.append(conflict['openingIndex']);continue
                coverage=_receiver_ray_coverage(feature,profile,op,job,zbase,
                                                conflict['openingIndex'],
                                                triangle_index=feature_triangle_index)
                ratio=coverage['expectedReceiverFirst']/max(1,coverage['sampled'])
                grouped_opening = op.get('_receiverPolicy') == 'preserve-existing-only'
                if grouped_opening and ratio < .30:
                    unproven_receivers.append(conflict['openingIndex'])
                    abstentions['group-native-receiver-not-proven'] += 1
                    continue
                action=('cut-host-preserve-receiver' if grouped_opening or ratio>=.30
                        else 'cut-host-rebuild-measured-receiver')
                conflict_rows.append({'openingIndex':conflict['openingIndex'],
                    'kind':(op.get('style') or {}).get('kind'),
                    'hostNode':conflict['firstHit']['node'],
                    'hostPlaneOffsetM':round(float(conflict['hitPlaneV']-profile['planeV']),5),
                    'supportedHostSlabV':[round(float(slab[0]),5),round(float(slab[1]),5)],
                    'normalAlignment':round(float(conflict['normalAlignment']),5),
                    'receiverCoverage':coverage,'proposedAction':action,
                    **({'groupIndex':op.get('_auditGroupIndex'),
                        'cell':op.get('_auditCell'),
                        'receiverPolicy':'preserve-existing-only'} if grouped_opening else {})})
            if unbounded_indices:abstentions['host-slab-not-supported-near-profile-plane']+=len(unbounded_indices)
            if unproven_receivers:rec['unprovenReceiverOpeningIndices']=unproven_receivers
            if not conflict_rows and unproven_receivers:
                rec.update(status='abstain-unproven-window-receiver',
                           unboundedHostOpeningIndices=unbounded_indices)
                records.append(rec);del triangle_index,ray_scene;continue
            if not conflict_rows:
                rec.update({'status':'abstain-unbounded-host-slab',
                    'reason':'camera ray intersects feature geometry outside a supported nearby host slab',
                    'unboundedHostOpeningIndices':unbounded_indices})
                records.append(rec);del triangle_index,ray_scene;continue
            if terrain_uncertain:abstentions['terrain-may-occlude-opening-ray']+=len(terrain_uncertain)
            rec.update(status='supported-host-aperture-conflict',camera={
                'image':job['image'],'registeredState':job['registeredState'],
                'fromSource':job['from'],'atSource':job['at'],'vfovDeg':job['vfovDeg'],
                'imageSizePx':[job['width'],job['height']]},
                conflicts=conflict_rows)
            affected.setdefault(feature_id,[]).append(profile['id'])
            records.append(rec)
            del triangle_index,ray_scene
            if profile_index and profile_index%25==0:
                print(f"aperture-audit {profile_index}/{len(feature_profiles)} profiles on {feature_id}",flush=True)
        del feature
        if feature_index%10==0:
            print(f"aperture-audit feature groups {feature_index}/{total_features}; profiles {len(records)}/{len(profiles)}",flush=True)
    stream.close()
    output={'schema':'closed-aperture-owner-audit-v1',
        'readOnly':True,'worldGlb':{'path':str(Path(glb_path).resolve()),
        'sha256':sha256_file(glb_path)},
        'profileInput':{'path':str(Path(profiles_path).resolve()),
        'sha256':sha256_file(profiles_path)},'registeredSideCount':len(sides),
        'cameraRegistration':{'path':relative_path(reg_path,Path.cwd()),
                              'sha256':sha256_file(reg_path),
                              'application':'accepted placement applied once; crop marker checked'},
        'profileCount':len(profiles),'provenanceAcceptedCount':accepted_profiles,
        'cameraStateCounts':dict(camera_states),
        'statusCounts':dict(Counter(r['status'] for r in records)),
        'abstentionReasons':dict(abstentions),'affectedFeatureCount':len(affected),
        'affectedFeatureIds':sorted(affected),
        'affectedProfiles':{k:sorted(v) for k,v in sorted(affected.items())},
        'featureShellNodeCounts':dict(feature_mesh_nodes),'records':records}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(output,indent=2)+'\n')
    if id_path:
        Path(id_path).parent.mkdir(parents=True,exist_ok=True)
        Path(id_path).write_text('\n'.join(sorted(affected))+('\n' if affected else ''))
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--glb',required=True,type=Path)
    ap.add_argument('--profiles',required=True,type=Path)
    ap.add_argument('--area',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--affected-ids',type=Path)
    a=ap.parse_args()
    result=audit_world(a.glb,a.profiles,a.area,a.output,a.affected_ids)
    print(json.dumps({'profiles':result['profileCount'],
        'provenanceAccepted':result['provenanceAcceptedCount'],
        'statusCounts':result['statusCounts'],
        'abstentionReasons':result['abstentionReasons'],
        'affectedFeatures':result['affectedFeatureCount']}))


if __name__=='__main__':main()
