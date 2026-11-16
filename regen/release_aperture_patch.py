"""Build a readback-verified aperture-patch release beside an immutable base.

The base GLB and world JSON are never modified. A serialized patch is kept only
when binary pass-through, material/node pose invariants, and exact source-camera
receiver rays all pass. Otherwise the release is a byte-identical copy of base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zlib
from pathlib import Path

import numpy as np

from regen.audit_aperture_repair_targets import (
    CompositeRayIndex, _ray_neighborhood_scene, audit_world,
    build_source_camera_job, resolve_exact_source,
)
from regen.glb_stream import GltfStream, bounds_zbase
from regen.host_shell_apertures import TriangleIndex
from regen.patch_apertures_glb import patch_world
from regen.prototype_host_shell_apertures import (
    _receiver_node_expected, _receiver_ray_coverage, opening_center_source,
)
from regen.reading_provenance import load_registered_sides, sha256_file
from regen.profile_apertures import audit_profile


def _hash_region(mapping, start: int, length: int) -> str:
    h = hashlib.sha256()
    end = start + length
    for offset in range(start, end, 8 << 20):
        h.update(mapping[offset:min(offset + (8 << 20), end)])
    return h.hexdigest()


def _fc_for_area(area: Path):
    os.environ['SFW_AREA'] = str(area.resolve())
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    _, frame = fc.load_world()
    reg_path = area / 'imagery/registered-cameras.json'
    fc.CAMS = apply_registration(fc.load_cameras(frame), reg_path)
    fc.REGISTRATIONS = {k: r for k, r in json.loads(reg_path.read_text()).items()
                        if r.get('accepted')}
    return fc


def _postwrite_invariants(base_glb: Path, candidate_glb: Path,
                          patch_report: dict) -> dict:
    """Check append-only GLB invariants and unchanged original node poses."""
    with GltfStream(base_glb) as base, GltfStream(candidate_glb) as cand:
        bd, cd = base.document, cand.document
        checks = {}
        checks['originalBinaryPrefixSha256'] = (
            _hash_region(base.mapping, base.bin_offset, base.bin_length) ==
            _hash_region(cand.mapping, cand.bin_offset, base.bin_length))
        checks['originalBinaryByteLength'] = base.bin_length
        checks['candidateBinaryByteLength'] = cand.bin_length
        checks['originalMeshesUnchanged'] = cd.get('meshes', [])[:len(bd.get('meshes', []))] == bd.get('meshes', [])
        checks['originalAccessorsUnchanged'] = cd.get('accessors', [])[:len(bd.get('accessors', []))] == bd.get('accessors', [])
        checks['originalBufferViewsUnchanged'] = cd.get('bufferViews', [])[:len(bd.get('bufferViews', []))] == bd.get('bufferViews', [])
        checks['originalMaterialsUnchanged'] = cd.get('materials', [])[:len(bd.get('materials', []))] == bd.get('materials', [])
        checks['imagesAndTexturesUnchanged'] = all(
            cd.get(k, []) == bd.get(k, []) for k in ('images', 'textures', 'samplers'))
        checks['originalCamerasUnchanged'] = cd.get('cameras', []) == bd.get('cameras', [])
        base_nodes = bd.get('nodes', [])
        cand_nodes = cd.get('nodes', [])
        checks['originalNodeOrderPreserved'] = len(cand_nodes) >= len(base_nodes) and all(
            a.get('name') == b.get('name') for a, b in zip(base_nodes, cand_nodes))
        allowed_mesh = set()
        for outcome in patch_report.get('outcomes', []):
            if outcome.get('status') != 'staged':
                continue
            # Patch reports also include expected host node names below when
            # available. Legacy staged receipts are resolved via added nodes.
            allowed_mesh.update(outcome.get('replacedNodes', []))
        replacement_nodes = []
        assignments_preserved = True
        for a, b in zip(base_nodes, cand_nodes):
            if a == b:
                continue
            structural_a = {k: v for k, v in a.items() if k != 'mesh'}
            structural_b = {k: v for k, v in b.items() if k != 'mesh'}
            if structural_a != structural_b:
                checks['originalNodePoseAndPropertiesPreserved'] = False
                break
            if a.get('mesh') != b.get('mesh'):
                replacement_nodes.append(a.get('name'))
                old_mesh=bd.get('meshes',[])[int(a['mesh'])]
                new_mesh=cd.get('meshes',[])[int(b['mesh'])]
                old_primitives=old_mesh.get('primitives',[])
                new_primitives=new_mesh.get('primitives',[])
                if len(old_primitives)!=len(new_primitives) or any(
                    op.get('material')!=np_.get('material')
                    for op,np_ in zip(old_primitives,new_primitives)):
                    assignments_preserved=False
        else:
            checks['originalNodePoseAndPropertiesPreserved'] = True
        checks['replacedNodeMaterialAssignmentsPreserved']=assignments_preserved
        # Only host shells named by positively supported audit conflicts may
        # change mesh assignment; every other original node stays byte-equal.
        expected_hosts = {c.get('hostNode') for r in patch_report.get('auditConflicts', [])
                          for c in r.get('conflicts', []) if c.get('hostNode')}
        checks['changedExistingNodeMeshes'] = replacement_nodes
        checks['existingMeshChangesAreSupportedHosts'] = set(replacement_nodes) <= expected_hosts
        scene_index = int(bd.get('scene', 0)); cscene = int(cd.get('scene', 0))
        broots = (bd.get('scenes') or [{}])[scene_index].get('nodes', [])
        croots = (cd.get('scenes') or [{}])[cscene].get('nodes', [])
        checks['originalSceneRootsPreserved'] = croots[:len(broots)] == broots
        checks['allInvariantsPassed'] = all(v for k, v in checks.items()
                                             if isinstance(v, bool))
        return checks


def _readback_receivers(candidate_glb: Path, profiles_path: Path, area: Path,
                        audit: dict, patch_report: dict) -> dict:
    """Recast center and interior receiver rays through the serialized GLB."""
    profiles = {p['id']: p for p in json.loads(profiles_path.read_text())['profiles']}
    sides = load_registered_sides(area / 'regen/sides-reg')
    fc = _fc_for_area(area)
    staged = {o['featureId']: o for o in patch_report.get('outcomes', [])
              if o.get('status') == 'staged'}
    audit_rows = {(r.get('featureId'), r.get('profileId')): r for r in audit.get('records', [])
                  if r.get('status') == 'supported-host-aperture-conflict'}
    rows = []; failures = []
    with GltfStream(candidate_glb) as stream:
        for feature_id, outcome in sorted(staged.items()):
            for check in outcome.get('openingChecks', []):
                pid = check.get('profileId')
                profile = profiles.get(pid)
                audit_row = audit_rows.get((feature_id, pid))
                if profile is None or audit_row is None:
                    failures.append({'featureId': feature_id, 'profileId': pid,
                                     'reason': 'staged-profile-or-audit-source-missing'})
                    continue
                opening_index = int(check['openingIndex'])
                source = resolve_exact_source(profile, area, sides)
                if source.get('status') != 'ok' or source.get('image') != audit_row.get('source', {}).get('image'):
                    failures.append({'featureId': feature_id, 'profileId': pid,
                                     'openingIndex': opening_index, 'reason': 'exact-source-provenance-changed'})
                    continue
                profile, _ = audit_profile(profile, source_hash_verified=True, fits=[])
                if not 0 <= opening_index < len(profile.get('openings', [])):
                    failures.append({'featureId': feature_id, 'profileId': pid,
                                     'openingIndex': opening_index, 'reason': 'opening-index-out-of-range'})
                    continue
                job, error = build_source_camera_job(profile, source, area, sides)
                if error:
                    failures.append({'featureId': feature_id, 'profileId': pid,
                                     'openingIndex': opening_index, 'reason': error})
                    continue
                feature_rows = stream.rows_for_feature(feature_id)
                zbase = bounds_zbase(feature_rows, feature_id)
                if zbase is None:
                    failures.append({'featureId': feature_id, 'profileId': pid,
                                     'openingIndex': opening_index, 'reason': 'feature-base-elevation-missing'})
                    continue
                ray_scene, large_rows = _ray_neighborhood_scene(
                    stream, feature_id, profile, job, zbase, stream.node_bounds,
                    [opening_index], interiors=True)
                ray_index = CompositeRayIndex(TriangleIndex.from_scene(ray_scene), stream, large_rows)
                opening = profile['openings'][opening_index]
                center = opening_center_source(profile, opening, zbase)
                camera = np.asarray(job['from'], dtype=float)
                direction = center - camera
                direction /= max(1e-9, float(np.linalg.norm(direction)))
                hits = ray_index.ray_hits(camera, camera + direction * 60.)
                center_ok = bool(hits and _receiver_node_expected(
                    hits[0]['node'], profile, opening, opening_index))
                coverage = _receiver_ray_coverage(
                    ray_scene, profile, opening, job, zbase, opening_index,
                    triangle_index=ray_index)
                ratio = coverage['expectedReceiverFirst'] / max(1, coverage['sampled'])
                before = check.get('before') or {}
                before_ratio = (before.get('expectedReceiverFirst', 0) /
                                max(1, before.get('sampled', 0)))
                receiver_coverage_non_decreasing = (
                    coverage['expectedReceiverFirst'] >= before.get('expectedReceiverFirst', 0))
                host_wall_samples_reduced = coverage['baseShellFirst'] < before.get('baseShellFirst', 0)
                action = next((c.get('proposedAction') for c in audit_row.get('conflicts', [])
                               if int(c.get('openingIndex', -1)) == opening_index), None)
                floor = .55 if action == 'cut-host-rebuild-measured-receiver' else .30
                passed = (center_ok and ratio >= floor and coverage['skyOrNoHit'] == 0
                          and coverage['sampled'] > 0 and receiver_coverage_non_decreasing
                          and host_wall_samples_reduced)
                row = {'featureId': feature_id, 'profileId': pid,
                       'openingIndex': opening_index, 'action': action,
                       'centerExpectedReceiverFirst': center_ok,
                       'centerFirstNode': hits[0]['node'] if hits else None,
                       'interiorCoverage': coverage, 'interiorRatio': round(ratio, 6),
                       'prePatchInteriorCoverage': before,
                       'prePatchInteriorRatio': round(before_ratio, 6),
                       'receiverCoverageNonDecreasing': receiver_coverage_non_decreasing,
                       'hostWallSamplesReduced': host_wall_samples_reduced,
                       'requiredRatio': floor, 'selectedSmallNodes': len(ray_scene.graph.nodes_geometry),
                       'largeCompositeNodes': len(large_rows), 'passed': passed}
                rows.append(row)
                if not passed:
                    failures.append({k: row[k] for k in ('featureId', 'profileId', 'openingIndex', 'action',
                                                          'centerExpectedReceiverFirst', 'interiorRatio',
                                                          'requiredRatio')
                                     } | {'reason': 'serialized-receiver-ray-guard-failed'})
    return {'checks': rows, 'failureCount': len(failures), 'failures': failures,
            'stagedOpeningCount': len(rows), 'passed': not failures and bool(rows)}


def _copy_world_and_build_receipt(base_dir: Path, release_dir: Path, receipt: dict):
    world_src = base_dir / 'public/world.json'
    if world_src.is_file():
        dest = release_dir / 'public/world.json'; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(world_src, dest)
        receipt['worldJson'] = {'path': str(dest), 'sha256': sha256_file(dest),
                                'copiedFromBase': True}


def _package_preview(release_dir: Path, label: str, repo_root: Path,
                     parent_manifest_path: Path | None = None):
    glb = release_dir / 'materials/candidate.glb'
    public = repo_root / 'work/king-street/public'
    compare = public / 'compare'; compare.mkdir(parents=True, exist_ok=True)
    # Hash and gzip in bounded blocks: the world GLB is larger than a gigabyte,
    # so never hold both an uncompressed copy and its compressed copy in RAM.
    digest = hashlib.sha256()
    with glb.open('rb') as src:
        for block in iter(lambda: src.read(8 << 20), b''):
            digest.update(block)
    content_hash = digest.hexdigest(); chunk_size = 4_400_000
    chunks=[]; pending=bytearray(); compressed_bytes=0
    def save_chunk(payload):
        index=len(chunks); name=f'world-{content_hash[:12]}-{index:02}.bin'
        path=public/name
        if path.exists() and _hash_file(path)==_sha256_bytes(payload):
            pass
        else:
            path.write_bytes(payload)
        chunks.append('/'+name)
    compressor=zlib.compressobj(level=6,wbits=31)
    with glb.open('rb') as src:
        for block in iter(lambda: src.read(8 << 20), b''):
            encoded=compressor.compress(block);compressed_bytes+=len(encoded);pending.extend(encoded)
            while len(pending)>=chunk_size:
                save_chunk(bytes(pending[:chunk_size]));del pending[:chunk_size]
        encoded=compressor.flush(zlib.Z_FINISH);compressed_bytes+=len(encoded);pending.extend(encoded)
    if pending:save_chunk(bytes(pending))
    manifest_path=compare/f'{label}.json'
    manifest={'compression':'gzip','contentHash':content_hash[:12],
              'chunks':chunks,'compressedBytes':compressed_bytes,
              'uncompressedBytes':glb.stat().st_size}
    public = repo_root / 'work/king-street/public'
    world_file = compare / f'{label}-world.json'
    world_src = release_dir / 'public/world.json'
    if world_src.is_file():
        world_data = json.loads(world_src.read_text())
        world_data['previewRevision'] = label
        world_file.write_text(json.dumps(world_data, separators=(',', ':')))
        manifest['world'] = f'/compare/{label}-world.json'
    # Retain the original R31's separately authored cars and overhead sidecars.
    # They describe complete-area context and remain valid for an aperture-only
    # GLB patch, so the reviewed manifest points to the same immutable URLs.
    if parent_manifest_path and parent_manifest_path.is_file():
        original = json.loads(parent_manifest_path.read_text())
        for key in ('cars', 'overhead'):
            if original.get(key):
                manifest[key] = original[key]
    manifest_path.write_text(json.dumps(manifest,separators=(',',':')))
    return {'manifest': str(manifest_path), 'world': str(world_file) if world_file.exists() else None,
            'chunks': manifest['chunks'], 'contentHash': manifest['contentHash']}


def _hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def release(base_dir: Path, profiles_path: Path, area: Path, release_dir: Path,
            audit_path: Path | None = None, feature_limit: int | None = None,
            package: bool = False, label: str | None = None,
            feature_ids: list[str] | None = None,
            parent_manifest_path: Path | None = None) -> dict:
    base_dir=base_dir.resolve();profiles_path=profiles_path.resolve();area=area.resolve()
    release_dir=release_dir.resolve()
    base_glb=base_dir/'materials/candidate.glb'
    if not base_glb.is_file():raise FileNotFoundError(base_glb)
    if release_dir.exists():raise FileExistsError(f'refusing to overwrite {release_dir}')
    release_dir.mkdir(parents=True)
    diagnostic=release_dir/'diagnostics';diagnostic.mkdir()
    audit_error=None
    if audit_path is None:
        audit_path=diagnostic/'base-aperture-audit.json'
        try:audit=audit_world(base_glb,profiles_path,area,audit_path)
        except Exception as exc:
            audit_error=f'{type(exc).__name__}: {exc}'
            audit={'records':[],'profileCount':None,'provenanceAcceptedCount':None,
                   'statusCounts':{},'affectedFeatureCount':0}
            audit_path.write_text(json.dumps({'schema':'aperture-audit-failure-v1',
                'readOnly':True,'baseGlbSha256':sha256_file(base_glb),
                'profilesSha256':sha256_file(profiles_path),'error':audit_error},indent=2)+'\n')
    else:
        try:
            audit_path=audit_path.resolve();audit=json.loads(audit_path.read_text())
            if audit.get('worldGlb',{}).get('sha256')!=sha256_file(base_glb):
                raise ValueError('supplied audit does not match exact base GLB')
            if audit.get('profileInput',{}).get('sha256')!=sha256_file(profiles_path):
                raise ValueError('supplied audit does not match exact profile input')
            shutil.copyfile(audit_path, diagnostic/'base-aperture-audit.json')
        except Exception as exc:
            audit_error=f'{type(exc).__name__}: {exc}'
            audit={'records':[],'profileCount':None,'provenanceAcceptedCount':None,
                   'statusCounts':{},'affectedFeatureCount':0}
            (diagnostic/'base-aperture-audit.json').write_text(json.dumps({
                'schema':'aperture-audit-failure-v1','readOnly':True,
                'baseGlbSha256':sha256_file(base_glb),
                'profilesSha256':sha256_file(profiles_path),'error':audit_error},indent=2)+'\n')
    if audit_error is None and (feature_limit is not None or feature_ids):
        supported=sorted({r['featureId'] for r in audit['records']
                          if r.get('status')=='supported-host-aperture-conflict'})
        keep=set(feature_ids) if feature_ids else set(supported[:feature_limit])
        unknown=keep-set(supported)
        if unknown:raise ValueError(f'feature ids have no supported conflicts in the audit: {sorted(unknown)}')
        audit={**audit,'records':[r for r in audit['records']
                                 if r.get('featureId') in keep or r.get('status')!='supported-host-aperture-conflict'],
               'testFeatureLimit':feature_limit,'testFeatureIds':sorted(keep)}
        subset_path=diagnostic/'bounded-audit-subset.json'
        subset_path.write_text(json.dumps(audit,indent=2)+'\n')
        audit_path=subset_path
    patch_candidate=diagnostic/'candidate-patched.glb'
    patch_report_path=diagnostic/'patch-report.json'
    patch_report=None;patch_error=None
    if audit_error is not None:
        patch_error='base audit unavailable: '+audit_error
    else:
        try:
            patch_report=patch_world(base_glb,profiles_path,area,audit_path,
                                     patch_candidate,patch_report_path,
                                     progress=lambda message:print(message,flush=True))
        except Exception as exc:
            patch_error=f'{type(exc).__name__}: {exc}'
    readback=None;invariants=None;use_patch=False
    if patch_report and patch_report.get('candidateOutput') and patch_candidate.is_file():
        try:
            patch_report['auditConflicts']=[r for r in audit['records']
                if r.get('status')=='supported-host-aperture-conflict']
            invariants=_postwrite_invariants(base_glb,patch_candidate,patch_report)
            readback=_readback_receivers(patch_candidate,profiles_path,area,audit,patch_report)
            use_patch=bool(invariants.get('allInvariantsPassed') and readback.get('passed'))
        except Exception as exc:
            readback={'passed':False,'failureCount':1,'failures':[{'reason':f'{type(exc).__name__}: {exc}'}]}
    final_glb=release_dir/'materials/candidate.glb';final_glb.parent.mkdir(parents=True,exist_ok=True)
    if use_patch:
        shutil.copyfile(patch_candidate,final_glb)
        final_state='patched-readback-verified'
    else:
        shutil.copyfile(base_glb,final_glb)
        final_state='immutable-base-fallback'
    receipt={'schema':'verified-aperture-release-v1','status':final_state,
        'base':{'path':str(base_glb),'sha256':sha256_file(base_glb)},
        'profiles':{'path':str(profiles_path),'sha256':sha256_file(profiles_path)},
        'baseAudit':{'path':str((diagnostic/'base-aperture-audit.json').resolve()),
                     'sha256':sha256_file(diagnostic/'base-aperture-audit.json'),
                     'profileCount':audit.get('profileCount'),
                     'provenanceAcceptedCount':audit.get('provenanceAcceptedCount'),
                     'statusCounts':audit.get('statusCounts'),
                     'affectedFeatureCount':audit.get('affectedFeatureCount')},
        'patchError':patch_error,
        'patchReport':patch_report,
        'postWriteInvariants':invariants,
        'serializedReceiverReadback':readback,
        'finalGlb':{'path':str(final_glb),'sha256':sha256_file(final_glb),
                    'byteLength':final_glb.stat().st_size,
                    'identicalToBase':sha256_file(final_glb)==sha256_file(base_glb)},
        'fallbackReason':(None if use_patch else patch_error or
            ('patcher produced no candidate because no source feature passed staging'
             if not patch_report or not patch_report.get('candidateOutput') else
             'serialized candidate failed post-write GLB or receiver-ray verification')),
        'promoted':False}
    _copy_world_and_build_receipt(base_dir,release_dir,receipt)
    receipt['preview']=None
    if package:
        if not label:label=release_dir.name
        repo_root=Path(__file__).resolve().parents[2]
        receipt['preview']=_package_preview(release_dir,label,repo_root,
            parent_manifest_path.resolve() if parent_manifest_path else None)
    receipt_path=release_dir/'release-receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    import sys
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-dir',required=True,type=Path)
    ap.add_argument('--profiles',required=True,type=Path)
    ap.add_argument('--area',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--audit',type=Path)
    ap.add_argument('--feature-limit',type=int,help='bounded diagnostic rehearsal only')
    ap.add_argument('--feature-id',action='append',help='limit diagnostic patch to this supported feature; repeatable')
    ap.add_argument('--package',action='store_true',help='write preview chunks and manifest, without promoting a menu/current pointer')
    ap.add_argument('--label')
    ap.add_argument('--parent-manifest',type=Path,
                    help='manifest whose cars/overhead sidecar URLs the reviewed preview should preserve')
    a=ap.parse_args()
    result=release(a.base_dir,a.profiles,a.area,a.output,a.audit,a.feature_limit,a.package,a.label,a.feature_id,
                   a.parent_manifest)
    print(json.dumps({'status':result['status'],'finalGlb':result['finalGlb'],
                      'postWritePassed':bool(result['serializedReceiverReadback'] and result['serializedReceiverReadback'].get('passed')),
                      'output':str(a.output.resolve())},indent=2))


if __name__=='__main__':main()
