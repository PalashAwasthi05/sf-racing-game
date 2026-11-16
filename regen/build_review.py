"""Build a source-defined review cohort with the shared world pipeline.

Run from work/. No site facts or review building identities live in this tool.
The generated local preview does not replace the area's default build.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    for field in ('area', 'profiles', 'roof-observations', 'raw', 'output'):
        ap.add_argument('--' + field, type=Path, required=True)
    ap.add_argument('--only', type=Path)
    ap.add_argument('--full-world', action='store_true',
                    help='regenerate the entire navigable area, including terrain and streets')
    ap.add_argument('--memory-gb', type=float, default=28., help='per-stage full-world memory ceiling')
    ap.add_argument('--massing', type=Path, help='archived per-building overrides, for historical comparisons only')
    ap.add_argument('--historical-reference', action='store_true',
                    help='explicitly reproduce a historical comparison that had per-building massing overrides')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    work = Path(__file__).resolve().parents[1]
    root = work.parent
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        ap.error('label must be a local filename stem')
    paths = {k: getattr(args, k).resolve() for k in ('area', 'profiles', 'roof_observations', 'raw', 'output')}
    if args.massing and not args.historical_reference:
        ap.error('per-building massing overrides are excluded from general generation; use --historical-reference only to reproduce a historical comparison')
    if args.full_world == bool(args.only):
        ap.error('choose --only <cohort> or --full-world')
    if not 1 <= args.memory_gb <= 64:
        ap.error('memory ceiling must be between 1 and 64 GB')
    if args.only: paths['only'] = args.only.resolve()
    if paths['raw'].exists() or paths['output'].exists():
        ap.error('use new output directories; existing review builds are preserved')
    only = json.loads(paths['only'].read_text()) if args.only else None
    if args.only and (not isinstance(only, list) or not only or any(not isinstance(x, str) for x in only)):
        ap.error('only must contain a nonempty JSON list of building identities')
    def run(script, *argv, cwd=work):
        command = [sys.executable, str(work / script), *map(str, argv)]
        if args.full_world and script in ('generate_detailed_v14.py', 'apply_textures_v10.py',
                                          'areas/slim_textures.py', 'apply_materials_v13.py'):
            log = paths['raw'].parent / (args.label + '.log')
            command = [sys.executable, str(work / 'watch_mem.py'), str(args.memory_gb), str(log), *command]
        print('Running', script, flush=True)
        subprocess.run(command, cwd=cwd, check=True,
                       env={**os.environ, 'SFW_AREA': str(paths['area']), 'PYTHONPATH': str(work)})
    # Take the actual bytes once, so an independently refreshed evidence file
    # cannot change this generation halfway through or invalidate its provenance.
    snapshot = paths['raw'].with_name(paths['raw'].name + '-inputs')
    snapshot.mkdir(parents=True, exist_ok=False)
    if args.massing: paths['massing'] = args.massing.resolve()
    inputs = {}
    for k in ('profiles', 'roof_observations', 'only', 'massing'):
        if k not in paths: continue
        source = paths[k]; data = source.read_bytes(); pinned = snapshot / (k + '.json')
        pinned.write_bytes(data)
        inputs[k] = {'path': str(source.relative_to(work)), 'snapshot': str(pinned.relative_to(work)),
                     'sha256': hashlib.sha256(data).hexdigest()}
        paths[k] = pinned
    if 'only' in paths: only = json.loads(paths['only'].read_text())
    rules = hashlib.sha256()
    rule_files = [work / 'generate_detailed_v14.py', work / 'apply_textures_v10.py', work / 'apply_materials_v13.py']
    rule_files += sorted((work / 'pipeline').rglob('*.py'))
    rule_files += sorted((work / 'general_world').rglob('*.py'))
    check_openings = (args.full_world and not args.historical_reference
                      and (paths['area'] / 'imagery/registered-cameras.json').exists()
                      and (paths['area'] / 'regen/sides-reg').is_dir())
    if check_openings:
        rule_files += [work / 'regen' / name for name in (
            'profile_apertures.py', 'host_shell_apertures.py', 'glb_stream.py',
            'audit_aperture_repair_targets.py', 'prototype_host_shell_apertures.py',
            'patch_apertures_glb.py', 'release_aperture_patch.py')]
    for file in rule_files:
        rules.update(str(file.relative_to(work)).encode()); rules.update(file.read_bytes())
    rules_hash = rules.hexdigest()
    cmd = ['--output', paths['raw'], '--area', paths['area'], '--regen-profiles', paths['profiles'],
           '--roof-shell', '--roof-observations', paths['roof_observations']]
    if args.only: cmd += ['--only', paths['only']]
    if args.massing:
        cmd += ['--massing', paths['massing']]
    run('generate_detailed_v14.py', *cmd)
    run('apply_textures_v10.py', '--scene', paths['raw'] / 'king-street-pilot.glb', '--output', paths['raw'] / 'textures')
    run('areas/slim_textures.py', paths['raw'] / 'textures/candidate.glb', paths['raw'] / 'textures/candidate-slim.glb')
    run('apply_materials_v13.py', '--scene', paths['raw'] / 'textures/candidate-slim.glb',
        '--observations', work / 'regen/v1/no-material-observations.json', '--output', paths['raw'] / 'materials')
    # Keep only the requested building meshes. This intentionally isolated review
    # scene is not the navigable full-world release; full-world rebuild is separate.
    if check_openings:
        run('regen/release_aperture_patch.py', '--base-dir', paths['raw'],
            '--profiles', paths['profiles'], '--area', paths['area'], '--output', paths['output'])
    else:
        (paths['output'] / 'materials').mkdir(parents=True)
        (paths['output'] / 'public').mkdir()
    mesh_count = None
    if args.full_world and not check_openings:
        shutil.copyfile(paths['raw'] / 'materials/candidate.glb', paths['output'] / 'materials/candidate.glb')
    elif not args.full_world:
        import trimesh
        from areas.ring.splice import source
        scene = trimesh.load(paths['raw'] / 'materials/candidate.glb', force='scene', process=False)
        selected = trimesh.Scene()
        identities = {identity.split('_p')[0] for identity in only}
        for node in scene.graph.nodes_geometry:
            transform, geometry = scene.graph[node]
            if node.split('__')[0] in ('buildings', 'facades', 'roofs') and source(node) in identities:
                selected.add_geometry(scene.geometry[geometry], node_name=node, geom_name=geometry, transform=transform)
        if not selected.geometry:
            raise RuntimeError('selection produced an empty review scene')
        selected.export(paths['output'] / 'materials/candidate.glb')
        mesh_count = len(selected.geometry)
    if not check_openings:
        shutil.copyfile(paths['raw'] / 'public/world.json', paths['output'] / 'public/world.json')
    # A build produced while another worker edits the rules is not reproducible.
    end_rules = hashlib.sha256()
    for file in rule_files:
        end_rules.update(str(file.relative_to(work)).encode()); end_rules.update(file.read_bytes())
    if end_rules.hexdigest() != rules_hash:
        raise RuntimeError('rules changed during generation; preserve this output for diagnosis and rebuild before preview')
    run('regen/review_area_local.py', paths['output'], args.label, cwd=root)
    if args.full_world:
        for script in ('areas/parking.py', 'areas/overhead.py'):
            run(script, paths['area'], args.label, '--glb', paths['output'] / 'materials/candidate.glb',
                '--profiles', paths['profiles'])
    (paths['output'] / 'review-build.json').write_text(json.dumps({
        'label': args.label, 'inputs': inputs, 'rulesSha256': rules_hash,
        'buildingCount': len(only) if only else len(json.loads((paths['output'] / 'public/world.json').read_text())['buildings']),
        'meshCount': mesh_count, 'visualAcceptance': 'pending',
        'fullWorldRebuilt': args.full_world, 'historicalReference': args.historical_reference,
        'openingReviewReceipt': 'release-receipt.json' if check_openings else None,
        'usesPerBuildingMassingOverrides': bool(args.massing)}, indent=2))


if __name__ == '__main__': main()
