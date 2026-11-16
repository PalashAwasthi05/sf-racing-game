"""Merge regenerated pier geometry into an intact neighborhood after checking ownership."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.texture import TextureVisuals

from regen.glb_stream import GltfStream, GlbAppendWriter, bounds_zbase
from regen.host_shell_apertures import GLB_TO_SOURCE


def triangle_keys(points):
    # Section generation can reorder vertices. Compare triangles in physical
    # coordinates, allowing one millimetre for GLB local-origin float rounding.
    q = np.rint(points * 1000).astype(np.int64)
    order = np.lexsort((q[:, :, 2], q[:, :, 1], q[:, :, 0]), axis=1)
    ordered = np.take_along_axis(q, order[:, :, None], axis=1)
    return np.ascontiguousarray(ordered).reshape(-1, 9).view('V72').ravel()


def faces_in_source(stream, row):
    mesh = stream.mesh(row)
    t = GLB_TO_SOURCE @ row['transform']
    return (mesh.vertices @ t[:3, :3].T + t[:3, 3])[mesh.faces]


def owned(triangles, profile, piers, zbase):
    n = np.asarray(profile['normal'], float)
    n /= np.linalg.norm(n)
    u = triangles[:, :, :2] @ np.array([-n[1], n[0]]) - profile['frameU'][0]
    width = profile['frameU'][1] - profile['frameU'][0]
    height = profile['heightRefM']
    z = triangles[:, :, 2] - zbase
    covered = np.zeros(len(triangles), dtype=bool)
    for pier in piers:
        x = pier['x'] * width
        half = pier.get('widthM', .5) / 2
        bottom, top = pier.get('y', [0, 1])
        covered |= ((u >= x-half-.025) & (u <= x+half+.025) &
                    (z >= bottom*height-.025) & (z <= top*height+.025)).all(axis=1)
    return bool(covered.all())


def joined_faces(original, keep, regenerated, take):
    """Keep the original normals and texture coordinates on every retained face."""
    vertices=[]; faces=[]; normals=[]; uvs=[]; offset=0
    textured=getattr(original.visual,'uv',None) is not None
    for mesh,mask in ((original,keep),(regenerated,take)):
        if not mask.any():continue
        indices,local=np.unique(mesh.faces[mask].ravel(),return_inverse=True)
        vertices.append(mesh.vertices[indices])
        normals.append(mesh.vertex_normals[indices])
        faces.append(local.reshape(-1,3)+offset)
        offset+=len(indices)
        if textured:
            if getattr(mesh.visual,'uv',None) is None:raise ValueError('Lost texture coordinates')
            uvs.append(mesh.visual.uv[indices])
    result=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),
                           vertex_normals=np.concatenate(normals),process=False)
    if textured:result.visual=TextureVisuals(uv=np.concatenate(uvs))
    return result


def merge(base_path, section_path, profiles_path, affected_path, output, audit_path):
    if output.exists() or output.resolve() == base_path.resolve():
        raise ValueError('Expected a new candidate GLB')
    profiles = {p['id']: p for p in json.loads(profiles_path.read_text())['profiles']}
    targets = defaultdict(set)
    for record in json.loads(affected_path.read_text()):
        targets[record['profile']].add(record['swatch'])
    records = []
    with GltfStream(base_path) as base, GltfStream(section_path) as section:
        writer = GlbAppendWriter(base)
        for pid, swatches in targets.items():
            profile = profiles[pid]
            feature = profile['match']['featureId']
            tag = '__observed_' + pid + '___'
            for row in base.rows:
                name = row['node']
                if not name.startswith('buildings__') or tag not in name: continue
                mat = name.rsplit('___', 1)[-1]
                if mat not in swatches or name not in section.by_node: continue
                part_id = name.split('__')[1]
                zbase = bounds_zbase(base.rows_for_feature(part_id), part_id)
                if zbase is None: continue
                replacement = section.by_node[name]
                old = faces_in_source(base, row)
                new = faces_in_source(section, replacement)
                a, b = triangle_keys(old), triangle_keys(new)
                keep = np.isin(a, b)
                take = ~np.isin(b, a)
                removed = old[~keep]
                added = new[take]
                record = {'profileId': pid, 'featureId': feature, 'node': name,
                          'removedTriangles': len(removed), 'addedTriangles': len(added)}
                if not len(removed) and not len(added): continue
                piers = [p for p in profile.get('piers', [])
                         if p.get('material', profile['material']) == mat]
                if not len(removed) or not owned(removed, profile, piers, zbase) or not owned(added, profile, piers, zbase):
                    record['status'] = 'held-outside-pier-change'
                else:
                    mesh = section.mesh(replacement).copy()
                    mesh.apply_transform(np.linalg.inv(row['transform']) @ replacement['transform'])
                    mesh = joined_faces(base.mesh(row),keep,mesh,take)
                    extras = base.document['meshes'][row['meshIndex']].get('extras', {})
                    writer.replace_node_mesh(name, mesh, row['material'], extras)
                    record['status'] = 'regenerated'
                records.append(record)
        if not any(r['status']=='regenerated' for r in records):
            raise RuntimeError('No source-owned replacements')
        writer.write(output)
        writer.close()
    def digest(path):
        with path.open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()
    audit_path.write_text(json.dumps({'baseSha256':digest(base_path),'sectionSha256':digest(section_path),
                                      'outputSha256':digest(output),'records':records},indent=1))
    print({s:sum(r['status']==s for r in records) for s in set(r['status'] for r in records)})


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for arg in ('base_path','section_path','profiles_path','affected_path','output','audit_path'):
        cli.add_argument(arg,type=Path)
    merge(**vars(cli.parse_args()))
