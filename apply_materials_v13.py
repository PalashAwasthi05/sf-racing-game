"""Apply external surface observations uniformly; no location-specific rules."""
import argparse, collections, hashlib, json, time
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from general_world.scanned_surface_variant import variant
from general_world.material_synthesis import synthesize, structure_key


def fingerprint(material):
    # A material identity joins external observations; it never chooses a rule.
    h=hashlib.sha256()
    for value in (material.baseColorFactor,material.roughnessFactor,material.metallicFactor):
        h.update(str(value).encode())
    for image in (material.baseColorTexture,material.normalTexture,material.metallicRoughnessTexture):
        if image is not None: h.update(str(image.size).encode());h.update(image.tobytes())
    return h.hexdigest()

def apply(scene_path, observations_path, output):
    start=time.monotonic();scene=trimesh.load(scene_path,force='scene',process=False)
    observations=json.loads(Path(observations_path).read_text())['observations']
    by_key={r['sourceMaterialFingerprint']:r for r in observations};cache={};fingerprints={};rows=[];counts=collections.Counter(); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,mesh in scene.geometry.items():
        source=mesh.visual.material;mid=id(source)
        if mid not in fingerprints:fingerprints[mid]=fingerprint(source)
        obs=by_key.get(fingerprints[mid])
        if obs is None: counts['retainedWithoutNewObservation']+=1;continue
        if mesh.visual.uv is None:counts['retainedMissingUV']+=1;continue
        key=obs['sourceMaterialFingerprint']
        if key not in cache:
            material=None;provenance={}
            if obs.get('scanCandidate'):
                a=obs['scanCandidate']
                try:
                    images=[Image.open(a[k]) for k in ['color','normal','roughness']]
                    material,evidence=variant(*images,a['tile_size_m'],obs['surface'])
                    identity=hashlib.sha256(b''.join(im.tobytes() for im in images)+str(evidence['tile_size_m']).encode()).hexdigest()[:20]
                    pigment=hashlib.sha256(str(obs['surface']['color_rgb']).encode()).hexdigest()[:8]
                    material.name='scan_'+identity+'_'+pigment
                    provenance.update(evidence,structureKey='scan_'+identity,variantKey=pigment,assetLicense=a['license'],assetSource=a['source_url'])
                except ValueError as exc: provenance['scanRejectedReason']=str(exc)
            if material is None:
                material,evidence=synthesize(obs['surface']);provenance.update(evidence)
            folder=output/'materials'/material.name;folder.mkdir(parents=True,exist_ok=True)
            for fn,img in [('basecolor.png',material.baseColorTexture),('normal.png',material.normalTexture),('roughness-metallic.png',material.metallicRoughnessTexture)]:img.save(folder/fn)
            cache[key]=(material,provenance)
        material,provenance=cache[key]
        old_scale=np.asarray(obs['sourceRepeatM'],float);new_scale=np.asarray(provenance['tile_size_m'])
        if old_scale.shape!=(2,) or not np.isfinite(old_scale).all() or (old_scale<=0).any():raise ValueError('Invalid source UV scale')
        mesh.visual.uv=mesh.visual.uv*old_scale/new_scale
        mesh.visual.material=material
        rows.append({'mesh':name,'sourceFingerprint':key,'structureKey':provenance['structureKey'],'material':material.name,'observationProvenance':obs['provenance'],'materialProvenance':provenance})
        counts['updated']+=1
    path=output/'candidate.glb';scene.export(path)
    before=trimesh.load(scene_path,force='scene',process=False);after=trimesh.load(path,force='scene',process=False)
    assert set(before.geometry)==set(after.geometry)
    for key,m in before.geometry.items():
        n=after.geometry[key];assert np.array_equal(m.vertices,n.vertices) and np.array_equal(m.faces,n.faces),key
        assert n.visual.uv is None or np.isfinite(n.visual.uv).all(),key
    assert before.graph.to_flattened().keys()==after.graph.to_flattened().keys()
    for key,entry in before.graph.to_flattened().items():assert np.array_equal(entry['transform'],after.graph.to_flattened()[key]['transform'])
    report={'revision':'0.13','counts':dict(counts),'newStructureAssets':len({p[1]['structureKey'] for p in cache.values() if not p[1]['structureKey'].startswith('scan_')}),'reusedScanStructures':len({p[1]['structureKey'] for p in cache.values() if p[1]['structureKey'].startswith('scan_')}),'colorVariants':len(cache),'meshesGeometryPreserved':len(before.geometry),'imageVerifiedAssignment':False,'inferenceStatus':'historical surface descriptors, not photo-derived classification','assetMethod':'measured existing scan reuse where compatible; deterministic metric PBR synthesis for missing structures','records':rows,'elapsedSeconds':round(time.monotonic()-start,2),'inputSha256':hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),'outputSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'observationSha256':hashlib.sha256(Path(observations_path).read_bytes()).hexdigest(),'ruleSha256':hashlib.sha256(b''.join((Path(__file__).parent/'general_world'/f).read_bytes() for f in ['material_synthesis.py','material_periodicity.py','scanned_surface_variant.py','texture_selection.py'])).hexdigest()}
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--observations',required=True);p.add_argument('--output',required=True);a=p.parse_args();apply(a.scene,a.observations,a.output)
