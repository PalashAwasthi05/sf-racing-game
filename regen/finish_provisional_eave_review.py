"""Texture/filter an external provisional scene without publishing or accepting it."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import trimesh
from areas.ring.splice import source
from regen.check_roof_sky_projection import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','output','audit','only'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--label',required=True);a=p.parse_args();work=Path(__file__).resolve().parent.parent
    if a.output.exists():raise ValueError('Preserve existing external output; choose a new path')
    def run(script,*args):
        subprocess.run([sys.executable,str(work/script),*map(str,args)],check=True,cwd=work,env={**os.environ,'PYTHONPATH':str(work)})
    run('apply_textures_v10.py','--scene',a.raw/'king-street-pilot.glb','--output',a.raw/'textures')
    run('areas/slim_textures.py',a.raw/'textures/candidate.glb',a.raw/'textures/candidate-slim.glb')
    run('apply_materials_v13.py','--scene',a.raw/'textures/candidate-slim.glb','--observations',work/'regen/v1/no-material-observations.json','--output',a.raw/'materials')
    scene=trimesh.load(a.raw/'materials/candidate.glb',force='scene',process=False);selected=trimesh.Scene();identities={i.split('_p')[0] for i in json.loads(a.only.read_text())}
    for node in scene.graph.nodes_geometry:
        transform,geometry=scene.graph[node]
        if node.split('__')[0] in ('buildings','facades','roofs') and source(node) in identities:selected.add_geometry(scene.geometry[geometry],node_name=node,geom_name=geometry,transform=transform)
    if not selected.geometry:raise ValueError('Empty provisional cohort')
    (a.output/'materials').mkdir(parents=True);(a.output/'public').mkdir();selected.export(a.output/'materials/candidate.glb');shutil.copyfile(a.raw/'public/world.json',a.output/'public/world.json')
    audit=json.loads(a.audit.read_text())
    (a.output/'review-build.json').write_text(json.dumps({'label':a.label,'status':'EXTERNAL-PROVISIONAL-NOT-ACCEPTED','visualAcceptance':'pending-source-comparison',
        'buildingCount':len(identities),'meshCount':len(selected.geometry),'fullWorldRebuilt':False,'productionFilesEdited':False,
        'observationConfidenceChanged':False,'wrapperAuditPath':str(a.audit),'wrapperAuditSha256':sha(a.audit),
        'sourceInputs':audit['generatorArgs'],'productionRuleHashes':audit['productionRuleHashes'],'supportMode':audit['supportMode'],
        'candidateSha256':sha(a.output/'materials/candidate.glb'),'qualification':'Provisional source-eave interval composition with frozen measured/source components. Camera placement and unseen shape remain uncertain.'},indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'meshes':len(selected.geometry),'sha256':sha(a.output/'materials/candidate.glb')}))


if __name__=='__main__':main()
