"""External scene harness for provisional source-span gable closure.

Only process-local copies change. Production files and source evidence stay
unchanged. No proposal is a standard accepted scene build.
"""
import argparse,inspect,json,runpy,sys,types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import nearest_points
from pipeline import roof_shell,roof_source_parts
from regen.provisional_gable_ownership import source_priors,solve
from regen.check_roof_sky_projection import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('generator_args',nargs=argparse.REMAINDER);a=p.parse_args();argv=a.generator_args[1:] if a.generator_args[:1]==['--'] else a.generator_args
    raw_profiles=json.loads(Path(argv[argv.index('--regen-profiles')+1]).read_text())['profiles']
    root=Path(__file__).resolve().parent.parent;generator=root/'generate_detailed_v14.py';original=roof_shell.build;history=[];hashes={str(q.relative_to(root)):sha(q) for q in [generator,*sorted((root/'pipeline').glob('*.py'))]};a.audit.parent.mkdir(parents=True,exist_ok=True)
    def save():a.audit.write_text(json.dumps({'status':'EXTERNAL-PROVISIONAL-NOT-ACCEPTED','productionFilesEdited':False,'observationConfidenceChanged':False,'generatorArgs':argv,'supportMode':'source-span-gable-unknown-only','productionRuleHashes':hashes,'wrapperSha256':sha(__file__),'helperSha256':sha(Path(__file__).with_name('provisional_gable_ownership.py')),'records':history},indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
    def build(footprint,points,base_z,*args,**kwargs):
        context=inspect.currentframe().f_back.f_globals
        if not all(k in context for k in ('PROFILES','ROOF_OBSERVATIONS','id','c','z0')):return original(footprint,points,base_z,*args,**kwargs)
        identity=str(context['id']);source_id=str(context['c'].get('sourceId',identity));priors,rejected=source_priors(raw_profiles,identity,source_id,float(context['z0']))
        if not priors:return original(footprint,points,base_z,*args,**kwargs)
        area=roof_shell._polygonal_footprint(footprint);parts,_=roof_source_parts.resolve(context['ROOF_OBSERVATIONS'],context['PROFILES'],identity,source_id,area);protected=[p['polygon'] for p in parts];protected.extend(kwargs.get('source_owned_domains',()))
        # Do not refit source components while diagnosing host-wall ownership.
        features=[f for o in context['ROOF_OBSERVATIONS'] if o.get('side') in {p['profileId'].removeprefix('regen-') for p in priors} for f in o.get('features',[]) if f.get('confidence') in ('medium','high')]
        if features:
            history.append({'building':identity,'changed':False,'reason':'source-component-domain-present-ownership-extension-required'});save();return original(footprint,points,base_z,*args,**kwargs)
        module=types.ModuleType('pipeline._external_gable_ownership');module.__package__='pipeline';source=Path(roof_shell.__file__).read_text();needle='    constraint_endpoints = []\n';assert source.count(needle)==1
        split=[]
        for prior in priors:
            for u,z in prior['traceUZ']:
                p=prior['normal']*prior['planeV']+prior['tangent']*u
                if area.boundary.distance(Point(p))<.1:split.append(np.asarray(nearest_points(area.boundary,Point(p))[0].coords[0]))
        module.__dict__['_external_outline_mesh_points']=split
        # This isolated source copy adds topology only; it does not encode
        # source annotations as zero-uncertainty eave measurements.
        source=source.replace(needle,'    constraint_endpoints = list(_external_outline_mesh_points)\n')
        exec(compile(source,str(roof_shell.__file__)+'[external-outline-mesh]', 'exec'),module.__dict__);harmonic=module._harmonic_structural_heights;limit=module._limit_mesh_slope;state={}
        def ownership(xy,faces,regions,*hargs,**hkwargs):
            z,known,unknown,audit=harmonic(xy,faces,regions,*hargs,**hkwargs);before=z.copy();new,photo,changed,proposal=solve(xy,faces,z,known,unknown,priors,area,protected_domains=protected)
            state.update(xy=xy.copy(),before=before,expected=new.copy(),known=known.copy(),photo=photo,changed=changed,audit=proposal)
            # Preserve the native masks; source locks are handled separately by
            # the external grade projection, not relabeled measured anchors.
            return new,known,unknown,audit
        def local_limit(xy,faces,z,eligible_vertices=None,**lkwargs):
            if state and np.shape(xy)==np.shape(state['xy']) and np.allclose(xy,state['xy'],atol=1e-8):eligible_vertices=np.asarray(eligible_vertices,bool)&state['changed']&~state['photo']
            return limit(xy,faces,z,eligible_vertices=eligible_vertices,**lkwargs)
        module._harmonic_structural_heights=ownership;module._limit_mesh_slope=local_limit;result=module.build(footprint,points,base_z,*args,**kwargs);proposal=state.get('audit',{'status':'no-structural-harmonic-domain','appliedBoundaryVertices':0})
        if state:
            distance,index=cKDTree(result['surface'].vertices[:,:2]).query(state['xy']);final=result['surface'].vertices[index,2];proposal['finalMeasuredAnchorMaximumChangeM']=float(abs(final[state['known']]-state['before'][state['known']]).max()) if state['known'].any() else 0.;proposal['finalSourceBoundaryChangeFromNativeM']=float(abs(final[state['photo']]-state['before'][state['photo']]).max()) if state['photo'].any() else 0.;proposal['finalSourceOutlineMaxErrorM']=float(abs(final[state['photo']]-state['expected'][state['photo']]).max()) if state['photo'].any() else None;proposal['finalGridMatchM']=float(distance.max())
        result['audit']['externalProvisionalGableOwnership']=proposal;history.append({'building':identity,'changed':bool(proposal.get('appliedBoundaryVertices',0)),'rejections':rejected,'proposal':proposal});save();return result if proposal.get('appliedBoundaryVertices',0) else original(footprint,points,base_z,*args,**kwargs)
    roof_shell.build=build;sys.argv=[str(generator),*argv]
    try:runpy.run_path(str(generator),run_name='__main__')
    finally:
        roof_shell.build=original;save();changed=[p for p,h in hashes.items() if sha(root/p)!=h]
        if changed:raise RuntimeError('Frozen production changed during external generation: '+','.join(changed))


if __name__=='__main__':main()
