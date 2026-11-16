"""Run the frozen generator with process-local provisional eave composition.

No source observation/confidence or production module is edited. The isolated
candidate is explicitly experimental and must not receive a standard accepted
build manifest. Existing source component placements are held fixed.
"""
import argparse,inspect,json,runpy,sys,types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from pipeline import roof_shell,roof_source_parts,roof_assemblies
from regen.provisional_roof_eaves import source_priors,solve_unknown_apron,solve_planar_front_cap
from regen.check_roof_sky_projection import sha


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--support-mode',choices=['grid-strip','observed-frontage','planar-cap'],default='grid-strip');p.add_argument('--include-new-supported-components',action='store_true');p.add_argument('generator_args',nargs=argparse.REMAINDER);a=p.parse_args()
    argv=a.generator_args[1:] if a.generator_args[:1]==['--'] else a.generator_args
    root=Path(__file__).resolve().parent.parent;generator=root/'generate_detailed_v14.py'
    a.audit.parent.mkdir(parents=True,exist_ok=True);records={};history=[]
    build_original=roof_shell.build;compose_original=roof_source_parts.compose;resolve_original=roof_assemblies.resolve
    production_hashes={str(path.relative_to(root)):sha(path) for path in [generator,*sorted((root/'pipeline').glob('*.py')),*sorted((root/'general_world').glob('*.py'))]}
    def save():
        a.audit.write_text(json.dumps({'status':'EXTERNAL-PROVISIONAL-NOT-ACCEPTED','productionFilesEdited':False,'observationConfidenceChanged':False,
            'generatorArgs':argv,'supportMode':a.support_mode,'includeNewSupportedComponents':a.include_new_supported_components,'productionRuleHashes':production_hashes,'wrapperSha256':sha(__file__),
            'helperSha256':sha(Path(__file__).with_name('provisional_roof_eaves.py')),'records':history},indent=2,default=serial)+'\n')
    def build(footprint,points,base_z,*args,**kwargs):
        context=inspect.currentframe().f_back.f_globals
        if not all(k in context for k in ('ROOF_OBSERVATIONS','PROFILES','z0','c')):return build_original(footprint,points,base_z,*args,**kwargs)
        identity=str(context['id']);source_id=str(context['c'].get('sourceId',identity));key=(identity,source_id)
        baseline=build_original(footprint,points,base_z,*args,**kwargs);area=roof_shell._polygonal_footprint(footprint)
        priors,rejections=source_priors(context['ROOF_OBSERVATIONS'],context['PROFILES'],identity,float(context['z0']),base_z)
        if baseline['audit'].get('observedHostForm',{}).get('status','').startswith('applied-'):
            rejections.append({'reason':'already-source-owned-host-form-preserved'});priors=[]
        record={'building':identity,'sourceId':source_id,'sourcePriorRejections':rejections,'changed':False,'confidenceUpgraded':False,'baseline':baseline,'proposed':baseline,'state':{}}
        records[key]=record
        if priors:
            parts,_=roof_source_parts.resolve(context['ROOF_OBSERVATIONS'],context['PROFILES'],identity,source_id,area)
            module=types.ModuleType('pipeline._isolated_provisional_eave');module.__package__='pipeline';exec(compile(Path(roof_shell.__file__).read_text(),str(roof_shell.__file__),'exec'),module.__dict__)
            harmonic=module._harmonic_structural_heights;limit=module._limit_mesh_slope;recover=module._recover_supported_ridges;tree=cKDTree(baseline['surface'].vertices[:,:2]);baseline_z=baseline['surface'].vertices[:,2];state=record['state']
            def capture_creases(*args,**kwargs):
                regions,creases,audit=recover(*args,**kwargs);state['validatedCreases']=creases;return regions,creases,audit
            def provisional(xy,faces,regions,*args,**kwargs):
                z,known,unknown,audit=harmonic(xy,faces,regions,*args,**kwargs);distance,index=tree.query(xy)
                if distance.max()>1e-5:
                    state['audit']={'status':'unmatched-baseline-grid-abstention','maximumDistanceM':float(distance.max())};return z,known,unknown,audit
                start=baseline_z[index].copy();solved=None
                if a.support_mode=='planar-cap':
                    solved,proposal=solve_planar_front_cap(xy,faces,start,known,unknown,priors,area,state.get('validatedCreases',[]),protected_domains=[part['polygon'] for part in parts])
                if solved is not None:new,new_known,new_unknown=solved
                else:
                    new,new_known,new_unknown,proposal=solve_unknown_apron(xy,faces,start,known,unknown,priors,area,
                        protected_domains=[part['polygon'] for part in parts],support_mode='observed-frontage' if a.support_mode=='planar-cap' else a.support_mode)
                state.update(xy=xy.copy(),before=start,known=known.copy(),editable=abs(new-start)>1e-8,audit=proposal)
                return new,new_known,new_unknown,audit
            def local_limit(xy,faces,z,eligible_vertices=None,**kwargs):
                if state.get('xy') is not None and len(xy)==len(state['xy']) and np.allclose(xy,state['xy'],atol=1e-8):eligible_vertices=np.asarray(eligible_vertices,bool)&state['editable']
                return limit(xy,faces,z,eligible_vertices=eligible_vertices,**kwargs)
            module._harmonic_structural_heights=provisional;module._limit_mesh_slope=local_limit;module._recover_supported_ridges=capture_creases
            proposed=module.build(footprint,points,base_z,*args,**kwargs)
            proposal=state.get('audit',{'status':'harmonic-domain-not-called','appliedBoundaryVertices':0});record['changed']=bool(proposal.get('appliedBoundaryVertices',0));record['proposed']=proposed
            if record['changed']:
                distance,index=cKDTree(proposed['surface'].vertices[:,:2]).query(state['xy']);final=proposed['surface'].vertices[index,2]
                proposal['finalMeasuredAnchorMaximumChangeM']=float(abs(final[state['known']]-state['before'][state['known']]).max()) if state['known'].any() else 0.
                proposal['finalOutsideApronMaximumChangeM']=float(abs(final[~state['editable']]-state['before'][~state['editable']]).max()) if (~state['editable']).any() else 0.
                proposed['audit']['provisionalSourceEaveOwnership']=proposal
            result=proposed
        else:result=baseline
        entry={'building':identity,'sourceId':source_id,'changed':record['changed'],'sourcePriorRejections':rejections,
            'proposal':record['state'].get('audit',{'status':'no-admissible-source-eave-prior'}),'sourceComponentsFixedAtBaseline':True}
        record['entry']=entry;history.append(entry);save();return result
    def compose(shell,parts,neighbors=()):
        matching=next((r for r in records.values() if shell is r['proposed']),None)
        result=compose_original(shell,parts,neighbors)
        if matching and matching['changed']:matching['baselineComposed']=compose_original(matching['baseline'],parts,neighbors)
        return result
    def resolve(observations,profiles,part_id,source_id,footprint,z0,roof_samples):
        record=records.get((str(part_id),str(source_id)))
        if not record or not record['changed']:return resolve_original(observations,profiles,part_id,source_id,footprint,z0,roof_samples)
        baseline=record.get('baselineComposed',record['baseline']);fixed,audit=resolve_original(observations,profiles,part_id,source_id,footprint,z0,baseline['surface'])
        proposed_features,alternative=resolve_original(observations,profiles,part_id,source_id,footprint,z0,roof_samples)
        record['entry']['fixedComponentAudit']=audit;record['entry']['alternativeComponentAuditNotApplied']=alternative;save()
        if a.include_new_supported_components:
            def feature_key(spec):
                facing=np.asarray(spec.get('facing',[0,1]));tangent=np.array([-facing[1],facing[0]])
                return (spec['kind'],round(float(np.asarray(spec['xy'])@tangent),5),round(float(spec.get('widthM',0)),5),round(float(spec.get('bottomM',spec.get('eaveM',0))),5),round(float(spec.get('ridgeM',0)),5))
            keys={feature_key(spec) for spec in fixed};added=[];added_audit=[]
            for spec,item in zip(proposed_features,alternative['accepted']):
                if feature_key(spec) not in keys:added.append(spec);added_audit.append({**item,'provisionalExistingSourcePriorsOnly':True})
            if added:
                record['entry']['newNaturallySupportedComponents']=added_audit;save()
                return [*fixed,*added],{**audit,'status':'partial-provisional','accepted':[*audit['accepted'],*added_audit],
                    'provisionalSourceEaveOwnership':{'status':'existing-components-fixed-newly-supported-priors-added','confidenceUpgraded':False}}
        return fixed,{**audit,'provisionalSourceEaveOwnership':{'status':'baseline-source-components-preserved','alternativeSupportFitApplied':False}}
    roof_shell.build=build;roof_source_parts.compose=compose;roof_assemblies.resolve=resolve
    sys.argv=[str(generator),*argv]
    try:runpy.run_path(str(generator),run_name='__main__')
    finally:
        roof_shell.build=build_original;roof_source_parts.compose=compose_original;roof_assemblies.resolve=resolve_original
        save()
        changed=[name for name,digest in production_hashes.items() if sha(root/name)!=digest]
        if changed:raise RuntimeError('Production rules changed during external replay: '+','.join(changed))


if __name__=='__main__':main()
