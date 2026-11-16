"""Isolated all-cohort replay of provisional source eave ownership; no production edits."""
import argparse,json,types
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon
from pipeline import roof_shell,roof_assemblies,roof_source_parts,spatial_rules
from regen.rebuild_roof_support_review import parameters,contexts
from regen.provisional_roof_eaves import source_priors,solve_unknown_apron
from regen.check_roof_sky_projection import sha


def serialize(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def close_roof(shell,base):
    mesh=shell['surface'];vertices=mesh.vertices.copy();vertices[:,2]=base
    cap=trimesh.Trimesh(vertices=vertices,faces=mesh.faces[:,::-1],process=False)
    parts=[mesh,cap]
    if shell.get('skirt') is not None:parts.append(shell['skirt'])
    closed=trimesh.util.concatenate(parts);closed.merge_vertices(digits_vertex=5)
    closed.update_faces(closed.nondegenerate_faces())
    # Zero-height native regions have coincident top/bottom triangles. Cancel
    # both sides of those zero-volume faces before assessing real holes.
    _,inverse,count=np.unique(np.sort(closed.faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    closed.update_faces(count[inverse]==1);closed.remove_unreferenced_vertices()
    edge_counts=np.bincount(closed.edges_unique_inverse)
    return {'watertight':bool(closed.is_watertight),'windingConsistent':bool(closed.is_winding_consistent),'volumeM3':float(closed.volume),
        'openBoundaryEdges':int((edge_counts==1).sum()),'nonManifoldEdges':int((edge_counts>2).sum()),'zeroHeightFacesCancelled':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('area','build','profiles','observations','jobs','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    masses=json.loads((a.build/'massing-dump.json').read_text());audits={r['buildingId']:r for r in json.loads((a.build/'roof-envelope-audit.json').read_text())};profiles=json.loads(a.profiles.read_text())['profiles'];observations=json.loads(a.observations.read_text());observations=observations.get('observations',observations);jobs=json.loads(a.jobs.read_text());rows={j['building']:i for i,j in enumerate(jobs)}
    clouds=contexts(a.area,masses,np.load(a.build/'rooftop-structure-returns.npz'));footprints=[Polygon(m['outline'][0],m['outline'][1:]) for m in masses]
    results=[]
    for mass,footprint in zip(masses,footprints):
        identity=mass['id'];row=rows[identity];base=audits[identity]['baseRoofZ'];raw=clouds[identity];raw=raw[contains_xy(footprint,raw[:,0],raw[:,1])&(raw[:,2]>mass['z0']+3)];x0,y0,x1,y1=footprint.bounds;nx=max(2,int(np.ceil((x1-x0)/1.5)));ny=max(2,int(np.ceil((y1-y0)/1.5)))
        inner=raw[contains_xy(footprint.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,x0,y1,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,x0,y1,1.5,nx,ny);boundary=raw[keep];constraints,host=parameters(observations,profiles,identity,mass['z0'])
        baseline=roof_shell.build(footprint,inner,base,boundary_points=boundary,eave_constraints=constraints,host_spec=host)
        parts,part_audit=roof_source_parts.resolve(observations,profiles,identity,identity,footprint)
        priors,rejected=source_priors(observations,profiles,identity,mass['z0'],base)
        original=audits[identity]['roofShell'].get('structuralPlaneSurface',{}).get('planes',[]);reproduced=baseline['audit'].get('structuralPlaneSurface',{}).get('planes',[]);design=np.array([[x,y,1] for x in (x0,x1) for y in (y0,y1)])
        errors=[min((float(abs(design@(np.asarray(p)-q)).max()) for q in reproduced),default=float('inf')) for p in original];max_error=max(errors,default=0.)
        if max_error>.02:rejected.append({'reason':'exported-plane-reproduction-exceeds2cm','maximumErrorM':max_error});priors=[]
        if baseline['audit'].get('observedHostForm',{}).get('status','').startswith('applied-'):
            rejected.append({'reason':'already-source-owned-host-form-preserved'});priors=[]
        state={};proposed=baseline
        if priors:
            module=types.ModuleType('pipeline._provisional_source_eave_replay');module.__package__='pipeline';exec(compile(Path(roof_shell.__file__).read_text(),str(roof_shell.__file__),'exec'),module.__dict__)
            original_harmonic=module._harmonic_structural_heights;original_limit=module._limit_mesh_slope
            tree=cKDTree(baseline['surface'].vertices[:,:2]);baseline_z=baseline['surface'].vertices[:,2]
            def provisional(xy,faces,regions,*args,**kwargs):
                z,known,unknown,audit=original_harmonic(xy,faces,regions,*args,**kwargs)
                distance,index=tree.query(xy)
                if np.max(distance)>1e-5:
                    state['proposalAudit']={'status':'unmatched-baseline-grid-abstention','maximumDistanceM':float(distance.max())};return z,known,unknown,audit
                start=baseline_z[index].copy();new,new_known,new_unknown,proposal=solve_unknown_apron(xy,faces,start,known,unknown,priors,footprint,
                    protected_domains=[part['polygon'] for part in parts])
                editable=abs(new-start)>1e-8
                state.update(xy=xy.copy(),before=start.copy(),proposed=new.copy(),known=known.copy(),editable=editable,proposalAudit=proposal)
                return new,new_known,new_unknown,audit
            def local_limit(xy,faces,z,eligible_vertices=None,**kwargs):
                if state.get('xy') is not None and len(xy)==len(state['xy']) and np.allclose(xy,state['xy'],atol=1e-8):
                    eligible_vertices=np.asarray(eligible_vertices,bool)&state['editable']
                return original_limit(xy,faces,z,eligible_vertices=eligible_vertices,**kwargs)
            module._harmonic_structural_heights=provisional;module._limit_mesh_slope=local_limit
            proposed=module.build(footprint,inner,base,boundary_points=boundary,eave_constraints=constraints,host_spec=host)
        # Reuse baseline source feature placements. Re-fitting an inset against
        # the new provisional roof must not move measured source apertures.
        baseline=roof_source_parts.compose(baseline,parts,[g for m,g in zip(masses,footprints) if m['id']!=identity])
        fixed_features,fixed_audit=roof_assemblies.resolve(observations,profiles,identity,identity,footprint,mass['z0'],baseline['surface'])
        if proposed is not baseline:proposed=roof_source_parts.compose(proposed,parts,[g for m,g in zip(masses,footprints) if m['id']!=identity])
        alternative_features,alternative_audit=roof_assemblies.resolve(observations,profiles,identity,identity,footprint,mass['z0'],proposed['surface'])
        cutters,_=roof_shell.component_cutters(fixed_features,mass['z0'])
        closure={'before':close_roof(baseline,base),'after':close_roof(proposed,base)}
        proposal=state.get('proposalAudit',{'status':'no-admissible-source-eave-prior','appliedBoundaryVertices':0})
        changed=bool(proposal.get('appliedBoundaryVertices',0))
        if state.get('xy') is not None:
            final_tree=cKDTree(proposed['surface'].vertices[:,:2]);distance,index=final_tree.query(state['xy']);final=proposed['surface'].vertices[index,2]
            reference_tree=cKDTree(baseline['surface'].vertices[:,:2]);_,reference_index=reference_tree.query(state['xy']);reference=baseline['surface'].vertices[reference_index,2]
            proposal['finalMeasuredAnchorMaximumChangeM']=float(abs(final[state['known']]-reference[state['known']]).max()) if state['known'].any() else 0.
            proposal['finalOutsideApronMaximumChangeM']=float(abs(final[~state['editable']]-reference[~state['editable']]).max()) if (~state['editable']).any() else 0.
            proposal['finalMaximumGridJoinDistanceM']=float(distance.max())
        for variant,shell in [('before',baseline),('after',proposed)]:
            surface=roof_shell.clip_component_intersections(shell['surface'],cutters);skirt=roof_shell.clip_component_intersections(shell['skirt'],cutters)
            surface.export(a.output/f'{row:02d}-{variant}-roof.ply');skirt.export(a.output/f'{row:02d}-{variant}-skirt.ply')
            scene=trimesh.Scene();scene.add_geometry(surface,geom_name='roof',node_name='roof');scene.add_geometry(skirt,geom_name='skirt',node_name='skirt');scene.export(a.output/f'{row:02d}-{variant}-roof.glb')
            (a.output/f'{row:02d}-{variant}-boundary-chains.json').write_text(json.dumps(shell.get('boundaryChains',[]),default=serialize))
        results.append({'row':row,'building':identity,'changed':changed,'sourcePriorRejections':rejected,'proposal':proposal,'reproductionMaximumPlaneErrorM':max_error,
            'closureBeforeSourceComponentCuts':closure,'fixedSourceComponents':fixed_audit,'alternativeComponentFitAuditNotApplied':alternative_audit,
            'sourceComponentWorldParametersUnchanged':True,'sourcePartAudit':part_audit,'heightBaseZ':mass['z0'],'roofBaseZ':base,
            'qualification':'All eave levels and aprons are provisional source-interval best fits. They are not verified metric surfaces, camera uncertainty remains unresolved, and source confidence was not upgraded.'})
        (a.output/'audit.json').write_text(json.dumps({'status':'external-provisional-eave-ownership-only','rows':results,'activeProductionChanges':0,
            'roofRuleSha256':sha(roof_shell.__file__),'prototypeSha256':sha(__file__),'helperSha256':sha(Path(__file__).with_name('provisional_roof_eaves.py')),
            'inputs':{k:{'path':str(getattr(a,k)),'sha256':sha(getattr(a,k))} for k in ('profiles','observations','jobs')}},indent=2,default=serialize)+'\n')
        print(json.dumps({'row':row,'changed':changed,'photoBoundaryVertices':proposal.get('appliedBoundaryVertices',0),'maximumLoweringM':proposal.get('maximumLoweringM',0.),'closure':closure['after']['watertight']}),flush=True)


if __name__=='__main__':main()
