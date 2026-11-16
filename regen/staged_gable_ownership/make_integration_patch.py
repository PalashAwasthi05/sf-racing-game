"""Stage a reviewable pure-call integration; never edit production files."""
from pathlib import Path
import difflib,json,hashlib
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
original=(root/'pipeline/roof_shell.py').read_text();s=original
s=s.replace('native_baseline_surface=None, source_owned_domains=()):','native_baseline_surface=None, source_owned_domains=(),\n          gable_priors=None, source_outline_points=()):')
s=s.replace('    constraint_endpoints = []\n','    constraint_endpoints = list(source_outline_points)\n')
s=s.replace('    front_ownership_audit = {"status": "not-requested"}\n','    front_ownership_audit = {"status": "not-requested"}\n    gable_ownership_audit = {"status": "not-requested"}\n    native_support_state = None\n')
needle='        structural_plane_audit["supportedOwnedFaceCount"] = int(sum('
insert='''        if gable_priors:
            from . import roof_gable_ownership
            if front_priors:
                gable_ownership_audit = {"status": "unresolved-competing-front-ownership-priors"}
            else:
                gable_z, gable_photo, gable_changed, gable_ownership_audit = roof_gable_ownership.solve(
                    xy, faces, z, structural_supported, structural_unknown,
                    gable_priors, footprint, protected_domains=source_owned_domains, cell_m=cell_m)
                if gable_ownership_audit.get('appliedBoundaryVertices', 0):
                    z = gable_z
                    front_adjustable = gable_changed & ~gable_photo & ~eave_mask
        native_support_state = {'xy':xy.copy(), 'faces':np.asarray(faces, np.int64).copy(),
                                'z':z.copy(), 'known':structural_supported.copy(),
                                'unknown':structural_unknown.copy()}
'''
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
s=s.replace('        "sourceFrontOwnership": front_ownership_audit,','        "sourceFrontOwnership": front_ownership_audit,\n        "sourceGableOwnership": gable_ownership_audit,')
s=s.replace('"boundaryChains": boundary_chains, "audit": audit}', '"boundaryChains": boundary_chains, "audit": audit,\n            "nativeSupportState": native_support_state}')
compile(s,'staged_roof_shell.py','exec');(out/'roof_shell.py').write_text(s)
patch=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/pipeline/roof_shell.py',tofile='b/pipeline/roof_shell.py'))
original=(root/'generate_detailed_v14.py').read_text();s=original
s=s.replace('roof_front_ownership\n','roof_front_ownership,roof_gable_ownership\n',1)
needle="  _native_baseline_shell=_shell\n"
insert='''  _gable_priors=[];_gable_validation=[]
  for _gable_profile in PROFILES:
   _gable_match=_gable_profile.get('match') or {}
   _gable_part=_gable_match.get('partId')
   if not (str(_gable_part)==str(id) or (_gable_part is None and
           str(_gable_match.get('featureId'))==str(c.get('sourceId',id)))):continue
   if not _gable_profile.get('gableOutline'):continue
   _gable_prior,_gable_audit=roof_gable_ownership.prepare_gable_prior(_gable_profile,z0)
   _gable_validation.append(_gable_audit)
   if _gable_prior is not None:
    if float(_gable_prior['traceUZ'][:,1].min())<base_roof-1e-8:
     _gable_validation.append({'profileId':_gable_profile['id'],'status':'unresolved',
       'reason':'source-outline-below-shell-floor-requires-separate-base-resolution'})
    else:_gable_priors.append(_gable_prior)
  if _gable_priors:
   _gable_state=_shell.get('nativeSupportState')
   if _gable_state is None:_gable_priors=[];_gable_validation.append({'status':'unresolved-no-native-support-state'})
   elif _front_priors:_gable_priors=[];_gable_validation.append({'status':'unresolved-competing-front-ownership-priors'})
   else:
    _gable_fixed,_gable_fixed_audit=roof_assemblies.resolve(
        ROOF_OBSERVATIONS,PROFILES,id,c.get('sourceId',id),roof_g,z0,_shell['surface'])
    if _gable_fixed:
     _gable_priors=[];_gable_validation.append({'status':'unresolved-existing-source-component-host-composition-required'})
    else:
     _gable_preview=roof_gable_ownership.solve(
         _gable_state['xy'],_gable_state['faces'],_gable_state['z'],
         _gable_state['known'],_gable_state['unknown'],_gable_priors,roof_g,
         protected_domains=[_part['polygon'] for _part in _source_parts],cell_m=min(.5,roof_step))
     _gable_validation.append(_gable_preview[3])
     if not _gable_preview[3].get('appliedBoundaryVertices',0):_gable_priors=[]
  _gable_mesh_points,_gable_mesh_audit=roof_gable_ownership.outline_mesh_points(_gable_priors,roof_g) if _gable_priors else ([],[])
'''
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
s=s.replace("  if _front_priors and not _shell['audit'].get('observedHostForm',{}).get('status','').startswith('applied-'):","  if (_front_priors or _gable_priors) and not _shell['audit'].get('observedHostForm',{}).get('status','').startswith('applied-'):")
s=s.replace("       source_owned_domains=[_part['polygon'] for _part in _source_parts])", "       source_owned_domains=[_part['polygon'] for _part in _source_parts],\n       gable_priors=_gable_priors,source_outline_points=_gable_mesh_points)")
s=s.replace("  _shell['audit']['sourceFrontPriorValidation']=_front_prior_validation", "  _shell['audit']['sourceFrontPriorValidation']=_front_prior_validation\n  _shell['audit']['sourceGablePriorValidation']=_gable_validation\n  _shell['audit']['sourceGableTopologyRegistration']=_gable_mesh_audit")
compile(s,'staged_generator.py','exec');(out/'generate_detailed_v14.py').write_text(s)
patch+=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/generate_detailed_v14.py',tofile='b/generate_detailed_v14.py'))
(out/'integration.patch').write_text(patch)
(out/'base-rule-hashes.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['pipeline/roof_shell.py','generate_detailed_v14.py']},indent=2))
print('staged pure helper +integration patch; production unchanged')
