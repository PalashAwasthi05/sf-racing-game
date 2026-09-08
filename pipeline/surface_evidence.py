"""Aperture admission from positive AND negative image evidence, in wall UV.
Unknown pixels are not blank-wall observations. Reject conflicting candidates whole;
never turn an occlusion boundary into a new, oddly shaped window.
"""
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

AUDIT=[]

def resolve_material(surface):
 """Select a recipe from observed family/finish, never geographic identity."""
 before=surface.get('material','stucco');ev=surface.get('materialEvidence')
 if not ev:return before
 if not ev.get('sourceIds'):raise ValueError('Material evidence needs a source')
 families={'brick':{'default':'brick_modular','dark':'brick_dark'},
           'terracotta':{'default':'terracotta_panel','ribbed':'terracotta_ribbed'},
           'render':{'default':'stucco','red':'stucco_red'},
           'pale-trim':{'default':'stone'},
           'metal-panel':{'default':'panel_silver','dark':'panel_dark'}}
 if ev['family'] not in families:raise ValueError('Unresolved material family: '+ev['family'])
 options=families[ev['family']];after=options.get(ev.get('finish','default'),options['default'])
 AUDIT.append({'rule':'material-family','surfaceId':surface.get('id'),'before':before,'after':after,'changed':before!=after,'sourceIds':ev['sourceIds']})
 return after

def shape(record):
 return Polygon(record['polygon']) if 'polygon' in record else box(*record['rect'])

def admissible(candidate, surface, explicit=False):
 aperture=shape(candidate)
 for region in surface.get('surfaceEvidence',[]):
  if region['label'] not in ['solid','opaque-panel','sign-carrier']:continue
  if aperture.intersection(shape(region)).area>aperture.area*.015:
   AUDIT.append({'rule':'solid-overrides-opening','surfaceId':surface.get('id'),'candidate':candidate,'evidence':region,'explicit':explicit});return False
 if surface.get('openingPolicy')=='observed-only' and not explicit:
  AUDIT.append({'rule':'unobserved-opening','surfaceId':surface.get('id'),'candidate':candidate});return False
 # Explicit boundaries supersede repetitions, including near-duplicate guesses.
 if not explicit:
  for op in surface.get('openings',[]):
   if aperture.intersection(shape(op)).area>aperture.area*.05:
    AUDIT.append({'rule':'explicit-boundary-overrides-repeat','surfaceId':surface.get('id'),'candidate':candidate,'observedOpening':op});return False
 return True

def ledger(surface,candidates):
 return {'policy':surface.get('openingPolicy','bounded-repeat'),
         'negativeRegions':sum(r['label'] in ['solid','opaque-panel','sign-carrier'] for r in surface.get('surfaceEvidence',[])),
         'explicitOpenings':len(surface.get('openings',[])),
         'repeatCandidates':len(candidates),
         'rejectedRepeatCandidates':sum(not admissible(c,surface) for c in candidates),
         'independentValidation':False}
