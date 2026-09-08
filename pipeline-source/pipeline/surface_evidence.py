"""Aperture admission from positive AND negative image evidence, in wall UV.
Unknown pixels are not blank-wall observations. Reject conflicting candidates whole;
never turn an occlusion boundary into a new, oddly shaped window.
"""
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

def shape(record):
 return Polygon(record['polygon']) if 'polygon' in record else box(*record['rect'])

def admissible(candidate, surface, explicit=False):
 aperture=shape(candidate)
 for region in surface.get('surfaceEvidence',[]):
  if region['label'] not in ['solid','opaque-panel','sign-carrier']:continue
  if aperture.intersection(shape(region)).area>aperture.area*.015:return False
 if surface.get('openingPolicy')=='observed-only' and not explicit:return False
 # Explicit boundaries supersede repetitions, including near-duplicate guesses.
 if not explicit:
  for op in surface.get('openings',[]):
   if aperture.intersection(shape(op)).area>aperture.area*.05:return False
 return True

def ledger(surface,candidates):
 return {'policy':surface.get('openingPolicy','bounded-repeat'),
         'negativeRegions':sum(r['label'] in ['solid','opaque-panel','sign-carrier'] for r in surface.get('surfaceEvidence',[])),
         'explicitOpenings':len(surface.get('openings',[])),
         'repeatCandidates':len(candidates),
         'rejectedRepeatCandidates':sum(not admissible(c,surface) for c in candidates),
         'independentValidation':False}
