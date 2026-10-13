"""Admission depends on measurements and provenance; names never select geometry."""
from datetime import date
import numpy as np
from shapely.geometry import Polygon,LineString

# Labels that assert a hole in a wall. Placing one wrongly puts a window where
# the building has brick, which is the defect class this project exists to stop.
OPENING_LABELS={'window','door','void'}

BLOCKED_AUTHORING_FIELDS={'match','osmWayId','featureIdOverride','displayName','referenceLine','manualOverride','templateName','buildingName','address'}

def contains_authored_override(value):
 if isinstance(value,dict):return bool(BLOCKED_AUTHORING_FIELDS.intersection(value)) or any(contains_authored_override(v) for v in value.values())
 if isinstance(value,list):return any(contains_authored_override(v) for v in value)
 return False

def inspect_record(record,source_index,policy,as_of):
 reasons=[]
 if contains_authored_override(record):reasons.append('authored-identity-or-override')
 if record.get('producer',{}).get('mode')!='automatic':reasons.append('not-machine-derived')
 if not record.get('producer',{}).get('inputHash'):reasons.append('missing-input-fingerprint')
 observations=[source_index[s] for s in record.get('sourceIds',[]) if s in source_index]
 valid=[]
 grades=[]
 for s in observations:
  if not s.get('imageSha256') or not s.get('captureDate') or not s.get('camera'):continue
  # Provider metadata is not proof that the camera agrees with this metric world.
  if s['camera'].get('worldRegistrationValidated') is not True:continue
  # Registration is a MEASURED GRADE, not a boolean. A coarse alignment places
  # an observation on the right surface and roughly along it; it cannot place a
  # window reveal. Records that assert an opening therefore need the fine grade,
  # and on the current evidence no cluster reaches it — so openings stay closed
  # by measurement instead of by the hardcoded False that closed everything.
  # A source predating the grade is treated as coarse: it was admitted under the
  # old boolean, and silently promoting it to fine would invent accuracy.
  grades.append(s['camera'].get('registrationGrade','coarse'))
  try:age=(date.fromisoformat(as_of)-date.fromisoformat(s['captureDate'])).days
  except ValueError:continue
  if 0<=age<=policy.maximum_observation_age_days:valid.append(s)
 views={s.get('viewGroup',s['imageSha256']) for s in valid}
 if len(views)<policy.minimum_independent_views:reasons.append('insufficient-independent-recent-views')
 if record.get('confidence',0)<policy.minimum_detail_confidence:reasons.append('low-confidence')
 if record.get('label') in OPENING_LABELS and grades and not any(g=='fine' for g in grades):reasons.append('opening-needs-fine-world-registration')
 if record.get('label') in OPENING_LABELS and record.get('contradictedBySolid'):reasons.append('contradicted-by-solid-evidence')
 if record.get('label')=='text':
  reads=[s.strip() for s in record.get('ocrReads',[]) if s.strip()]
  if len(reads)<2 or len(set(reads))!=1:reasons.append('text-does-not-agree-across-views')
  if not record.get('carrierSurface'):reasons.append('missing-sign-carrier')
 if record.get('label') in ['overhang','railing','stairs','sign','text'] and not record.get('support'):reasons.append('missing-physical-support')
 return {'accepted':not reasons,'reasons':reasons}

def contiguous_frames(footprint,min_length=.3):
 """One source boundary segment per frame. Never min/max across disconnected edges."""
 result=[]
 polys=[footprint] if footprint.geom_type=='Polygon' else list(footprint.geoms)
 for p in polys:
  for interior,ring in [(False,p.exterior),*[(True,r) for r in p.interiors]]:
   for a,b in zip(ring.coords,list(ring.coords)[1:]):
    a,b=np.asarray(a,float),np.asarray(b,float);v=b-a;length=np.linalg.norm(v)
    if length<min_length:continue
    u=v/length;n=np.array([u[1],-u[0]]);mid=(a+b)/2
    from shapely.geometry import Point
    if footprint.contains(Point(*(mid+n*.05))):n=-n
    result.append({'a':a.tolist(),'b':b.tolist(),'normal':n.tolist(),'lengthM':float(length),'courtyard':interior})
 return result

def validate_frame(frame,footprint,tolerance=.12):
 line=LineString([frame['a'],frame['b']]);remaining=line.difference(footprint.boundary.buffer(tolerance))
 if remaining.length>tolerance:return False
 return any(line.hausdorff_distance(LineString([e['a'],e['b']]))<=tolerance for e in contiguous_frames(footprint))

def unique_links(links):
 return sorted({tuple(sorted((a,b))) for a,b in links if a!=b})
