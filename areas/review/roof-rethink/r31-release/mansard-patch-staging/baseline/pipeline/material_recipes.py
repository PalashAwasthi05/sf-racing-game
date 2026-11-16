"""Deterministic metric appearance recipes, not scans of a particular building.
Small joints and shallow relief belong to a material family; no baked lighting.
"""
import re
from urllib.parse import quote, unquote
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial

RECIPES={
 'render_salmon':{'scale':[2,2],'color':[182,123,107],'rough':.82,'noise':.8},
 'render_gray':{'scale':[2,2],'color':[163,166,161],'rough':.87,'noise':.8},
 'render_white':{'scale':[2,2],'color':[218,216,204],'rough':.87,'noise':.8},
 'timber_dark':{'scale':[2.4,1.2],'cell':[2.4,.15],'joint':.003,'color':[60,69,75],'mortar':[38,44,47],'relief':.001,'rough':.84,'variation':2},
 'timber_tan':{'scale':[2.4,1.2],'cell':[.10,1.2],'joint':.003,'color':[151,128,96],'mortar':[94,78,56],'relief':.001,'rough':.85,'variation':2},
 'brick_modular':{'scale':[1.8,.6],'cell':[.225,.075],'joint':.006,'color':[145, 70,49],'mortar':[139,128,108],'relief':.0018,'variation':5,'rough':.82},
 'brick_dark':{'scale':[1.8,.6],'cell':[.225,.075],'joint':.006,'color':[112, 65,48],'mortar':[120,112,98],'relief':.0018,'variation':4,'rough':.85},
 'terracotta_panel':{'scale':[2.4,1.2],'cell':[.6,.3],'joint':.004,'color':[174,92, 61],'mortar':[ 90,66, 50],'relief':.001,'variation':2,'rough':.67},
 'terracotta_ribbed':{'scale':[2.4,1.2],'cell':[.6,.3],'joint':.004,'color':[170,80, 50],'mortar':[ 90,61,45],'relief':.001,'variation':2,'rough':.72,'ribs':.025},
 'stone':{'scale':[2.4,1.2],'color':[199,193,177],'rough':.8,'noise':.9},
 'limestone':{'scale':[2.4,1.2],'color':[199,193,177],'rough':.8,'noise':.9},
 'cast_stone':{'scale':[2.4,1.2],'cell':[3.,1.5],'joint':.008,'color':[205,199,184],'mortar':[186,180,166],'relief':.0008,'variation':1,'rough':.86},
 'stucco':{'scale':[2.,2.],'color':[201,198,182],'rough':.9,'noise':1.3},
 'stucco_red':{'scale':[2.,2.],'color':[142,80,65],'rough':.9,'noise':1.3},
 'panel_dark':{'scale':[2.4,1.2],'cell':[1.2,.3],'joint':.003,'color':[55,62,64],'mortar':[36,41,43],'relief':.0006,'variation':1,'rough':.74},
 # An opaque surface coating determines the optical response even when the
 # panel underneath is metal. Keep the same measured/synthesized joint map.
 'panel_coated':{'scale':[2.4,1.2],'cell':[1.2,.6],'joint':.003,'color':[188,187,180],'mortar':[131,130,124],'relief':.0005,'variation':1,'rough':.67,'metal':0.},
 'panel_silver':{'scale':[2.4,1.2],'cell':[1.2,.6],'joint':.003,'color':[176,181,177],'mortar':[117,123,121],'relief':.0005,'variation':1,'rough':.37,'metal':.55},
 # Venue surfaces. Seating: one tile is 8 rows of 0.85 m treads by 14 m along the row, with a 1.1 m aisle; v runs away
 # from the field, so each row is a seat band (backs and pans) behind a concrete tread. Turf: a 2 x 2 mowing check of
 # 5 m squares. Clay: infield and track dirt.
 'seats':{'scale':[14,6.8],'pattern':'seats','rows':8,'seat':.5,'aisle':1.1,'color':[33,74,55],'tread':[148,149,144],'rough':.62,'noise':2.},
 'turf':{'scale':[10,10],'pattern':'check','color':[74,116,57],'alt':[92,138,68],'rough':.92,'noise':5.},
 'clay':{'scale':[4,4],'color':[152,96,64],'rough':.96,'noise':7.},
 # Painted and weathered structural concrete of a venue: mid grey, darker than new paving.
 'venue_concrete':{'scale':[2.4,1.2],'color':[146,147,142],'rough':.88,'noise':4.}
}

_OBSERVED_PATTERNS = {
 'brick': {'running_bond', 'stack_bond'},
 'stone': {'block_grid', 'smooth_grain'},
 'concrete': {'block_grid', 'smooth_grain'},
 'render': {'plaster_grain', 'smooth_grain'},
 'timber': {'horizontal_courses', 'vertical_boards', 'straight_shingles', 'scalloped_shingles'},
 'metal-panel': {'panel_grid', 'ribbed_panel'},
 'terracotta': {'panel_grid', 'ribbed_panel'},
 'tile': {'panel_grid', 'ribbed_panel'},
 'glass': {'smooth_grain', 'panel_grid'},
}
_FINISH_ROUGHNESS = {'matte': .88, 'satin': .58, 'glossy': .24,
                     'polished': .2, 'rough': .96, 'weathered': .9}


def make_observed_key(base, color_hex, *, family=None, pattern=None, finish=None, moduleM=None):
 """Encode optional typed surface observations in a material name.

 The legacy ``base#rrggbb`` spelling is returned unchanged when no optional
 descriptors are present. The versioned suffix is deliberately self-contained
 so scene post-processors can recover module axes without external ID tables.
 """
 color_hex = str(color_hex).lower().lstrip('#')
 if not re.fullmatch(r'[0-9a-f]{6}', color_hex):
  raise ValueError('material color must be a six-digit hex value')
 if pattern is None and finish is None and moduleM is None:
  return f'{base}#{color_hex}'
 if base not in RECIPES:
  raise ValueError(f'cannot attach observations to unknown recipe {base!r}')
 if family is not None and family not in _OBSERVED_PATTERNS:
  raise ValueError(f'unsupported observed material family {family!r}')
 if pattern is not None:
  if family not in _OBSERVED_PATTERNS or pattern not in _OBSERVED_PATTERNS[family]:
   raise ValueError(f'unsupported observed pattern {pattern!r} for family {family!r}')
 if finish is not None and finish not in _FINISH_ROUGHNESS:
  raise ValueError(f'unsupported observed finish class {finish!r}')
 modules = {}
 if moduleM is not None:
  if not isinstance(moduleM, dict) or not moduleM:
   raise ValueError('moduleM must map measured local axes x/z to positive metres')
  for axis, value in moduleM.items():
   if axis not in ('x', 'z') or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
    raise ValueError('moduleM axes must be positive finite x/z dimensions in metres')
   modules[axis] = float(value)
 if pattern in ('running_bond', 'stack_bond', 'block_grid', 'panel_grid', 'ribbed_panel'):
  if pattern in ('block_grid', 'panel_grid', 'ribbed_panel') and set(modules) != {'x', 'z'}:
   raise ValueError(f'{pattern} requires measured moduleM on both x and z axes')
  if modules and set(modules) != {'x', 'z'}:
   raise ValueError(f'{pattern} requires measured moduleM on both x and z axes')
 if pattern == 'horizontal_courses' and modules and set(modules) != {'z'}:
  raise ValueError('horizontal_courses requires only measured vertical course pitch moduleM.z')
 if pattern == 'vertical_boards' and modules and set(modules) != {'x'}:
  raise ValueError('vertical_boards requires only measured horizontal board pitch moduleM.x')
 if pattern in ('straight_shingles', 'scalloped_shingles') and modules and set(modules) != {'x', 'z'}:
  raise ValueError(f'{pattern} requires measured moduleM on both x and z axes when supplied')
 if moduleM is not None and pattern is None:
  raise ValueError('moduleM requires an explicit pattern; module alone does not identify surface structure')
 # Keep the color immediately after '#': several existing readers use that exact
 # six-character field. Observed descriptors go before it and use URL-safe values.
 suffix = ['obs1']
 if family is not None: suffix.append('fam=' + quote(str(family), safe='-'))
 if pattern is not None: suffix.append('pat=' + quote(pattern, safe='-'))
 if finish is not None: suffix.append('fin=' + quote(finish, safe='-'))
 for axis in ('x', 'z'):
  if axis in modules: suffix.append(f'm{axis}={modules[axis]:.6g}')
 return f"{base}~{'~'.join(suffix)}#{color_hex}"


def parse_observed_key(key):
 """Decode a recipe name; return None for legacy names without typed metadata."""
 name = str(key)
 recipe_part, sep, color = name.partition('#')
 if not sep or not re.fullmatch(r'[0-9a-fA-F]{6}', color):
  return None
 if '~obs1' not in recipe_part:
  return None
 base, suffix = recipe_part.split('~obs1', 1)
 if base not in RECIPES:
  return None
 fields = {}
 for item in suffix.split('~'):
  if not item: continue
  if '=' not in item: raise ValueError('malformed observed material key')
  field, value = item.split('=', 1)
  fields[unquote(field)] = unquote(value)
 family = fields.pop('fam', None)
 pattern = fields.pop('pat', None)
 finish = fields.pop('fin', None)
 modules = {}
 for axis in ('x', 'z'):
  if 'm' + axis in fields:
   modules[axis] = float(fields.pop('m' + axis))
 if fields: raise ValueError('unknown observed material key fields')
 # Revalidate decoded records with exactly the same rules as raw input.
 make_observed_key(base, color, family=family, pattern=pattern, finish=finish,
                   moduleM=modules or None)
 return {'base': base, 'color_rgb': [int(color[i:i+2], 16) for i in (0, 2, 4)],
         'family': family, 'pattern': pattern, 'finish': finish,
         'moduleM': modules or None}


def observed_recipe(key):
 """Return a generated recipe configuration for a typed variant key."""
 data = parse_observed_key(key)
 if data is None: return None
 base = RECIPES[data['base']]
 recipe = dict(base)
 recipe['color'] = data['color_rgb']
 if data['finish'] is not None: recipe['rough'] = _FINISH_ROUGHNESS[data['finish']]
 pattern, modules = data['pattern'], data['moduleM'] or {}
 if pattern is not None:
  if pattern in ('running_bond', 'stack_bond', 'block_grid', 'panel_grid', 'ribbed_panel'):
   if modules:
    old_scale = np.asarray(base['scale'], float)
    old_cell = np.asarray(base.get('cell', [.6, .3]), float)
    unit = np.asarray([modules['x'], modules['z']], float)
    counts = np.maximum(1, np.rint(old_scale / old_cell).astype(int))
    recipe['scale'] = (counts * unit).tolist()
    recipe['cell'] = unit.tolist()
  elif pattern == 'horizontal_courses':
   if 'z' in modules:
    base_cell = base.get('cell', [base['scale'][0], .15])
    count = max(1, int(round(base['scale'][1] / base_cell[1])))
    recipe['cell'] = [float(base['scale'][0]), modules['z']]
    recipe['scale'] = [float(base['scale'][0]), count * modules['z']]
   else:
    # Direction is observed, but course pitch is not. Keep the timber
    # recipe's standard pitch and make its unmeasured provenance explicit.
    recipe['cell'] = list(base.get('cell', [base['scale'][0], .15]))
    recipe['module_source'] = 'recipe_default_unmeasured'
  elif pattern == 'vertical_boards':
   template = RECIPES.get('timber_tan', base)
   base_cell = template.get('cell', [.1, template['scale'][1]])
   if 'x' in modules:
    count = max(1, int(round(template['scale'][0] / base_cell[0])))
    recipe['cell'] = [modules['x'], float(template['scale'][1])]
    recipe['scale'] = [count * modules['x'], float(template['scale'][1])]
   else:
    recipe['cell'] = list(base_cell)
    recipe['scale'] = list(template['scale'])
    recipe['module_source'] = 'recipe_default_unmeasured'
  elif pattern in ('straight_shingles', 'scalloped_shingles'):
   # One module is one shingle wide by one exposed course high. Keep an even
   # number of courses so alternating half-shingle rows close at the tile edge.
   template = RECIPES['timber_tan']
   default_cell = [.2, .15]
   if modules:
    cell = [modules['x'], modules['z']]
    columns = max(2, int(round(template['scale'][0] / cell[0])))
    rows = max(2, int(round(template['scale'][1] / cell[1])))
    if rows % 2: rows += 1
   else:
    cell = default_cell
    columns = max(2, int(round(template['scale'][0] / cell[0])))
    rows = max(2, int(round(template['scale'][1] / cell[1])))
   recipe['cell'] = cell
   recipe['scale'] = [columns * cell[0], rows * cell[1]]
   recipe['joint'] = .004
   recipe['relief'] = .001
   recipe['variation'] = 2
   recipe['module_source'] = 'observed' if modules else 'recipe_default_unmeasured'
  recipe['observed_pattern'] = pattern
  recipe['observed_moduleM'] = modules or None
  if not modules: recipe['module_source'] = 'recipe_default_unmeasured'
  if data['family'] == 'timber' and pattern in ('horizontal_courses', 'vertical_boards',
                                                  'straight_shingles', 'scalloped_shingles'):
   # Painted cladding seams follow the target paint. Never carry the source
   # timber recipe's dark mortar into a differently painted surface.
   recipe['mortar'] = (np.asarray(data['color_rgb'], float) * .72).round().astype(int).tolist()
  if pattern in ('smooth_grain', 'plaster_grain'):
   for field in ('cell', 'joint', 'mortar', 'relief', 'variation'):
    recipe.pop(field, None)
  if pattern == 'ribbed_panel': recipe['ribs'] = modules['z']
 return recipe


def selector_target(key):
 """Convert an encoded per-surface observation to the generic scan selector schema."""
 data = parse_observed_key(key)
 if data is None: return None
 families = {'brick': 'brick', 'render': 'plaster', 'stone': 'stone',
             'concrete': 'concrete', 'timber': 'wood', 'metal-panel': 'metal',
             'terracotta': 'ceramic', 'tile': 'ceramic', 'glass': 'glass'}
 patterns = {'running_bond': 'brick_running', 'stack_bond': 'stack_bond',
             'block_grid': 'block_grid', 'panel_grid': 'panel_grid',
             'ribbed_panel': 'ribbed_panel', 'horizontal_courses': 'timber_boards',
             'vertical_boards': 'timber_boards', 'smooth_grain': 'smooth_grain',
             'plaster_grain': 'plaster_grain',
             'straight_shingles': 'timber_straight_shingles',
             'scalloped_shingles': 'timber_scalloped_shingles'}
 if data.get('family') not in families or data.get('pattern') not in patterns or data.get('finish') is None:
  return {'incomplete': True}
 recipe = observed_recipe(key)
 modules = data.get('moduleM') or {}
 units = [float(modules.get('x', 1.)), float(modules.get('z', 1.))]
 open_axes = [i for i, axis in enumerate(('x', 'z')) if axis not in modules]
 return {'substrate_family': families[data['family']], 'pattern': patterns[data['pattern']],
         'finish_class': data['finish'], 'strong_weathering': False,
         'unit_size_m': units if modules else None, 'unit_open_axes': open_axes,
         'tile_size_m': list(recipe['scale']), 'old_scale': list(recipe['scale']),
         'color_rgb': list(data['color_rgb']), 'intrinsic_color': False,
         'module_source': 'observed' if modules else 'recipe_default_unmeasured'}


def _shingle_surface(x, z, *, width, course, period_x, period_z, pattern, joint, relief):
 """Evaluate periodic shingle height and joint masks in facade metres."""
 # Normalize and round away floating point residue at exact repeat boundaries.
 x = np.round(np.mod(x, period_x), 12)
 z = np.round(np.mod(z, period_z), 12)
 row0 = np.floor(z / course).astype(int)
 local_z = np.mod(z, course)

 def edge_for_row(row):
  stagger_x = x + np.mod(row, 2) * width / 2
  local_x = np.mod(stagger_x, width)
  if pattern == 'scalloped_shingles':
   # A circular segment spans each shingle's full width. Its endpoints meet
   # the course line and its center hangs down by the radius, capped to leave
   # part of the exposed course for wide shingles.
   half_width = width / 2
   depth = min(half_width, course * .9)
   radius = (half_width * half_width + depth * depth) / (2 * depth)
   lower = np.sqrt(np.maximum(radius * radius - (local_x - half_width) ** 2, 0.)) \
       - np.sqrt(radius * radius - half_width * half_width)
  else:
   lower = np.zeros_like(local_x)
  return local_x, lower

 local_x0, lower0 = edge_for_row(row0)
 row = row0 - (local_z < lower0)
 local_x, lower = edge_for_row(row)
 edge_z = row * course + lower
 next_x, next_lower = edge_for_row(row + 1)
 next_edge_z = (row + 1) * course + next_lower
 since_lower = np.maximum(z - edge_z, 0.)
 until_next = np.maximum(next_edge_z - z, 0.)
 side_distance = np.minimum(local_x, width - local_x)
 side_joint = side_distance < joint
 lower_joint = since_lower < joint
 upper_joint = until_next < joint
 seam = side_joint | lower_joint | upper_joint

 # Each new course laps over the previous one: height falls to zero at the
 # exposed lower edge, then rises smoothly over the tight joint to the shingle
 # face. The curved boundary moves with the half-width row stagger.
 ramp_z = np.clip((since_lower - joint) / max(joint, 1e-6), 0., 1.)
 ramp_x = np.clip((side_distance - joint) / max(joint, 1e-6), 0., 1.)
 ramp_z = ramp_z * ramp_z * (3 - 2 * ramp_z)
 ramp_x = ramp_x * ramp_x * (3 - 2 * ramp_x)
 height = relief * ramp_z * ramp_x
 return height, seam, side_joint, row


def _periodic_normals(height, dx, dz):
 """Central height derivatives with wraparound at both texture boundaries."""
 gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * dx)
 gz = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (2 * dz)
 normal = np.stack([-gx, -gz, np.ones_like(gx)], axis=2)
 return normal / np.linalg.norm(normal, axis=2, keepdims=True)

def create(key):
 a=RECIPES.get(key) or observed_recipe(key)
 if a is None: raise KeyError(key)
 rng=np.random.default_rng(710);n=512;w,h=a['scale'];yy,xx=np.mgrid[0:n,0:n];x=xx*w/n;y=yy*h/n
 pattern=a.get('observed_pattern')
 color=np.broadcast_to(np.array(a['color'],float),(n,n,3)).copy();relief=np.zeros((n,n));noise=rng.normal(0,a.get('noise',1.1),(n,n))
 if pattern in ('straight_shingles', 'scalloped_shingles'): noise.fill(0)
 color+=noise[:,:,None]
 if pattern in ('straight_shingles', 'scalloped_shingles'):
  cw,ch=a['cell']
  relief,seam,side_seam,row=_shingle_surface(x,y,width=cw,course=ch,
      period_x=w,period_z=h,
      pattern=pattern,joint=a['joint'],relief=.0015)
  color[seam]=a['mortar']
 elif 'cell' in a:
  cw,ch=a['cell'];row=np.floor(y/ch).astype(int)
  running=a.get('observed_pattern')=='running_bond' or (a.get('observed_pattern') is None and key.split('~obs1')[0].startswith('brick'))
  offset=(row%2)*cw/2 if running else 0
  dx=np.mod(x+offset,cw);dy=np.mod(y,ch);j=a['joint'];pattern=a.get('observed_pattern')
  if pattern=='horizontal_courses': seam=dy<j
  elif pattern=='vertical_boards': seam=dx<j
  else: seam=(dx<j)|(dy<j)
  nr=max(1,int(round(h/ch)));nc=max(1,int(round(w/cw)))
  variation=rng.normal(0,a.get('variation',2),(nr+1,nc+2))
  color+=variation[np.minimum(row,nr),np.floor((x+offset)/cw).astype(int)][:,:,None]
  color[seam]=a['mortar'];relief[~seam]=a['relief']
 if 'ribs' in a:relief+=.00035*np.sin(y/a['ribs']*2*np.pi)
 if a.get('pattern')=='seats':
  row=h/a['rows'];f=np.mod(y,row)/row;tread=f<.36;shadow=f>.94;aisle=np.mod(x,w)<a['aisle']
  gap=(np.mod(x,a['seat'])<.035)&~tread
  color[tread|aisle]=np.array(a['tread'],float)[None,:]+noise[tread|aisle][:,None]
  color[gap&~aisle]*=.55;color[shadow&~aisle]*=.35
  relief[~tread&~aisle]=.004*np.clip((f[~tread&~aisle]-.36)/.58,0,1)          # seat backs rise toward the next tread
 if a.get('pattern')=='check':
  alt=(np.floor(x/(w/2))+np.floor(y/(h/2)))%2==1;color[alt]=np.array(a['alt'],float)[None,:]+noise[alt][:,None]
 if pattern in ('straight_shingles', 'scalloped_shingles'):
  normal=_periodic_normals(relief,w/n,h/n)
 else:
  gy,gx=np.gradient(relief,h/n,w/n);normal=np.stack([-gx,-gy,np.ones_like(gx)],axis=2);normal/=np.linalg.norm(normal,axis=2,keepdims=True)
 mr=np.zeros((n,n,3),np.uint8);mr[:,:,1]=np.clip(a['rough']*255+noise,0,255);mr[:,:,2]=int(a.get('metal',0)*255)
 mat=PBRMaterial(name=key,baseColorTexture=Image.fromarray(np.clip(color,0,255).astype('uint8')),normalTexture=Image.fromarray(np.clip((normal*.5+.5)*255,0,255).astype('uint8')),metallicRoughnessTexture=Image.fromarray(mr),baseColorFactor=[255,255,255,255],roughnessFactor=1.,metallicFactor=1.)
 return mat,a['scale']
