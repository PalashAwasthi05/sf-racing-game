"""Deterministic metric PBR surfaces from explicit, location-free observations.
Structure keys exclude pigment. Synthesis supplies missing material assets, never
assigns substrate or invents a facade region. All maps share one physical frame.
"""
import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from trimesh.visual.material import PBRMaterial

PATTERNS = {'brick_running', 'panel_grid', 'ribbed_panel', 'timber_boards',
            'stone_grain', 'bitumen_grain', 'metal_grain', 'plaster_grain', 'smooth_grain'}
STRUCTURE_FIELDS = ('substrate_family', 'pattern', 'finish_class', 'tile_size_m',
                    'unit_size_m', 'joint_width_m', 'relief_m', 'rib_pitch_m',
                    'grain_direction', 'roughness', 'metallic', 'strong_weathering')

def structure_key(observation):
    return hashlib.sha256(json.dumps({k: observation.get(k) for k in STRUCTURE_FIELDS},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]

def validate(o):
    if o.get('pattern') not in PATTERNS or not o.get('substrate_family'):
        raise ValueError('unsupported or missing material structure')
    scale = np.asarray(o.get('tile_size_m', []), float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('positive metric repeat required')
    for field in ['color_rgb', 'joint_rgb']:
        color = np.asarray(o.get(field, o.get('color_rgb', [])), float)
        if color.shape != (3,) or not np.isfinite(color).all() or ((color < 0) | (color > 255)).any():
            raise ValueError('finite observed color required')
    if o['pattern'] in {'brick_running', 'panel_grid', 'ribbed_panel', 'timber_boards'}:
        unit = np.asarray(o.get('unit_size_m', []), float)
        if unit.shape != (2,) or not np.isfinite(unit).all() or (unit <= 0).any():
            raise ValueError('explicit module dimensions required')
        cells = scale / unit
        if not np.allclose(cells, np.round(cells), atol=1e-5):
            raise ValueError('tile must contain whole modules')
        if o['pattern'] == 'brick_running' and int(round(cells[1])) % 2:
            raise ValueError('running bond requires an even row count')
        joint = o.get('joint_width_m', 0)
        if not np.isfinite(joint) or not 0 < joint < min(unit) * .25:
            raise ValueError('invalid joint width')
    for field, default in [('roughness', .8), ('metallic', 0)]:
        value = o.get(field, default)
        if not np.isfinite(value) or not 0 <= value <= 1: raise ValueError('invalid PBR factor')
    if not np.isfinite(o.get('relief_m', 0)) or not 0 <= o.get('relief_m', 0) <= .02:
        raise ValueError('relief must remain shallow; deep relief requires geometry')
    if o['pattern'] == 'ribbed_panel':
        pitch = o.get('rib_pitch_m', 0)
        if not np.isfinite(pitch) or pitch <= 0 or not np.isclose(scale[1]/pitch, round(scale[1]/pitch)):
            raise ValueError('rib pitch must tile periodically')
    return scale

def _noise(rng, n, sigma):
    field = gaussian_filter(rng.normal(size=(n,n)), sigma, mode='wrap')
    return field / max(field.std(), 1e-8)

def synthesize(observation, resolution=1024):
    o = dict(observation); scale = validate(o)
    if resolution < 128 or resolution > 2048: raise ValueError('resolution outside material budget')
    key = structure_key(o); rng = np.random.default_rng(int(key[:8],16)); n = resolution
    micro = _noise(rng,n,.65); fine = _noise(rng,n,2); broad = _noise(rng,n,22)
    pattern = o['pattern']; y,x = np.mgrid[:n,:n]; x=x*scale[0]/n; y=y*scale[1]/n
    value = fine*.009 + micro*.008 + broad*.018
    height = fine*.000025 + micro*.000012
    rough = np.full((n,n), o.get('roughness',.8)) + fine*.025 + broad*.035
    joint_blend = np.zeros((n,n)); module_variation = np.zeros((n,n))
    if pattern in {'brick_running','panel_grid','ribbed_panel','timber_boards'}:
        cw,ch=o['unit_size_m']; row=np.floor(y/ch+1e-7).astype(int)
        offset = (row%2)*cw*.5 if pattern=='brick_running' else 0
        dx=(x+offset)%cw; dy=y%ch
        edge=np.minimum(np.minimum(dx,cw-dx),np.minimum(dy,ch-dy))
        j=o['joint_width_m']; blend=np.clip((edge-j*.4)/(j*.6),0,1)
        blend=blend*blend*(3-2*blend); joint_blend=1-blend
        cols,rows=np.rint(scale/np.array([cw,ch])).astype(int)
        variations=rng.normal(0, .075 if pattern=='brick_running' else .025, (rows,cols))
        module_variation=variations[row%rows,np.floor((x+offset)/cw).astype(int)%cols]
        height += blend*o.get('relief_m',.001)
        rough += joint_blend*.10
        if pattern=='brick_running':
            value += micro*.018+fine*.014+broad*.025
            height += blend*(fine*.00010+micro*.00007)
        if pattern=='ribbed_panel':
            height += blend*o.get('relief_m',.001)*.3*np.cos(2*np.pi*y/o['rib_pitch_m'])
        if pattern=='timber_boards':
            vertical=o.get('grain_direction','vertical')=='vertical'
            grain=_noise(rng,n,(25,1.1) if vertical else (1.1,25))
            value += grain*.055; height += grain*.00009; rough += grain*.035
    elif pattern=='stone_grain':
        value += _noise(rng,n,5)*.025 + broad*.025
        height += micro*.00005
    elif pattern=='bitumen_grain':
        grains=np.clip(micro-.3,0,2)
        value += grains*.095; height += grains*.00025; rough += grains*.04
    elif pattern=='metal_grain':
        grain=_noise(rng,n,(1,28)); value=grain*.012+fine*.003; height=grain*.000008
    elif pattern=='plaster_grain':
        height += micro*.00010; value += fine*.01
    rgb=np.broadcast_to(np.array(o['color_rgb'],float)/255,(n,n,3)).copy()
    # Pigment changes preserve the same structural realization and mortar relief.
    rgb *= np.clip(1+value+module_variation,.35,1.65)[:,:,None]
    joint=np.array(o.get('joint_rgb',o['color_rgb']),float)/255
    rgb=rgb*(1-joint_blend[:,:,None]) + joint[None,None,:]*(1+fine[:,:,None]*.025)*joint_blend[:,:,None]
    gx=(np.roll(height,-1,1)-np.roll(height,1,1))/(2*scale[0]/n)
    # glTF UV v increases upward while image rows increase downward.
    gy=-(np.roll(height,-1,0)-np.roll(height,1,0))/(2*scale[1]/n)
    normal=np.stack([-gx,-gy,np.ones_like(gx)],2);normal/=np.linalg.norm(normal,axis=2,keepdims=True)
    mr=np.zeros((n,n,3)); mr[:,:,1]=np.clip(rough,.06,1);mr[:,:,2]=o.get('metallic',0)*(1-joint_blend)
    maps=[Image.fromarray(np.uint8(np.clip(a,0,1)*255)) for a in [rgb,normal*.5+.5,mr]]
    variant=hashlib.sha256(json.dumps([o['color_rgb'],o.get('joint_rgb')]).encode()).hexdigest()[:8]
    return PBRMaterial(name='surface_'+key+'_'+variant,baseColorTexture=maps[0],normalTexture=maps[1],metallicRoughnessTexture=maps[2],baseColorFactor=[255]*4,roughnessFactor=1.,metallicFactor=1.), {'structureKey':key,'variantKey':variant,'tile_size_m':scale.tolist(),'source':'metric procedural PBR synthesis; surface assignment accuracy depends on observation provenance'}
