"""Reuse a measured running-bond scan before synthesizing a new structure."""
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial
from .material_periodicity import infer_joint_periods
from .texture_selection import recolor

def running_bond_scale(color, roughness, source_tile_m, target_module_m):
    """Require two maps to agree on full staggered module, never a half period."""
    estimates=[infer_joint_periods(np.asarray(image)) for image in (color,roughness)]
    units=[]
    for image,e in zip((color,roughness),estimates):
        row=e['horizontal'].get('period_px'); bond=e.get('banded_running_bond',{})
        col=bond.get('full_module_period_px')
        if not row or not col or not bond.get('stagger_consistent'):
            raise ValueError('scan module cannot be measured unambiguously')
        units.append(np.array(source_tile_m)*np.array([col/image.width,row/image.height]))
    if not np.allclose(units[0],units[1],rtol=.08,atol=0):raise ValueError('scan maps disagree on module dimensions')
    unit=np.mean(units,axis=0);target=np.asarray(target_module_m,float);source=np.asarray(source_tile_m,float)
    if target.shape!=(2,) or not np.isfinite(target).all() or np.min(target)<=0:raise ValueError('invalid target module')
    ratio=target/unit
    if np.min(ratio)<.75 or np.max(ratio)>1.3 or max(ratio)/min(ratio)>1.1:
        raise ValueError('scan morphology needs a different source')
    return (source*ratio).tolist(), {'moduleMeasurementM':unit.tolist(),'physicalRescale':ratio.tolist(),'measurement':'independent color/roughness periods plus staggered per-row agreement','estimates':estimates}

def variant(color,normal,roughness,source_tile_m,target):
    if target['pattern']!='brick_running':raise ValueError('unsupported scan morphology')
    if len({color.size,normal.size,roughness.size})!=1:raise ValueError('scan maps have inconsistent dimensions')
    scale,evidence=running_bond_scale(color,roughness,source_tile_m,target['unit_size_m'])
    rgb=recolor(color,target['color_rgb'],preserve_joints=True)
    # Adjust normal slopes to the new physical footprint of the same scanned relief.
    n=np.asarray(normal.convert('RGB'),float)/255*2-1
    n[:,:,:2] *= np.asarray(source_tile_m)/np.asarray(scale)
    n/=np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-8)
    normal_out=Image.fromarray(np.uint8(np.clip(n*.5+.5,0,1)*255))
    r=roughness.convert('L');mr=Image.merge('RGB',(Image.new('L',r.size,0),r,Image.new('L',r.size,0)))
    material=PBRMaterial(baseColorTexture=rgb,normalTexture=normal_out,metallicRoughnessTexture=mr,baseColorFactor=[255]*4,roughnessFactor=1.,metallicFactor=0.)
    return material,dict(evidence,tile_size_m=scale,source='existing scanned material, recolored with measured module rescale')
