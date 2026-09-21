"""Location-free material selection, recoloring and deduplicated acquisition requests.
Inputs are surface observations, not feature IDs. Color cannot override structure.
"""
import hashlib, json
import numpy as np
from PIL import Image


def select_texture(target, assets):
    required=('substrate_family','pattern','finish_class')
    color=np.asarray(target.get('color_rgb',[]),dtype=float)
    if color.shape!=(3,) or not np.isfinite(color).all() or np.any((color<0)|(color>255)):
        return {'action':'retain_unresolved','reason':'invalid color observation'}
    if any(not target.get(k) for k in required):
        return {'action':'retain_unresolved','reason':'missing surface structure observation'}
    compatible=[]
    for aid,a in assets.items():
        if any(a.get(k)!=target[k] for k in required):continue
        if a.get('strong_weathering',False)!=target.get('strong_weathering',False):continue
        rotate=False;rescale=None;fit=None
        if target.get('unit_size_m'):
            # The module measured in the scan must match the observed module within a quarter, on every jointed
            # axis; a scan may be turned by a right angle (boards run either way), never stretched into a pattern.
            if not a.get('unit_size_m'):continue
            unit=np.array(a['unit_size_m'],float);open_axes=set(a.get('unit_open_axes',[]))
            fits=[]
            for turned,(u,oa) in enumerate(((unit,open_axes),(unit[::-1],{1-i for i in open_axes}))):
                errors=[abs(np.log(u[i]/target['unit_size_m'][i])) for i in range(2) if i not in oa]
                fits.append((max(errors) if errors else 0.,bool(turned),u,oa))
            fit,rotate,unit,open_axes=min(fits,key=lambda f:f[0])
            if fit>np.log(1.25):continue
            # The observed module sets the physical scale: the source tile is rescaled so its module equals the
            # observed one; the two axes must agree within a tenth (an open axis follows the jointed one).
            ratios=[target['unit_size_m'][i]/unit[i] for i in range(2) if i not in open_axes]
            if max(ratios)/min(ratios)>1.1:continue
            rescale=[target['unit_size_m'][i]/unit[i] if i not in open_axes else float(np.mean(ratios)) for i in range(2)]
        scale=a.get('tile_size_m'); estimated=False
        if scale is None:
            # Only unjointed fine grain may use an explicitly provisional repeat.
            if a['pattern'] not in ('plaster_grain','smooth_grain') or not target.get('provisional_repeat_m'):continue
            scale=target['provisional_repeat_m'];estimated=True
        elif rescale is not None:
            source=np.array(scale,float)[::-1] if rotate else np.array(scale,float)
            scale=(source*np.array(rescale)).round(4).tolist()
        delta=float(np.linalg.norm(np.array(a['mean_rgb'])-color)/255)
        if a.get('intrinsic_color',True) and delta>.35:continue
        compatible.append((delta+.08*estimated+.02*rotate,aid,a,scale,estimated,delta,rotate,rescale,fit))
    if not compatible:
        signature={k:target.get(k) for k in (*required,'strong_weathering','unit_size_m')}
        key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
        return {'action':'request_new','request_key':key,'requirements':signature,'reason':'no compatible pattern, finish and physical scale; color alone is not a new asset requirement'}
    _,aid,a,scale,estimated,delta,rotate,rescale,fit=min(compatible,key=lambda x:(x[0],x[1]))
    provenance='provisional target repeat, not measured' if estimated else 'asset source metadata'
    if rescale is not None:provenance='asset source metadata, rescaled to the observed module measured in the scan (module fit %.3f in log units)'%fit
    return {'action':'recolor' if delta>.06 else 'reuse','asset_id':aid,'tile_size_m':scale,'rotate90':rotate,
            'physical_rescale':rescale,'module_source':a.get('unit_size_m'),'scale_provenance':provenance,
            'color_distance':delta,'color_rgb':target['color_rgb'],'reason':'compatible substrate, pattern, finish and weathering'}


def recolor(image, target_rgb, preserve_joints=False):
    """Preserve luminance variation; retain pale mortar on intrinsic masonry.
    This is a computational material transform, not generated facade imagery.
    """
    rgb=np.asarray(image.convert('RGB'),dtype=float)/255
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    color=np.asarray(target_rgb,dtype=float)/255
    target=np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4)
    mask=np.ones(rgb.shape[:2],bool)
    if preserve_joints:
        chroma=rgb.max(2)-rgb.min(2)
        mask=chroma>.12
        if mask.mean()<.2:raise ValueError('Cannot safely isolate masonry from mortar')
    mean=linear[mask].mean(0)
    changed=linear*np.clip(target/np.maximum(mean,.002),.03,30)
    linear[mask]=changed[mask]
    srgb=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(srgb*255,0,255)))


def metric_uv(mesh, scale):
    """Metric planar UVs per face; existing vertex splits are respected.
    Reject incompatible shared seams rather than splitting/changing geometry.
    """
    uv=np.zeros((len(mesh.vertices),2));seen=np.zeros(len(mesh.vertices),bool)
    for normal,face in zip(mesh.face_normals,mesh.faces):
        if not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-8:
            raise ValueError('empty or invalid face has no texture frame')
        if abs(normal[2])>.999:
            u=np.array([1.,0,0]);v=np.array([0.,1,0])
        else:
            u=np.cross([0,0,1.],normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
        coords=np.column_stack((mesh.vertices[face]@u/scale[0],mesh.vertices[face]@v/scale[1]))
        old=seen[face]
        if old.any() and not np.allclose(uv[face[old]],coords[old],atol=1e-5):
            raise ValueError('shared UV seam requires vertex split')
        uv[face]=coords;seen[face]=True
    return uv
