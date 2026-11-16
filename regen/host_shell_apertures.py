"""External prototype for openings covered by a nearby structural shell.

This is deliberately separate from the production compiler. It only acts when
the matched source-camera ray hits the feature's own structural mesh inside a
measured opening, and the local shell slab is supported by normal-line hits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate


SOURCE_TO_GLB = np.array([[1., 0., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])
GLB_TO_SOURCE = np.linalg.inv(SOURCE_TO_GLB)


def profile_tangent(profile):
    n = np.asarray(profile['normal'], dtype=float)
    return np.array([-n[1], n[0]], dtype=float)


def segment_boxes(origin, target, low, high):
    """Intersect a finite camera segment with rows of axis-aligned bounds."""
    origin = np.asarray(origin, float)
    delta = np.asarray(target, float) - origin
    length = np.linalg.norm(delta)
    if length < 1e-8:
        return np.zeros(len(low), dtype=bool)
    direction = delta / length
    near = np.zeros(len(low))
    far = np.full(len(low), length + .02)
    for axis, component in enumerate(direction):
        if abs(component) < 1e-10:
            outside = (origin[axis] < low[:, axis]) | (origin[axis] > high[:, axis])
            far[outside] = -np.inf
            continue
        a = (low[:, axis] - origin[axis]) / component
        b = (high[:, axis] - origin[axis]) / component
        near = np.maximum(near, np.minimum(a, b))
        far = np.minimum(far, np.maximum(a, b))
    return (near <= far) & (far > 1e-5)


def opening_outline_uv(profile, opening, z_base, samples=24):
    """Return the opening contour in global profile coordinates (u, source z)."""
    x0, z0, x1, z1 = map(float, opening['rect'])
    u0, u1 = map(float, profile['frameU'])
    u0, u1 = u0 + (u1-u0)*x0, u0 + (u1-u0)*x1
    z0 = float(z_base) + float(profile['heightRefM'])*z0
    z1 = float(z_base) + float(profile['heightRefM'])*z1
    style = opening.get('style', {})
    head = style.get('head', 'flat')
    if head == 'arch':
        spring = z0 + (z1-z0)*float(style.get('springRatio', .68))
        half_width = (u1-u0)/2.
        rise = max(1e-6, z1-spring)
        radius = (half_width*half_width + rise*rise)/(2.*rise)
        center_z = spring + rise - radius
        left_angle = math.atan2(radius-rise, -half_width)
        right_angle = math.atan2(radius-rise, half_width)
        points = [(u0,z0), (u1,z0), (u1,spring)]
        points.extend(((u0+u1)/2.+radius*math.cos(theta),
                       center_z+radius*math.sin(theta))
                      for theta in np.linspace(right_angle, left_angle, samples+1))
        points.append((u0,spring))
        return Polygon(points)
    if head == 'segmental':
        rise_ratio = float(style.get('riseRatio', .14))
        rise = (z1-z0)*rise_ratio
        spring = z1-rise
        points = [(u0,z0), (u1,z0), (u1,spring)]
        points.extend(((u0+u1)/2.+(u1-u0)/2.*math.cos(t),
                       spring+rise*math.sin(t))
                      for t in np.linspace(0., math.pi, samples+1))
        points.append((u0,spring))
        return Polygon(points)
    if head == 'pointed':
        spring = z0 + (z1-z0)*float(style.get('springRatio', .68))
        return Polygon([(u0,z0),(u1,z0),(u1,spring),((u0+u1)/2.,z1),(u0,spring)])
    return Polygon([(u0,z0),(u1,z0),(u1,z1),(u0,z1)])


def _project(point_source, job):
    c = np.asarray(job['from'], dtype=float)
    f = np.asarray(job['at'], dtype=float)-c
    f /= np.linalg.norm(f)
    world_up = np.array([0., 0., 1.])
    right = np.cross(f, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    d = np.asarray(point_source, dtype=float)-c
    depth = float(d@f)
    focal = float(job['height'])/(2.*math.tan(math.radians(float(job['vfovDeg']))/2.))
    return np.array([float(job['width'])/2.+focal*(d@right)/depth,
                     float(job['height'])/2.-focal*(d@up)/depth,
                     depth])


@dataclass
class TriangleIndex:
    triangles_glb: np.ndarray
    owners: np.ndarray
    source_triangles: np.ndarray
    source_min: np.ndarray
    source_max: np.ndarray

    @classmethod
    def from_scene(cls, scene):
        triangles, owners = [], []
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph.get(node)
            mesh = scene.geometry[geometry_name]
            world = mesh.vertices @ transform[:3,:3].T + transform[:3,3]
            triangles.append(world[mesh.faces])
            owners.extend([node]*len(mesh.faces))
        glb = np.concatenate(triangles) if triangles else np.empty((0,3,3))
        source=glb @ GLB_TO_SOURCE[:3,:3].T
        return cls(glb,np.asarray(owners,dtype=object),source,source.min(axis=1),source.max(axis=1))

    def ray_hits(self, origin_source, target_source, limit=None):
        """Return ordered triangle hits from camera to a measured source point."""
        target_source = np.asarray(target_source, dtype=float)
        origin_source = np.asarray(origin_source, dtype=float)
        origin = (SOURCE_TO_GLB @ np.r_[origin_source,1.])[:3]
        target = target_source
        origin = origin_source
        delta = target-origin
        length = float(np.linalg.norm(delta))
        if length <= 1e-8:
            return []
        direction = delta/length
        if not len(self.source_triangles):
            return []
        # Reject triangles whose AABBs do not intersect this narrow camera
        # segment before doing the more expensive barycentric intersection.
        low=self.source_min-1e-5;high=self.source_max+1e-5
        broad=segment_boxes(origin,target,low,high)
        ids=np.flatnonzero(broad)
        if not len(ids):return []
        tri=self.source_triangles[ids]
        e1 = tri[:,1]-tri[:,0]; e2 = tri[:,2]-tri[:,0]
        h = np.cross(np.broadcast_to(direction, e2.shape),e2)
        a = np.einsum('ij,ij->i',e1,h)
        mask = np.abs(a)>1e-10
        inv = np.zeros_like(a);inv[mask]=1./a[mask]
        s=origin-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h)
        mask &= (u>=-1e-8)&(u<=1.+1e-8)
        q=np.cross(s,e1);v=inv*np.einsum('ij,ij->i',np.broadcast_to(direction,q.shape),q)
        mask &= (v>=-1e-8)&(u+v<=1.+1e-8)
        dist=inv*np.einsum('ij,ij->i',e2,q)
        cap=length if limit is None else min(length,float(limit))
        mask &= (dist>1e-5)&(dist<=cap+.02)
        hit_ids=np.where(mask)[0];hit_ids=hit_ids[np.argsort(dist[hit_ids])]
        hits=[]
        for local_ix in hit_ids:
            ix=int(ids[local_ix]);d=float(dist[local_ix]);node=str(self.owners[ix])
            if any(node==h['node'] and abs(d-h['distanceM'])<.015 for h in hits):
                continue
            point_source=origin+direction*d
            tri_source=self.source_triangles[ix]
            normal=np.cross(tri_source[1]-tri_source[0],tri_source[2]-tri_source[0])
            normal/=max(1e-12,np.linalg.norm(normal))
            hits.append({'distanceM':d,'node':node,'pointSource':point_source,
                         'triangleIndex':int(ix),'normalSource':normal})
        return hits


def opening_center_source(profile, opening, z_base):
    contour=opening_outline_uv(profile,opening,z_base)
    u,z=contour.representative_point().coords[0]
    n=np.asarray(profile['normal'],dtype=float);t=profile_tangent(profile)
    xy=t*u+n*float(profile['planeV'])
    return np.array([xy[0],xy[1],z],dtype=float)


def opening_host_conflicts(scene, profile, job, z_base, index=None,
                           alignment_min=.55, triangle_index=None):
    """Find measured opening centers blocked by their own structural shell."""
    from shapely.geometry import Point
    fid=str(profile.get('match',{}).get('featureId') or profile['id'].split('__')[0].removeprefix('regen-'))
    triangles=triangle_index or TriangleIndex.from_scene(scene)
    camera=np.asarray(job['from'],dtype=float)
    records=[]
    openings=profile.get('openings',[])
    for i,op in enumerate(openings):
        if index is not None and i not in index:
            continue
        outline=opening_outline_uv(profile,op,z_base)
        center=opening_center_source(profile,op,z_base)
        projection=_project(center,job)
        if projection[2]<=0 or projection[0]<0 or projection[0]>=job['width'] or projection[1]<0 or projection[1]>=job['height']:
            continue
        hits=triangles.ray_hits(camera,center)
        if not hits:
            continue
        first=hits[0]
        is_own_host=first['node'].startswith(f'buildings__{fid}___') and '__observed_' not in first['node']
        if not is_own_host:
            continue
        n=np.asarray(profile['normal'],dtype=float);t=profile_tangent(profile)
        u=float(first['pointSource'][:2]@t);z=float(first['pointSource'][2])
        if not outline.buffer(.025).covers(Point(u,z)):
            continue
        align=abs(float(first['normalSource'][:2]@n))
        if align<alignment_min:
            continue
        records.append({'openingIndex':i,'opening':op,'outlineUV':outline,
                        'centerSource':center,'projectionPx':projection[:2],
                        'firstHit':first,'rayHits':hits,'normalAlignment':align,
                        'profilePlaneV':float(profile['planeV']),
                        'hitPlaneV':float(first['pointSource'][:2]@n),
                        'openingHitUV':[u,z]})
    return records


def _line_mesh_hits(mesh, normal, tangent, u, z, plane_v, slab_limit=1.5,
                    alignment_min=.55):
    """Return supported shell depth crossings along a profile-normal line."""
    vertices=np.asarray(mesh.vertices,dtype=float);triangles=vertices[mesh.faces]
    n=np.asarray(normal,dtype=float);t=np.asarray(tangent,dtype=float)
    center=np.r_[t*u+n*plane_v,z];direction=np.r_[n,0.]
    e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0]
    normal_faces=np.cross(e1,e2);length=np.linalg.norm(normal_faces,axis=1)
    valid=length>1e-10;normal_faces[valid]/=length[valid,None]
    align=np.abs(normal_faces[:,:2]@n)
    h=np.cross(np.broadcast_to(direction,e2.shape),e2)
    det=np.einsum('ij,ij->i',e1,h);mask=(np.abs(det)>1e-10)&(align>=alignment_min)
    inv=np.zeros_like(det);inv[mask]=1./det[mask];s=center-triangles[:,0]
    a=inv*np.einsum('ij,ij->i',s,h);mask &= (a>=-1e-8)&(a<=1.+1e-8)
    q=np.cross(s,e1);b=inv*np.einsum('ij,ij->i',np.broadcast_to(direction,q.shape),q)
    mask &= (b>=-1e-8)&(a+b<=1.+1e-8)
    lam=inv*np.einsum('ij,ij->i',e2,q);mask &= np.abs(lam)<=slab_limit
    result=sorted(float(plane_v+x) for x in lam[mask])
    unique=[]
    for val in result:
        if not unique or abs(val-unique[-1])>.015:unique.append(val)
    return unique


def opening_slab(mesh, profile, conflict, z_base, margin=.03):
    """Measure the local parallel host slab from its supported normal crossings."""
    n=np.asarray(profile['normal'],dtype=float);t=profile_tangent(profile)
    contour=conflict['outlineUV'];u,z=conflict['openingHitUV']
    samples=[(u,z),contour.representative_point().coords[0]]
    minx,minz,maxx,maxz=contour.bounds
    for fu in (.2,.5,.8):
        for fz in (.2,.5,.8):
            q=(minx+(maxx-minx)*fu,minz+(maxz-minz)*fz)
            if contour.covers(Point(q)):samples.append(q)
    depths=[]
    for su,sz in samples:
        depths.extend(_line_mesh_hits(mesh,n,t,su,sz,float(profile['planeV'])))
    # The camera hit itself is authoritative if sparse, but never permit a
    # large slab or a cross-building cut from that fallback.
    depths=[v for v in depths if abs(v-float(profile['planeV']))<=1.5]
    if not depths:
        return None
    lo,hi=min(depths)-margin,max(depths)+margin
    if hi-lo>1.5 or not (lo<=conflict['hitPlaneV']+margin and hi>=conflict['hitPlaneV']-margin):
        return None
    return (lo,hi)


def _clip_ring_linear(coords, coeff, bound, keep_above):
    """Clip a 2D polygon ring against a linear depth half-space."""
    points=[np.asarray(p,dtype=float) for p in coords]
    if len(points)>1 and np.allclose(points[0],points[-1]):points=points[:-1]
    out=[]
    def val(p):return coeff[0]*p[0]+coeff[1]*p[1]+coeff[2]-bound
    for a,b in zip(points,points[1:]+points[:1]):
        va,vb=val(a),val(b);ina=va>=-1e-9 if keep_above else va<=1e-9;inb=vb>=-1e-9 if keep_above else vb<=1e-9
        if ina and inb:out.append(b)
        elif ina and not inb:
            f=va/(va-vb);out.append(a+f*(b-a))
        elif not ina and inb:
            f=va/(va-vb);out.extend([a+f*(b-a),b])
    return out


def _triangulated_polygons(geom):
    if geom.is_empty:return []
    polys=[geom] if geom.geom_type=='Polygon' else [g for g in getattr(geom,'geoms',[]) if g.geom_type=='Polygon']
    out=[]
    for poly in polys:
        for tri in triangulate(poly):
            if poly.covers(tri) and tri.area>1e-10:
                coords=np.asarray(tri.exterior.coords[:-1],dtype=float)
                if len(coords)==3:out.append(coords)
    return out


def _barycentric(points, tri_uv):
    matrix=np.column_stack((tri_uv[1]-tri_uv[0],tri_uv[2]-tri_uv[0]))
    coeff=np.linalg.solve(matrix,(np.asarray(points)-tri_uv[0]).T).T
    return np.column_stack((1.-coeff[:,0]-coeff[:,1],coeff[:,0],coeff[:,1]))


def clip_shell_mesh(mesh, profile, outline, slab, z_base, alignment_min=.55):
    """Remove only parallel host-slab surface patches inside one measured contour.

    The input mesh uses source XYZ coordinates. Vertex UVs are interpolated on
    split triangles so texture-backed host materials remain intact.
    """
    import trimesh
    from shapely.geometry import Polygon
    n=np.asarray(profile['normal'],dtype=float);t=profile_tangent(profile);lo,hi=map(float,slab)
    old_uv=getattr(mesh.visual,'uv',None)
    has_uv=old_uv is not None and len(old_uv)==len(mesh.vertices)
    old_normals=np.asarray(mesh.vertex_normals,dtype=float)
    has_normals=old_normals.shape==np.asarray(mesh.vertices).shape
    vertices_out=[];faces_out=[];uv_out=[];normals_out=[];removed_area=0.;affected=0
    for face in mesh.faces:
        src=np.asarray(mesh.vertices[face],dtype=float)
        uv=np.column_stack((src[:,:2]@t,src[:,2]));depth=src[:,:2]@n
        face_normal=np.cross(src[1]-src[0],src[2]-src[0]);norm=np.linalg.norm(face_normal)
        alignment=abs(float(face_normal[:2]@n))/max(norm,1e-12)
        tri_poly=Polygon(uv)
        if (alignment<alignment_min or tri_poly.area<1e-10
                or depth.max()<lo-1e-5 or depth.min()>hi+1e-5
                or not tri_poly.intersects(outline)):
            start=len(vertices_out);vertices_out.extend(src);faces_out.append([start,start+1,start+2])
            if has_uv:uv_out.extend(old_uv[face])
            if has_normals:normals_out.extend(old_normals[face])
            continue
        # V is affine over each supported triangle. Clip the overlap to the
        # measured slab, then preserve every part of the original triangle
        # outside that opening volume.
        coeff=np.linalg.solve(np.column_stack((uv,np.ones(3))),depth)
        overlap=tri_poly.intersection(outline)
        cut_parts=[]
        polys=[overlap] if overlap.geom_type=='Polygon' else [g for g in getattr(overlap,'geoms',[]) if g.geom_type=='Polygon']
        for poly in polys:
            ring=_clip_ring_linear(poly.exterior.coords,coeff,lo,True)
            if len(ring)<3:continue
            ring=_clip_ring_linear(ring,coeff,hi,False)
            if len(ring)<3:continue
            cut=Polygon(ring)
            if cut.is_valid and cut.area>1e-9:cut_parts.append(cut)
        if not cut_parts:
            start=len(vertices_out);vertices_out.extend(src);faces_out.append([start,start+1,start+2])
            if has_uv:uv_out.extend(old_uv[face])
            if has_normals:normals_out.extend(old_normals[face])
            continue
        cut_geom=cut_parts[0] if len(cut_parts)==1 else __import__('shapely').geometry.MultiPolygon(cut_parts)
        removed_area+=float(cut_geom.area);affected+=1
        remainder=tri_poly.difference(cut_geom)
        for fragment in _triangulated_polygons(remainder):
            weights=_barycentric(fragment,uv)
            src_fragment=weights@src
            start=len(vertices_out);vertices_out.extend(src_fragment);faces_out.append([start,start+1,start+2])
            if has_uv:uv_out.extend(weights@np.asarray(old_uv[face],dtype=float))
            if has_normals:
                normals_fragment=weights@old_normals[face]
                lengths=np.linalg.norm(normals_fragment,axis=1,keepdims=True)
                normals_out.extend(normals_fragment/np.maximum(lengths,1e-12))
    if affected==0:
        return mesh.copy(),{'removedProjectedAreaM2':0.,'affectedTriangles':0}
    result=trimesh.Trimesh(vertices=np.asarray(vertices_out),faces=np.asarray(faces_out),process=False)
    if has_normals and len(normals_out)==len(vertices_out):
        result.vertex_normals=np.asarray(normals_out,dtype=float)
    if has_uv:
        from trimesh.visual.texture import TextureVisuals
        result.visual=TextureVisuals(uv=np.asarray(uv_out),material=mesh.visual.material)
    elif hasattr(mesh.visual,'material'):
        result.visual.material=mesh.visual.material
    result.metadata=dict(getattr(mesh,'metadata',{}))
    return result,{'removedProjectedAreaM2':removed_area,'affectedTriangles':affected}


def reveal_mesh(profile, outline, slab, z_base, material=None, segments=24):
    """Create a simple ruled perimeter return between measured shell and frame."""
    import trimesh
    n=np.asarray(profile['normal'],dtype=float);t=profile_tangent(profile);lo,hi=map(float,slab)
    u0,z0,u1,z1=outline.bounds
    # A closed sampled outer ring is enough for rectilinear and curved heads.
    coords=list(outline.exterior.coords[:-1])
    # densify long contour edges so the return remains smooth on arches.
    dense=[]
    for a,b in zip(coords,coords[1:]+coords[:1]):
        dense.append(a)
        count=max(1,int(math.ceil(np.linalg.norm(np.asarray(b)-a)/.22)))
        dense.extend((np.asarray(a)+(np.asarray(b)-a)*k/count).tolist() for k in range(1,count))
    # Place the inner edge on the back of the observed local slab; the existing
    # inset frame/glass/leaf remains visible there.
    inner=float(profile['planeV'])
    if not (lo-.1<=inner<=hi+.1):inner=float(np.clip(inner,lo,hi))
    vertices=[];faces=[]
    for u,z in dense:
        vertices.extend([[*(t*u+n*hi),z],[*(t*u+n*inner),z]])
    for i in range(len(dense)):
        j=(i+1)%len(dense);a=2*i;b=2*j
        faces.extend([[a,b,b+1],[a,b+1,a+1]])
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    if material is not None:
        from trimesh.visual.texture import TextureVisuals
        # Meter-valued planar UVs keep the return pattern tied to the measured
        # source frame rather than restarting on each generated quad.
        uv=[]
        for u,z in dense:uv.extend([[u,z],[u,z]])
        mesh.visual=TextureVisuals(uv=np.asarray(uv),material=material)
    return mesh


def _receiver_material(name, scene=None, feature_id=None):
    """Resolve a receiver finish from the measured style, reusing GLB materials."""
    import trimesh
    if scene is not None and feature_id:
        suffix='___'+str(name)
        for node in scene.graph.nodes_geometry:
            if node.startswith(f'facades__{feature_id}__observed') and node.endswith(suffix):
                _,geometry_name=scene.graph.get(node)
                material=getattr(scene.geometry[geometry_name].visual,'material',None)
                if material is not None:return material
        if name=='glass':
            for node in scene.graph.nodes_geometry:
                if node.startswith(f'facades__{feature_id}__observed') and node.endswith('___glass'):
                    _,geometry_name=scene.graph.get(node)
                    material=getattr(scene.geometry[geometry_name].visual,'material',None)
                    if material is not None:return material
    color=[90,90,90,255]
    if isinstance(name,str) and len(name)==7 and name.startswith('#'):
        try:color=[int(name[i:i+2],16) for i in (1,3,5)]+[255]
        except ValueError:pass
    elif name=='glass':
        color=[90,115,133,255]
    return trimesh.visual.material.PBRMaterial(name=str(name),baseColorFactor=np.asarray(color,dtype=np.uint8),
                                                metallicFactor=.05,roughnessFactor=.65)


def receiver_prism(profile, poly, z_base, depth, thickness, material):
    """Extrude a measured facade polygon inward from its source plane."""
    import trimesh
    from trimesh.visual.texture import TextureVisuals
    poly=poly if poly.geom_type=='Polygon' else max(poly.geoms,key=lambda g:g.area)
    if poly.is_empty or poly.area<1e-5:return None
    tangent=profile_tangent(profile);normal=np.asarray(profile['normal'],dtype=float)
    # Local prism axes are source tangent, world up, and inward (-normal).
    local_poly=Polygon([(u,z-z_base) for u,z in poly.exterior.coords],
                       [[(u,z-z_base) for u,z in ring.coords]
                        for ring in poly.interiors])
    mesh=trimesh.creation.extrude_polygon(local_poly,height=float(thickness),engine='earcut')
    transform=np.eye(4)
    transform[:3,0]=[tangent[0],tangent[1],0.]
    transform[:3,1]=[0.,0.,1.]
    transform[:3,2]=[-normal[0],-normal[1],0.]
    transform[:3,3]=[* (normal*(float(profile['planeV'])+float(depth))),float(z_base)]
    mesh.apply_transform(transform)
    # Source-coordinate metre UVs preserve course direction across the whole
    # reassembled receiver instead of restarting per pane/frame piece.
    uv=np.column_stack((mesh.vertices[:,:2]@tangent,mesh.vertices[:,2]))
    mesh.visual=TextureVisuals(uv=uv,material=material)
    return mesh


def build_measured_receiver(profile, opening, z_base, scene=None, opening_index=0):
    """Rebuild a full source-style receiver for one opening, uncut by edge splits.

    The receiver uses only the opening's measured contour and style. A window
    gets shaped glazing, perimeter frame, meeting rails and measured lites; an
    entry gets the specified number/kind of opaque or glazed leaves.
    """
    from shapely.geometry import box
    style=opening.get('style') or {};feature_id=profile.get('match',{}).get('featureId')
    outline=opening_outline_uv(profile,opening,z_base)
    if outline.is_empty or not outline.is_valid:return []
    x0,z0,x1,z1=outline.bounds
    frame_width=float(style.get('frameWidthM',.06));recess=float(style.get('recessM',.2))
    frame_name=style.get('frameMaterial','#333333');parts=[]
    def add(poly,depth,thick,material,role):
        if poly.is_empty:return
        if hasattr(poly,'geoms'):
            for piece in poly.geoms:add(piece,depth,thick,material,role)
            return
        if poly.geom_type!='Polygon' or poly.area<1e-5:return
        mesh=receiver_prism(profile,poly,z_base,depth,thick,
                            _receiver_material(material,scene,feature_id))
        if mesh is not None:parts.append({'mesh':mesh,'role':role,'openingIndex':opening_index})

    inner=outline.buffer(-max(.015,frame_width),join_style=2)
    if inner.is_empty:inner=outline
    kind=style.get('kind','window')
    if kind=='entry':
        door=style.get('door') or {}
        door_kind=door.get('kind','none')
        if door_kind!='none':
            mat=door.get('material',frame_name);leaf_bottom=z0+frame_width;leaf_top=z1-frame_width
            leaf_left=x0+frame_width;leaf_right=x1-frame_width
            if leaf_top-leaf_bottom>.5 and leaf_right-leaf_left>.3:
                if door_kind=='double':
                    gap=min(.035,(leaf_right-leaf_left)*.025);mid=(leaf_left+leaf_right)/2.
                    bounds=[(leaf_left,leaf_bottom,mid-gap/2,leaf_top),
                            (mid+gap/2,leaf_bottom,leaf_right,leaf_top)]
                else:bounds=[(leaf_left,leaf_bottom,leaf_right,leaf_top)]
                for j,(a,b,c,d) in enumerate(bounds):
                    leaf=box(a,b,c,d).intersection(inner)
                    cells=door.get('cells') or []
                    if door_kind=='glazed' and not cells:
                        # The position of the glazing in this leaf is unknown.
                        # Keep the source assembly rather than inventing a cut.
                        return []
                    if cells:
                        for cell in cells:
                            ca,cb,cc,cd=map(float,cell['rect'])
                            cell_poly=box(a+(c-a)*ca,b+(d-b)*cb,a+(c-a)*cc,b+(d-b)*cd).intersection(leaf)
                            if cell.get('kind')=='glass':
                                add(cell_poly,-recess+.11,.025,'glass','door-glass')
                            else:add(cell_poly,-recess+.08,.025,mat,'door-panel')
                        for cell in cells:
                            if cell.get('kind')=='glass':
                                ca,cb,cc,cd=map(float,cell['rect'])
                                leaf=leaf.difference(box(a+(c-a)*ca,b+(d-b)*cb,
                                                         a+(c-a)*cc,b+(d-b)*cd))
                    if leaf.area>.01:add(leaf,-recess+.045,.08,mat,f'door-leaf-{j}')
        if inner.area>0:
            add(outline.difference(inner),.03,recess+.03,frame_name,'entry-frame')
    elif kind in ('window','shopfront','terrace_frame'):
        glass_name=style.get('glassMaterial','glass')
        if inner.area>0:add(inner,-recess,.025,glass_name,'glazing')
        if inner.area>0:add(outline.difference(inner),.03,recess+.03,frame_name,'window-frame')
        if inner.area>0:
            from pipeline.window_panes import inner_bars
            spring_ratio=float(style.get('springRatio',.68))
            head=style.get('head','flat')
            top=z1 if head=='flat' else z0+(z1-z0)*spring_ratio
            left=x0+frame_width;right=x1-frame_width;bottom=z0+frame_width
            for bar,depth in inner_bars((left,bottom,right,top),style):
                role='meeting-rail' if depth==.02 else 'muntin'
                add(bar.intersection(inner),-recess+.035,.06,frame_name,role)
            for axis,key,role in ((0,'mullionFractions','mullion'),(1,'transomFractions','transom')):
                for fraction in style.get(key) or []:
                    if not 0.<float(fraction)<1.:continue
                    if axis==0:
                        x=left+(right-left)*float(fraction)
                        bar=box(x-.021,bottom,x+.021,z1)
                    else:
                        z=bottom+(top-bottom)*float(fraction)
                        bar=box(left,z-.021,right,z+.021)
                    add(bar.intersection(inner),-recess+.04,.065,frame_name,role)
    else:
        return []

    surround=style.get('surround')
    if surround:
        width=float(surround.get('widthM',.12));ring=outline.buffer(width,join_style=2).difference(outline)
        if kind=='entry':ring=ring.difference(box(x0-width-.1,z0-1.,x1+width+.1,z0))
        if not ring.is_empty:add(ring,float(surround.get('projectionM',.04)),.04,
                                 surround.get('material',frame_name),'surround')
    return parts
