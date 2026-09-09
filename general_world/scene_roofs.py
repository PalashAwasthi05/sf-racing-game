"""Uniform roof repair of an existing detailed scene using point observations.

This compatibility pass preserves the input's other geometry and its provenance;
it does not certify the input facades or make their prior generation general.
The input must be in metres, X/Y horizontal and Z up, in the cloud's datum.
"""
from dataclasses import replace
import numpy as np
import trimesh
from shapely import contains_xy,intersects_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union
from trimesh.visual.texture import TextureVisuals
from .geometry import polygons
from .roof_planes import RoofPolicy,fit_roof,roof_solid,evaluate

def caps(mesh):
    ids=np.where(mesh.face_normals[:,2]>.99999)[0]
    elevations=np.round(mesh.triangles_center[ids,2],4)
    for height in np.unique(elevations):
        selected=ids[elevations==height]
        region=unary_union([Polygon(t[:,:2]) for t in mesh.triangles[selected]])
        for g in polygons(region):
            if g.area>=20:yield float(height),g

def _textured(mesh,material,scale=(2.,2.)):
    # Per-face vertices preserve hard roof ridges and physical UV scale.
    vertices=mesh.triangles.reshape(-1,3);normals=np.repeat(mesh.face_normals,3,axis=0)
    uv=np.zeros((len(vertices),2));horizontal=np.abs(normals[:,2])>.5
    uv[horizontal]=vertices[horizontal,:2]/scale
    tangent=np.c_[-normals[~horizontal,1],normals[~horizontal,0]]
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    uv[~horizontal,0]=np.sum(vertices[~horizontal,:2]*tangent,axis=1)/scale[0];uv[~horizontal,1]=vertices[~horizontal,2]/scale[1]
    return trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),vertex_normals=normals,
                          visual=TextureVisuals(uv=uv,material=material),process=False)

def clip_under_roof(mesh,region,model):
    """Trim triangles to the same roof plane while preserving material UVs.

    Only triangles wholly owned by this cap's horizontal domain are trimmed.
    Straddling ownership boundaries are retained and reported, never guessed.
    """
    if model['operation']!='min':raise ValueError('Clipping requires a convex plane envelope')
    tri=mesh.triangles;hi=evaluate(model,tri[:,:,:2].reshape(-1,2)).reshape(-1,3)
    exceeds=(tri[:,:,2]>hi+1e-5).any(axis=1)
    if not exceeds.any():return mesh,{'clippedTriangles':0,'ambiguousTriangles':0}
    ids=np.where(exceeds)[0];owned=intersects_xy(region.buffer(.002),tri[ids,:,:2].reshape(-1,2)[:,0],tri[ids,:,:2].reshape(-1,2)[:,1]).reshape(-1,3)
    selected=ids[owned.all(axis=1)];ambiguous=int((owned.any(axis=1)&~owned.all(axis=1)).sum())
    if not len(selected):return mesh,{'clippedTriangles':0,'ambiguousTriangles':ambiguous}
    uv=mesh.visual.uv if hasattr(mesh.visual,'uv') and mesh.visual.uv is not None else np.zeros((len(mesh.vertices),2))
    vertices=[];coords=[];remove=np.zeros(len(mesh.faces),bool);remove[selected]=True
    for face in mesh.faces[selected]:
        poly=np.c_[mesh.vertices[face],uv[face]]
        for a,b,c in model['planes']:
            if len(poly)<3:break
            result=[]
            for start,end in zip(poly,np.roll(poly,-1,axis=0)):
                ds=start[2]-a*start[0]-b*start[1]-c;de=end[2]-a*end[0]-b*end[1]-c
                if ds<=0:result.append(start)
                if (ds<=0)!=(de<=0):result.append(start+(end-start)*(ds/(ds-de)))
            poly=np.asarray(result)
        if len(poly)>=3:
            for i in range(1,len(poly)-1):
                t=poly[[0,i,i+1]]
                if np.linalg.norm(np.cross(t[1,:3]-t[0,:3],t[2,:3]-t[0,:3]))>1e-9:vertices.extend(t[:,:3]);coords.extend(t[:,3:])
    remaining=mesh.submesh([np.where(~remove)[0]],append=True,repair=False)
    if vertices:
        extra=trimesh.Trimesh(np.asarray(vertices),np.arange(len(vertices)).reshape(-1,3),visual=TextureVisuals(uv=np.asarray(coords),material=mesh.visual.material),process=False)
        remaining=trimesh.util.concatenate([remaining,extra])
    remaining.metadata=mesh.metadata.copy()
    return remaining,{'clippedTriangles':len(selected),'ambiguousTriangles':ambiguous}

def repair_scene(scene,cloud):
    report={'capsAttempted':0,'capsReplaced':0,'records':[],'inputFacadesReconstructed':False,'locationRules':0}
    staged=[];policy=replace(RoofPolicy(),minimum_cells=18)
    roof_items=[(name,m) for name,m in scene.geometry.items() if m.metadata.get('layer')=='roofs']
    for name,mesh in roof_items:
        for height,g in caps(mesh):
            report['capsAttempted']+=1
            q=cloud[((cloud[:,:2]>=g.bounds[:2])&(cloud[:,:2]<=g.bounds[2:])).all(axis=1)]
            model,fit=fit_roof(g,q,policy)
            record={'mesh':name,'capHeightM':height,'areaM2':g.area,**fit,'applied':False}
            report['records'].append(record)
            if not model or not fit.get('sloped') or model['operation']!='min':continue
            vertices=np.concatenate([np.asarray(p.exterior.coords) for p in polygons(g)])
            predicted=evaluate(model,vertices)
            if np.max(np.abs(predicted-height))>4:
                record['repairRejection']='incompatible-existing-roof-support';continue
            bottom=min(height-.10,float(predicted.min())-.10)
            replacement=roof_solid(g,model,bottom)
            staged.append((name,mesh.metadata.get('featureId'),g,height,model,replacement,record))
    for index,(name,fid,g,height,model,replacement,record) in enumerate(staged):
        source=scene.geometry[name];tri=source.triangles
        xy=tri[:,:,:2].reshape(-1,2);inside=intersects_xy(g.buffer(.002),xy[:,0],xy[:,1]).reshape(-1,3).all(axis=1)
        at_old_cap=(tri[:,:,2].max(axis=1)<=height+.002)&(tri[:,:,2].min(axis=1)>=height-.105)
        selected=inside&at_old_cap
        if not selected.any():record['repairRejection']='ambiguous-cap-ownership';continue
        kept=source.submesh([np.where(~selected)[0]],append=True,repair=False)
        kept.metadata=source.metadata.copy();scene.geometry[name]=kept
        new=_textured(replacement,source.visual.material);new.metadata={**source.metadata,'evidence':'held-out-lidar-roof-plane'}
        scene.add_geometry(new,geom_name='roof_plane_'+str(index),node_name='roof_plane_'+str(index))
        clipped=ambiguous=0
        for wallname,wall in list(scene.geometry.items()):
            if wall.metadata.get('featureId')!=fid or wall.metadata.get('layer') not in {'buildings','facades'}:continue
            changed,check=clip_under_roof(wall,g,model);scene.geometry[wallname]=changed
            clipped+=check['clippedTriangles'];ambiguous+=check['ambiguousTriangles']
        record.update(applied=True,clippedWallTriangles=clipped,ambiguousBoundaryTriangles=ambiguous,
                      replacementWatertight=bool(replacement.is_watertight),replacementVolumeM3=float(replacement.volume))
        report['capsReplaced']+=1
    report['buildingsChanged']=len({r['mesh'] for r in report['records'] if r['applied']})
    return report
