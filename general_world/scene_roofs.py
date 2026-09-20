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
from .roof_planes import RoofPolicy,fit_roof,roof_solid,evaluate,sample_cells,fit_patches

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

def _clip_poly(poly,normal,offset):
    if len(poly)<3:return np.empty((0,5))
    result=[]
    for start,end in zip(poly,np.roll(poly,-1,axis=0)):
        ds=start[:3]@normal-offset;de=end[:3]@normal-offset
        if ds<=1e-9:result.append(start)
        if (ds<=0)!=(de<=0):result.append(start+(end-start)*(ds/(ds-de)))
    return np.asarray(result).reshape(-1,poly.shape[1])

def _owned_pieces(poly,region):
    """Split triangle attributes at footprint boundaries without changing UVs."""
    from shapely.geometry import LineString
    xy=poly[:,:2];basis=(xy[1:]-xy[0]).T
    if abs(np.linalg.det(basis))>1e-8:
        tri=Polygon(xy);inverse=np.linalg.inv(basis)
        result=[]
        for owned,g in [(True,tri.intersection(region)),(False,tri.difference(region))]:
            for p in polygons(g):
                v,f=trimesh.creation.triangulate_polygon(p,engine='earcut')
                weights=(v-xy[0])@inverse.T
                values=poly[0]+weights@(poly[1:]-poly[0])
                for face in f:
                    q=values[face]
                    if np.dot(np.cross(q[1,:3]-q[0,:3],q[2,:3]-q[0,:3]),np.cross(poly[1,:3]-poly[0,:3],poly[2,:3]-poly[0,:3]))<0:q=q[::-1]
                    result.append((q,owned))
        return result
    lengths=np.linalg.norm(xy[1:]-xy[0],axis=1);u=(xy[1+np.argmax(lengths)]-xy[0])/max(lengths.max(),1e-9)
    if lengths.max()<1e-8:return [(poly,False)]
    distances=(xy-xy[0])@u;lo,hi=distances.min(),distances.max();line=LineString([xy[0]+u*lo,xy[0]+u*hi]);intersection=line.intersection(region)
    pieces=[intersection] if intersection.geom_type=='LineString' else list(getattr(intersection,'geoms',[]));intervals=[]
    for p in pieces:
        if p.geom_type=='LineString' and p.length>1e-8:intervals.append(sorted((np.asarray(p.coords)-xy[0])@u))
    boundaries=sorted({lo,hi,*[float(s) for r in intervals for s in [r[0],r[-1]]]});result=[];n=np.r_[u,0];base=xy[0]@u
    for a,b in zip(boundaries,boundaries[1:]):
        if b-a<1e-8:continue
        q=_clip_poly(_clip_poly(poly,-n,-a-base),n,b+base)
        if len(q)>=3:result.append((q,any(r[0]-1e-8<=(a+b)/2<=r[-1]+1e-8 for r in intervals)))
    return result

def _unexplained_cells(footprint,points,model,policy):
    """Cells inside the footprint whose returns the accepted envelope does not explain: a rooftop structure,
    a parapet, plant. The replacement passes under them; whatever stands there is not trimmed."""
    from shapely.geometry import box
    cells,_=sample_cells(footprint,points,policy)
    if not len(cells):return None
    bad=np.abs(evaluate(model,cells[:,:2])-cells[:,2])>policy.maximum_p95_error_m
    if not bad.any():return None
    origin=np.array(footprint.bounds[:2]);keys=np.floor((cells[bad,:2]-origin)/policy.cell_m)
    return unary_union([box(*(origin+k*policy.cell_m),*(origin+(k+1)*policy.cell_m)) for k in keys])

_NOT_WALL=('glass','curtain','recess','interior','aluminum','steel','zinc')

def _continue_wall_below(scene,replacement,owner,sides,by_material,reach_m=1.0,below_m=1.0,above_m=.6,minimum_band_m=1.0):
    """Faces of a roof solid that stand on a wall of the building take the material at the top of that wall.

    Candidates are vertical wall faces (the building's wall layer, and clad panels of its facade layer) whose top
    lies within `below_m` under or `above_m` over the roof face's bottom, in plan within `reach_m` of the roof
    face's plane and inside its run, and at least `minimum_band_m` tall (a trim or coping is not a wall band).
    The topmost such band (highest bottom edge) wins: a corrugated cladding above a painted base continues up
    the sloped wall the roof returns show, the base colour does not. Returns the set of faces assigned.
    """
    walls=[]
    for name,m in scene.geometry.items():
        if m.metadata.get('featureId')!=owner or not hasattr(m.visual,'uv') or m.visual.uv is None:continue
        layer=m.metadata.get('layer');mname=str(getattr(m.visual.material,'name','') or '')
        if layer=='facades':
            if not mname.startswith('panel'):continue
        elif layer!='buildings':continue
        if any(mname.startswith(p) for p in _NOT_WALL):continue
        vertical=np.where(np.abs(m.face_normals[:,2])<.05)[0]
        if not len(vertical):continue
        tri=m.triangles[vertical];walls.append((m,vertical,tri[:,:,:2].mean(axis=1),tri[:,:,2].max(axis=1),tri[:,:,2].min(axis=1),m.area_faces[vertical]))
    if not walls:return set()
    done=set()
    for face in sides:
        q=replacement.triangles[face];bottom=float(q[:,2].min());nf=replacement.face_normals[face][:2];nf=nf/max(np.linalg.norm(nf),1e-9)
        t=np.array([-nf[1],nf[0]]);cxy=q[:,:2].mean(axis=0);half=float(np.abs((q[:,:2]-cxy)@t).max())+.5;tally={}
        for m,vertical,cxys,tops,bottoms,areas in walls:
            d=cxys-cxy;sel=(np.abs(d@nf)<=reach_m)&(np.abs(d@t)<=half)&(tops>=bottom-below_m)&(tops<=bottom+above_m)&(tops-bottoms>=minimum_band_m)
            if not sel.any():continue
            key=id(m.visual.material);slot=tally.setdefault(key,[m,int(vertical[np.where(sel)[0][0]]),-1e9])
            slot[2]=max(slot[2],float(bottoms[sel].max()))   # the band standing highest on the wall
        if not tally:continue
        ref,rf,_=max(tally.values(),key=lambda s:s[2]);source=ref.triangles[rf];source_uv=ref.visual.uv[ref.faces[rf]]
        edges=np.linalg.norm(source[1:]-source[0],axis=1);uv_edges=np.linalg.norm(source_uv[1:]-source_uv[0],axis=1);ok=edges>1e-6
        scale=float(np.median(uv_edges[ok]/edges[ok])) if ok.any() else 1.
        if not (np.isfinite(scale) and scale>0):scale=1.
        t3=np.array([-nf[1],nf[0],0.])
        slot=by_material.setdefault(id(ref.visual.material),{'material':ref.visual.material,'v':[],'uv':[]});slot['v'].extend(q);slot['uv'].extend(np.c_[q@t3*scale,q[:,2]*scale]);done.add(int(face))
    return done

def _inherit_wall_sides(scene,replacement,owner,roof_material):
    """New gable returns inherit the nearest aligned input wall's UV mapping.

    This propagates existing material evidence; it does not establish that the
    input material classification is correct. Roof covering stays on roof faces.
    """
    from scipy.spatial import cKDTree
    references=[];centers=[];normals=[]
    for name,m in scene.geometry.items():
        if m.metadata.get('featureId')!=owner or m.metadata.get('layer')!='buildings' or not hasattr(m.visual,'uv') or m.visual.uv is None:continue
        for i in np.where(np.abs(m.face_normals[:,2])<.05)[0]:
            references.append((m,i));centers.append(m.triangles_center[i]);normals.append(m.face_normals[i])
    sides=np.where(np.abs(replacement.face_normals[:,2])<.05)[0];by_material={};unassigned=[]
    # A roof solid's vertical face continues the wall it stands on: it takes the material found at the top of
    # this building's wall directly below it (a clad band, a panel, the wall itself), mapped by metres at that
    # material's texture scale. Only when nothing stands below does the nearest aligned wall lend its material.
    continued=_continue_wall_below(scene,replacement,owner,sides,by_material)
    if references:
        kd=cKDTree(centers);normals=np.asarray(normals)
        for face in sides:
            if int(face) in continued:continue
            distance,indices=kd.query(replacement.triangles_center[face],k=min(24,len(references)))
            valid=[(d,int(i)) for d,i in zip(np.atleast_1d(distance),np.atleast_1d(indices)) if d<=3.0 and normals[int(i)]@replacement.face_normals[face]>.95]
            if not valid:unassigned.append(face);continue
            _,i=min(valid);ref,rf=references[i];q=replacement.triangles[face];source=ref.triangles[rf]
            bary=(q-source[0])@np.linalg.pinv((source[1:]-source[0]).T).T
            source_uv=ref.visual.uv[ref.faces[rf]];uv=source_uv[0]+bary@(source_uv[1:]-source_uv[0])
            key=id(ref.visual.material);slot=by_material.setdefault(key,{'material':ref.visual.material,'v':[],'uv':[]});slot['v'].extend(q);slot['uv'].extend(uv)
    else:unassigned=list(sides)
    # A vertical face of the roof solid with no wall beside it is still a wall of this building (the step
    # between two roof sections, the gable above an eave): it takes the building's dominant wall material,
    # mapped by metres at that material's texture scale, not the roof covering.
    given=0
    if unassigned and references:
        tally={}
        for ref,rf in references:tally.setdefault(id(ref.visual.material),[ref,rf,0])[2]+=1
        ref,rf,_=max(tally.values(),key=lambda t:t[2]);source=ref.triangles[rf];source_uv=ref.visual.uv[ref.faces[rf]]
        edges=np.linalg.norm(source[1:]-source[0],axis=1);uv_edges=np.linalg.norm(source_uv[1:]-source_uv[0],axis=1);scale=float(np.median(uv_edges[edges>1e-6]/edges[edges>1e-6])) if (edges>1e-6).any() else 1.
        if np.isfinite(scale) and scale>0:
            slot=by_material.setdefault(id(ref.visual.material),{'material':ref.visual.material,'v':[],'uv':[]})
            for face in unassigned:
                q=replacement.triangles[face];nf=replacement.face_normals[face];t=np.array([-nf[1],nf[0],0.]);t/=max(np.linalg.norm(t),1e-9)
                slot['v'].extend(q);slot['uv'].extend(np.c_[q@t*scale,q[:,2]*scale])
            given=len(unassigned);unassigned=[]
    top=np.where(np.abs(replacement.face_normals[:,2])>=.05)[0]
    ids=np.r_[top,np.asarray(unassigned,dtype=int)];parts=[_textured(replacement.submesh([ids],append=True,repair=False),roof_material)]
    for data in by_material.values():
        vertices=np.asarray(data['v']);parts.append(trimesh.Trimesh(vertices,np.arange(len(vertices)).reshape(-1,3),visual=TextureVisuals(uv=np.asarray(data['uv']),material=data['material']),process=False))
    return parts,{'sideFacesContinuingWallBelow':len(continued),'sideFacesInheritingAdjacentWallMaterial':len(sides)-len(unassigned)-given-len(continued),'sideFacesGivenBuildingWallMaterial':given,'sideFacesWithUnresolvedMaterial':len(unassigned)}

def clip_under_roof(mesh,region,model):
    """Trim triangles to the same roof plane while preserving material UVs.

    Triangles crossing the cap domain are split at that boundary; their outside
    pieces retain their original positions and texture coordinates.
    """
    if model['operation']!='min':raise ValueError('Clipping requires a convex plane envelope')
    tri=mesh.triangles;hi=evaluate(model,tri[:,:,:2].reshape(-1,2)).reshape(-1,3)
    exceeds=(tri[:,:,2]>hi+1e-5).any(axis=1)
    if not exceeds.any():return mesh,{'clippedTriangles':0,'ambiguousTriangles':0}
    near=((tri[:,:,:2].min(axis=1)<=np.array(region.bounds[2:])+.002)&(tri[:,:,:2].max(axis=1)>=np.array(region.bounds[:2])-.002)).all(axis=1)
    selected=np.where(exceeds&near)[0];owned=intersects_xy(region.buffer(.002),tri[selected,:,:2].reshape(-1,2)[:,0],tri[selected,:,:2].reshape(-1,2)[:,1]).reshape(-1,3)
    ambiguous=int((~owned.all(axis=1)).sum())
    if not len(selected):return mesh,{'clippedTriangles':0,'ambiguousTriangles':ambiguous}
    uv=mesh.visual.uv if hasattr(mesh.visual,'uv') and mesh.visual.uv is not None else np.zeros((len(mesh.vertices),2))
    vertices=[];coords=[];remove=np.zeros(len(mesh.faces),bool);remove[selected]=True
    for face in mesh.faces[selected]:
        original=np.c_[mesh.vertices[face],uv[face]]
        for poly,is_owned in _owned_pieces(original,region.buffer(.002)):
            if is_owned:
                for a,b,c in model['planes']:poly=_clip_poly(poly,np.array([-a,-b,1.]),c)
            if len(poly)>=3:
                for i in range(1,len(poly)-1):
                    t=poly[[0,i,i+1]]
                    if np.linalg.norm(np.cross(t[1,:3]-t[0,:3],t[2,:3]-t[0,:3]))>1e-9:vertices.extend(t[:,:3]);coords.extend(t[:,3:])
    remaining=mesh.submesh([np.where(~remove)[0]],append=True,repair=False) if (~remove).any() else trimesh.Trimesh()
    if vertices:
        extra=trimesh.Trimesh(np.asarray(vertices),np.arange(len(vertices)).reshape(-1,3),visual=TextureVisuals(uv=np.asarray(coords),material=mesh.visual.material),process=False)
        remaining=trimesh.util.concatenate([remaining,extra]) if len(remaining.faces) else extra
    remaining.metadata=mesh.metadata.copy()
    return remaining,{'clippedTriangles':len(selected),'ambiguousTriangles':0,'splitBoundaryTriangles':ambiguous}

def repair_scene(scene,cloud):
    report={'capsAttempted':0,'capsReplaced':0,'records':[],'inputFacadesReconstructed':False,'locationRules':0}
    staged=[];absorbed=[];protected={};policy=replace(RoofPolicy(),minimum_cells=18);caps_by_mesh={}
    roof_items=[(name,m) for name,m in scene.geometry.items() if m.metadata.get('layer')=='roofs']
    for name,mesh in roof_items:
        all_caps=sorted(caps(mesh),key=lambda hg:-hg[1].area);roof_area=sum(g.area for _,g in all_caps);accepted=[];caps_by_mesh[name]=all_caps
        if not all_caps:continue
        # The roof is read as a whole before its quantised steps: one sloped envelope that explains every
        # cap replaces them all, and the steps the pilot cut into that slope are trimmed under it.
        whole=unary_union([g for _,g in all_caps]);base_height=min(h for h,_ in all_caps)
        if len(all_caps)>1 and whole.area>=60:
            q=cloud[((cloud[:,:2]>=whole.bounds[:2])&(cloud[:,:2]<=whole.bounds[2:])).all(axis=1)]
            model,fit=fit_roof(whole,q,policy);report['capsAttempted']+=1
            record={'mesh':name,'capHeightM':base_height,'areaM2':whole.area,'roofAreaM2':roof_area,'wholeRoof':True,**fit,'applied':False}
            report['records'].append(record)
            if model and fit.get('sloped') and model['operation']=='min':
                vertices=np.concatenate([np.asarray(p.exterior.coords) for p in polygons(whole)]);predicted=evaluate(model,vertices)
                if base_height-float(predicted.min())>4:record['repairRejection']='incompatible-existing-roof-support'
                else:
                    bottom=min(base_height-.10,float(predicted.min())-.10)
                    keep=_unexplained_cells(whole,q,model,policy)
                    if keep is not None:protected[id(record)]=keep;record['protectedStructureM2']=float(keep.intersection(whole).area)
                    staged.append((name,mesh.metadata.get('featureId'),whole,None,model,roof_solid(whole,model,bottom),record));continue
        for height,g in all_caps:
            report['capsAttempted']+=1
            q=cloud[((cloud[:,:2]>=g.bounds[:2])&(cloud[:,:2]<=g.bounds[2:])).all(axis=1)]
            # A smaller cap whose returns an already accepted envelope of this roof explains is the same
            # roof (a quantised step of one slope); it is trimmed under that envelope, not fitted again.
            explained=None
            for prior in accepted:
                cells,_=sample_cells(g,q,policy)
                if len(cells) and np.quantile(np.abs(evaluate(prior,cells[:,:2])-cells[:,2]),.95)<=policy.maximum_p95_error_m:explained=prior;break
            if explained is not None:
                # The quantised step the pilot built for this part of the slope (its cap and riser walls)
                # is the same roof: everything of the building above the envelope inside this cap goes.
                absorbed.append((mesh.metadata.get('featureId'),g,explained,_unexplained_cells(g,q,explained,policy)))
                report['records'].append({'mesh':name,'capHeightM':height,'areaM2':g.area,'roofAreaM2':roof_area,'accepted':False,'status':'absorbed-by-accepted-envelope','applied':False});continue
            model,fit=fit_roof(g,q,policy)
            record={'mesh':name,'capHeightM':height,'areaM2':g.area,'roofAreaM2':roof_area,**fit,'applied':False}
            report['records'].append(record)
            if not model:
                # One envelope does not explain this cap: a roof of several slopes and steps. Each locally
                # validated sloped plane replaces its own part of the cap; the steps between them are the
                # solids' own sides, walls the returns actually show. Flat parts keep the cap.
                patches,preport=fit_patches(g,q,policy);record['patches']=preport
                for patch in patches:
                    pg=patch['geometry'].intersection(g);pm=patch['model'];grade=max(np.linalg.norm(pl[:2]) for pl in pm['planes'])
                    # A roof patch is a surface with a roof's pitch. A strip of cells a cell or two wide with a
                    # steep plane through it is a wall face the sampler saw from above, not a roof.
                    if pg.area<max(40.,.1*g.area) or not (.05<=grade<=.7) or pg.area/max(pg.length/2,1e-9)<3.:continue
                    vertices=np.concatenate([np.asarray(p.exterior.coords) for p in polygons(pg)]);predicted=evaluate(pm,vertices)
                    if height-float(predicted.min())>4:continue
                    bottom=min(height-.10,float(predicted.min())-.10)
                    try:solid=roof_solid(pg,pm,bottom)
                    except ValueError:continue
                    prec={'mesh':name,'capHeightM':height,'areaM2':pg.area,'roofAreaM2':roof_area,'patchOfCapM2':g.area,'status':'held-out-plane-patch','heldOutP95M':patch['heldOutP95M'],
                          'explainedInterior':patch.get('explainedInterior'),'explainedUnderOutline':patch.get('explainedUnderOutline'),'sloped':True,'accepted':True,'applied':False}
                    if patch.get('unexplained') is not None:protected[id(prec)]=patch['unexplained'];prec['protectedStructureM2']=float(patch['unexplained'].intersection(pg).area)
                    report['records'].append(prec);staged.append((name,mesh.metadata.get('featureId'),pg,height,pm,solid,prec))
                continue
            if not fit.get('sloped') or model['operation']!='min':continue
            # A sloped envelope is a roof only when it explains a substantial part of this roof. A small,
            # high cap that sloped planes happen to fit is rooftop equipment or signage, not a pitched roof.
            if g.area<max(60.,.25*roof_area):
                record['repairRejection']='sloped-fit-on-minor-cap';continue
            vertices=np.concatenate([np.asarray(p.exterior.coords) for p in polygons(g)])
            predicted=evaluate(model,vertices)
            # A slope rising above its flat cap is the roof being restored; a plane dropping far below the
            # cap would carve into support the returns did not measure.
            if height-float(predicted.min())>4:
                record['repairRejection']='incompatible-existing-roof-support';continue
            accepted.append(model)
            bottom=min(height-.10,float(predicted.min())-.10)
            replacement=roof_solid(g,model,bottom)
            keep=_unexplained_cells(g,q,model,policy)
            if keep is not None:protected[id(record)]=keep;record['protectedStructureM2']=float(keep.intersection(g).area)
            staged.append((name,mesh.metadata.get('featureId'),g,height,model,replacement,record))
    for index,(name,fid,g,height,model,replacement,record) in enumerate(staged):
        source=scene.geometry[name];tri=source.triangles
        # A cap triangle belongs to the replaced region when its plan outline meets it; a vertex test missed the
        # large triangles of an earcut cap that cover a small patch without a vertex inside it, and the patch was
        # then refused as 'ambiguous-cap-ownership' while its neighbours were applied, leaving a step between them.
        from shapely import polygons as _polygons,intersects as _intersects
        touching=_intersects(_polygons(tri[:,:,:2]),g.buffer(.002))
        # A whole-roof envelope (height None) takes every cap and step inside it; a cap fit takes only its own.
        at_old_cap=(source.face_normals[:,2]>.99) if height is None else ((tri[:,:,2].max(axis=1)<=height+.002)&(tri[:,:,2].min(axis=1)>=height-.105))
        selected=touching&at_old_cap
        if not selected.any():record['repairRejection']='ambiguous-cap-ownership';continue
        # Cap triangles are split at the replacement boundary; only their part inside it is removed.
        uv=source.visual.uv if hasattr(source.visual,'uv') and source.visual.uv is not None else np.zeros((len(source.vertices),2))
        vertices=[];coords=[]
        for face in source.faces[np.where(selected)[0]]:
            for poly,is_owned in _owned_pieces(np.c_[source.vertices[face],uv[face]],g.buffer(.002)):
                if is_owned or len(poly)<3:continue
                for i in range(1,len(poly)-1):
                    t=poly[[0,i,i+1]]
                    if np.linalg.norm(np.cross(t[1,:3]-t[0,:3],t[2,:3]-t[0,:3]))>1e-9:vertices.extend(t[:,:3]);coords.extend(t[:,3:])
        kept=source.submesh([np.where(~selected)[0]],append=True,repair=False) if (~selected).any() else trimesh.Trimesh()
        if vertices:
            extra=trimesh.Trimesh(np.asarray(vertices),np.arange(len(vertices)).reshape(-1,3),visual=TextureVisuals(uv=np.asarray(coords),material=source.visual.material),process=False)
            kept=trimesh.util.concatenate([kept,extra]) if len(kept.faces) else extra
        kept.metadata=source.metadata.copy();scene.geometry[name]=kept
        new_parts,material_report=_inherit_wall_sides(scene,replacement,fid,source.visual.material)
        for part,new in enumerate(new_parts):
            new.metadata={**source.metadata,'evidence':'held-out-lidar-roof-plane'}
            scene.add_geometry(new,geom_name=f'roof_plane_{index}_{part}',node_name=f'roof_plane_{index}_{part}')
        clipped=ambiguous=0
        # Under cells the plane did not explain a measured structure stands; it is not trimmed.
        clip_region=g.difference(protected[id(record)]) if id(record) in protected else g
        if height is not None:
            # A higher roof cap of this building beside the replaced one is a measured structure this plane does
            # not explain (were it explained, it would have been absorbed into the envelope): its footprint, with
            # the riser walls standing on its edge, is never trimmed under a plane fitted to a lower cap. Before,
            # a gable fitted to a low front cap cut away the risers of the taller block behind it and left that
            # block's roof floating on wall slivers.
            higher=[cg for ch,cg in caps_by_mesh.get(name,[]) if ch>height+.5 and cg.intersects(g.buffer(.5))]
            if higher:clip_region=clip_region.difference(unary_union(higher).buffer(.3))
        for wallname,wall in list(scene.geometry.items()):
            # Everything of this building that stands above the accepted roof envelope is trimmed to it,
            # including stepped roof tiers that the same slope had produced as a staircase.
            if wall.metadata.get('featureId')!=fid or wall.metadata.get('layer') not in {'buildings','facades','roofs'}:continue
            if wall.metadata.get('evidence')=='held-out-lidar-roof-plane':continue   # the replacement itself lies on the plane
            if clip_region.is_empty:continue
            changed,check=clip_under_roof(wall,clip_region,model);scene.geometry[wallname]=changed
            clipped+=check['clippedTriangles'];ambiguous+=check['ambiguousTriangles']
        record.update(applied=True,model=model,**material_report,clippedWallTriangles=clipped,ambiguousBoundaryTriangles=ambiguous,
                      replacementWatertight=bool(replacement.is_watertight),replacementVolumeM3=float(replacement.volume))
        report['capsReplaced']+=1
    for fid,g,model,keep in absorbed:
        region=g.difference(keep) if keep is not None else g
        if region.is_empty:continue
        for wallname,wall in list(scene.geometry.items()):
            if wall.metadata.get('featureId')!=fid or wall.metadata.get('layer') not in {'buildings','facades','roofs'} or wall.metadata.get('evidence')=='held-out-lidar-roof-plane':continue
            changed,_=clip_under_roof(wall,region,model);scene.geometry[wallname]=changed
    report['absorbedCaps']=len(absorbed)
    report['buildingsChanged']=len({r['mesh'] for r in report['records'] if r['applied']})
    return report
