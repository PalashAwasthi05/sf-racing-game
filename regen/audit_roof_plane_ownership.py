"""Read-only native plane inlier ownership and exact source-camera overlays."""
import argparse,json,types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,Point,shape
from shapely import contains_xy
from shapely.ops import transform
from pyproj import Transformer
from PIL import Image,ImageDraw
from pipeline import roof_shell,spatial_rules
from regen.rebuild_roof_support_review import parameters
from regen.propose_roof_boundaries import project_target
from regen.check_roof_sky_projection import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('area','build','profiles','observations','jobs','poses','metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',nargs='+',type=int,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    masses=json.loads((a.build/'massing-dump.json').read_text());jobs=json.loads(a.jobs.read_text());poses=json.loads(a.poses.read_text())['poses'];profiles=json.loads(a.profiles.read_text())['profiles'];obs=json.loads(a.observations.read_text());obs=obs.get('observations',obs);meta=json.loads(a.metadata.read_text())['pieces'];clouds=np.load(a.build/'rooftop-structure-returns.npz');world=json.loads((a.area/'world.json').read_text());to=Transformer.from_crs(4326,32610,always_xy=True);ox,oy=to.transform(world['center']['lon'],world['center']['lat']);neighbors_data=json.loads((a.area/'data/buildings.json').read_text())
    def local(x,y,z=None):
        xx,yy=to.transform(x,y);return np.asarray(xx)-ox,np.asarray(yy)-oy
    results=[];colors=[(255,125,0),(180,35,255),(20,190,100)]
    for row in a.rows:
        identity=jobs[row]['building'];mass=next(m for m in masses if m['id']==identity);g=Polygon(mass['outline'][0]);ctx=clouds[identity+'__context'];raw=ctx[contains_xy(g,ctx[:,0],ctx[:,1])&(ctx[:,2]>mass['z0']+3)];x0,y0,x1,y1=g.bounds;nx=max(2,int(np.ceil((x1-x0)/1.5)));ny=max(2,int(np.ceil((y1-y0)/1.5)));inner=raw[contains_xy(g.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,x0,y1,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,x0,y1,1.5,nx,ny);boundary=raw[keep];constraints,host=parameters(obs,profiles,identity,mass['z0']);capture={}
        module=types.ModuleType('pipeline._plane_ownership_audit');module.__package__='pipeline';exec(compile(Path(roof_shell.__file__).read_text(),str(roof_shell.__file__),'exec'),module.__dict__);original=module._fit_structural_regions
        def fit(footprint,points,*args,**kwargs):
            regions,audit=original(footprint,points,*args,**kwargs);capture.update(regions=regions,points=points.copy(),fit=audit);return regions,audit
        module._fit_structural_regions=fit;module.build(g,inner,mass['baseRoofZ'],boundary_points=boundary,eave_constraints=constraints,host_spec=host)
        piece=next(q for q in meta if q['id']==jobs[row]['side']);n=np.array(piece['normal']);n/=np.linalg.norm(n)
        neighbors=[]
        for item in neighbors_data:
            geometry=transform(local,shape(item['shape']))
            if geometry.distance(g)<12:neighbors.append({'id':'sf_'+item['sf16_bldgid'].replace('.','_'),'geometry':geometry,'sourceMedianHeightM':float(item['hgt_median_m'])})
        regions=[]
        for index,region in enumerate(capture['regions'] or []):
            points=capture['points'];keep=contains_xy(region['geometry'].buffer(.001),points[:,0],points[:,1])&(abs(np.c_[points[:,:2],np.ones(len(points))]@region['plane']-points[:,2])<=.22);q=points[keep]
            regions.append({'region':index,'plane':region['plane'].tolist(),'directSupportGeometryWkt':region['geometry'].wkt,'areaM2':region['geometry'].area,'directCells':len(region['inlierXY']),'rawInlierCount':len(q),'rawInlierXYZ':q.tolist(),'inlierCellXY':region['inlierXY'].tolist(),'heightRangeM':np.percentile(q[:,2]-mass['z0'],[0,10,50,90,100]).tolist(),'inwardDepthRangeM':np.percentile(piece['planeV']-q[:,:2]@n,[0,50,100]).tolist(),'footprintOwnership':[{'id':other['id'],'inlierPointsInside':int(contains_xy(other['geometry'],q[:,0],q[:,1]).sum()),'sourceMedianHeightM':other['sourceMedianHeightM']} for other in neighbors if contains_xy(other['geometry'],q[:,0],q[:,1]).any()]})
        views=[]
        for v,job in enumerate(j for j in poses if j['building']==identity):
            image=Image.open(job['persp']).convert('RGB');draw=ImageDraw.Draw(image);projection=project_target(job,piece,image.size);basis=np.array(projection['basisRightDownForward']);camera=np.array(projection['cameraXYZ']);projected=[]
            for region in regions:
                q=np.array(region['rawInlierXYZ']);directions=(q-camera)@basis.T;pixels=directions[:,:2]/directions[:,2,None]*projection['focalPx']+np.array(image.size)/2-.5
                for xyz,px,depth in zip(q,pixels,directions[:,2]):
                    if depth>0 and 0<=px[0]<image.width and 0<=px[1]<image.height:
                        x,y=px;draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[region['region']%3]);projected.append({'region':region['region'],'pointXYZ':xyz.tolist(),'pixelPx':px.tolist()})
            draw.rectangle((0,0,min(image.width,950),35),fill='white');draw.text((5,3),'Actual direct native plane inliers: orange=0 purple=1. Conditional camera placement.',fill='black');draw.text((5,18),'Projection alone does not establish visibility, material or ownership.',fill='black');image.save(a.output/f'{row:02d}-view-{v}-inliers.jpg',quality=94);views.append({'image':job['image'],'sourcePath':job['persp'],'sourceSha256':sha(job['persp']),'cameraXYZ':camera.tolist(),'projection':projection,'projectedInliers':projected})
        # Source footprint and return map with metric coordinate ticks.
        size=(1000,850);image=Image.new('RGB',size,'white');draw=ImageDraw.Draw(image);bounds=(x0-16,y0-10,x1+10,y1+10);scale=min(900/(bounds[2]-bounds[0]),730/(bounds[3]-bounds[1]))
        def px(point):return (50+(point[0]-bounds[0])*scale,790-(point[1]-bounds[1])*scale)
        for other in neighbors:
            for poly in ([other['geometry']] if other['geometry'].geom_type=='Polygon' else other['geometry'].geoms):draw.polygon([px(p) for p in poly.exterior.coords],fill='#eeeeee',outline='#888888')
            draw.text(px(other['geometry'].centroid.coords[0]),other['id'].removeprefix('sf_201006_'),fill='#777777')
        draw.line([px(p) for p in g.exterior.coords],fill='red',width=3)
        for region in regions:
            for q in region['rawInlierXYZ']:
                x,y=px(q);draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[region['region']%3])
        for view in views:
            x,y=px(view['cameraXYZ']);draw.polygon([(x,y-6),(x-5,y+4),(x+5,y+4)],fill='blue');draw.text((x+5,y),view['image'][-5:],fill='blue')
        for x in np.arange(np.ceil(bounds[0]/5)*5,bounds[2],5):
            xx,yy=px([x,bounds[1]]);draw.text((xx,yy+8),f'{x:.0f}',fill='black')
        for y in np.arange(np.ceil(bounds[1]/5)*5,bounds[3],5):
            xx,yy=px([bounds[0],y]);draw.text((5,yy),f'{y:.0f}',fill='black')
        draw.text((10,10),f'row{row:02d} direct inliers / generated footprint(red) / municipal neighbors(gray). Local metres.',fill='black');image.save(a.output/f'{row:02d}-xy-ownership.png')
        result={'row':row,'building':identity,'generatedFootprintWkt':g.wkt,'sourceGroundZ':piece['groundZ'],'buildingGroundZ':mass['z0'],'datumDifferenceM':piece['groundZ']-mass['z0'],'regions':regions,'views':views,'nativeFit':capture['fit'],'neighborFootprints':[{'id':q['id'],'wkt':q['geometry'].wkt,'sourceMedianHeightM':q['sourceMedianHeightM']} for q in neighbors]};results.append(result)
        print(json.dumps({'row':row,'regions':[{k:r[k] for k in ('region','directCells','rawInlierCount','heightRangeM','inwardDepthRangeM','footprintOwnership')} for r in regions]}),flush=True)
    (a.output/'audit.json').write_text(json.dumps({'status':'read-only-actual-native-plane-inlier-audit','rows':results,'productionChanges':0,'roofRuleSha256':sha(roof_shell.__file__),'qualification':'Classification/material/visibility remain unresolved unless independently inspected; point count and a plane fit are not surface semantics.'},indent=2)+'\n')


if __name__=='__main__':main()
