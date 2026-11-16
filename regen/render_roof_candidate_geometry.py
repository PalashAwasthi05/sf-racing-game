"""Source-camera flat-shaded geometry comparison, not a browser/material acceptance."""
import argparse,json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import trimesh
from regen.propose_roof_boundaries import project_target
from regen.check_roof_sky_projection import raster


def render(scene,identity,projection,size):
    image=np.full((size[1],size[0],3),[218,225,231],np.uint8);depth=np.full((size[1],size[0]),np.inf,np.float32)
    light=np.array([-.3,-.5,1.]);light/=np.linalg.norm(light)
    for name,mesh in scene.geometry.items():
        if not any(name.startswith(role+'__'+identity+'__') for role in ('buildings','facades','roofs','foundations')):continue
        dd,faces=raster(mesh,projection,size);use=dd<depth
        if not use.any():continue
        color=np.array([178.,178.,172.]);hexes=re.findall(r'#([0-9a-fA-F]{6})',name)
        if hexes:color=np.array([int(hexes[-1][i:i+2],16) for i in (0,2,4)],float)
        elif 'glass' in name:color=np.array([80.,113.,128.])
        elif 'roof' in name:color=np.array([75.,76.,77.])
        shade=.55+.45*np.maximum(0,np.asarray(mesh.face_normals)@light)
        image[use]=np.clip(color[None,:]*shade[faces[use],None],0,255).astype(np.uint8);depth[use]=dd[use]
    return Image.fromarray(image)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('before','after','jobs','metadata','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--rows',type=int,nargs='+',required=True);p.add_argument('--before-label',default='r28 actual geometry');p.add_argument('--after-label',default='external provisional geometry');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(a.jobs.read_text());meta=json.loads(a.metadata.read_text())['pieces'];scenes=[trimesh.load(path,force='scene',process=False) for path in (a.before,a.after)]
    for row in a.rows:
        job=jobs[row];piece=next(p for p in meta if p['id']==job['side']);height=round(job['height']*.64);size=(640,height);projection=project_target(job,piece,size)
        photo=Image.open(job['persp']).convert('RGB').resize(size)
        images=[photo,*[render(s,job['building'],projection,size) for s in scenes]]
        sheet=Image.new('RGB',(1920,height+48),'white');draw=ImageDraw.Draw(sheet)
        for i,(image,title) in enumerate(zip(images,['source photograph',a.before_label,a.after_label])):
            sheet.paste(image,(i*640,48));draw.text((i*640+8,8),f'row{row:02d}: {title}',fill='black')
        draw.text((8,28),'Flat-shaded source-camera shape check; PBR/shadows/context omitted. External proposal, not acceptance.',fill='black')
        sheet.save(a.output/f'{row:02d}-source-before-after.jpg',quality=92)
        images[1].save(a.output/f'{row:02d}-before.jpg');images[2].save(a.output/f'{row:02d}-after.jpg')
        print(row,flush=True)


if __name__=='__main__':main()
