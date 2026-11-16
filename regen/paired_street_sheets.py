"""Assemble actual photos and same-pose viewer captures without stretching either."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
ap=argparse.ArgumentParser();ap.add_argument('jobs',type=Path);ap.add_argument('directory',type=Path);ap.add_argument('revisions',nargs='+');a=ap.parse_args();jobs=json.loads(a.jobs.read_text());w=335;h=242
for start in range(0,len(jobs),4):
    s=Image.new('RGB',(w*(1+len(a.revisions)),4*(h+24)),(239,237,230));d=ImageDraw.Draw(s)
    for i,j in enumerate(jobs[start:start+4],start):
        for c,(f,label) in enumerate([(Path(j['persp']),'Street photo')]+[(a.directory/f'{r}-{i}.png',r) for r in a.revisions]):
            im=ImageOps.contain(Image.open(f).convert('RGB'),(w-3,h));y=(i-start)*(h+24);d.text((c*w+3,y+2),f'{i+1}   {label}',fill=(30,30,35));s.paste(im,(c*w+(w-im.width)//2,y+18+(h-im.height)//2))
    s.save(a.directory/('compare-'+'-'.join(a.revisions)+f'-{start//4+1}.jpg'),quality=92)
