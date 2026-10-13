import math
import numpy as np
import trimesh
from shapely.geometry import Point

def compile(ctx):
 positions=[];put=ctx['put'];cyl=ctx['cylinder'];count=0
 def blade(a,b,width,key):
  a,b=np.array(a),np.array(b);v=b-a;n=np.cross(v,[0,0,1]);n=n/max(.001,np.linalg.norm(n));mid=a+v*.55
  vv=np.array([a,mid+n*width,b,mid-n*width,mid+[0,0,width*.2]])
  put(trimesh.Trimesh(vv,[[0,1,4],[1,2,4],[2,3,4],[3,0,4]],process=False),'vegetation',key)
 for item in ctx['tree_placements']:
  t=item['record'];x,y=item['position'];positions.append((x,y))
  z=float(ctx['elev'](x,y))+.095;seed=int(float(t.get('treeid',0)));rng=np.random.default_rng(seed)
  palm=any(s in t.get('qspecies','').lower() for s in ['palm','phoenix','washingtonia'])
  h=9+rng.random()*3 if palm else 5+rng.random()*3
  radius=item['radius'];cyl([x,y,z],[x,y,z+h],radius,'vegetation','bark',12)
  if palm:
   for zh in np.arange(.3,h-.3,.27):cyl([x,y,z+zh],[x,y,z+zh+.045],radius+.015,'vegetation','bark',10)
   for k in range(14):
    ang=k*math.tau/14+rng.uniform(-.1,.1);d=np.array([math.cos(ang),math.sin(ang),0]);side=np.array([-d[1],d[0],0]);root=np.array([x,y,z+h]);length=rng.uniform(2.9,4.)
    previous=root
    for s in np.linspace(.08,1,18):
     mid=root+d*length*s+np.array([0,0,1.5*math.sin(s*math.pi)-1.0*s]);cyl(previous,mid,.025*(1-s)+.012,'vegetation','leaf1',5);previous=mid
     for sign in [-1,1]:
      tip=mid+side*sign*(.8*math.sin(s*math.pi)+.08)+d*.13+np.array([0,0,-.16]);blade(mid,tip,.065,'leaf'+str(1+(k%3)))
  else:
   for k in range(7):
    ang=k*math.tau/7;d=np.array([math.cos(ang),math.sin(ang),0]);root=np.array([x,y,z+h*.6]);tip=root+d*rng.uniform(1.3,2.3)+[0,0,rng.uniform(1.,2.6)];cyl(root,tip,.05,'vegetation','bark',6)
   for k in range(270):
    d=rng.normal(size=3);d/=np.linalg.norm(d);r=rng.random()**.3;pos=np.array([x,y,z+h])+d*r*np.array([2.4,2.4,2.8]);ang=rng.uniform(0,math.tau);end=pos+np.array([math.cos(ang),math.sin(ang),rng.uniform(-.4,.4)])*.42;blade(pos,end,.15,'leaf'+str(1+k%3))
  count+=1
 ctx['counts']['trees']=count
