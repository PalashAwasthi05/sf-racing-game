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
  sp=t.get('qspecies','').lower();palm=any(s in sp for s in ['palm','phoenix','washingtonia'])
  # Fan palms (Washingtonia, 'fan palm'): tall slender trunks under a dense round crown of palmate leaves over a skirt of
  # dead ones. Drawn as feather palms they read as thin spidery ribs at a third of their height.
  fan=palm and any(s in sp for s in ['washingtonia','fan palm'])
  h=14+rng.random()*4 if fan else 9+rng.random()*3 if palm else 5+rng.random()*3
  radius=item['radius'];cyl([x,y,z],[x,y,z+h],radius,'vegetation','bark',12)
  # A palm's fronds are their own material: the viewer grows broadleaf leaf-cluster cards on 'leaf' quads, and a row of
  # fan palms rendered as flat green umbrellas.
  if fan:
   for zh in np.arange(.3,h-2.2,.35):cyl([x,y,z+zh],[x,y,z+zh+.05],radius+.012,'vegetation','bark',10)
   def fan_leaf(base,axis,up,rad,material='frond'):
    # a palmate leaf: 13 narrow leaflets radiating over 160 degrees in the plane of axis and up, their tips drooping, with
    # gaps between them (a solid fan read as a green disc), drawn from both sides
    V=[];F=[];droop=np.cross(axis,up);droop=droop if droop[2]<0 else -droop
    for a_ in np.linspace(math.radians(-80),math.radians(80),13):
     r=math.cos(a_)*axis+math.sin(a_)*up;w=math.sin(math.radians(4.5))*rad*.5;side=math.cos(a_)*up-math.sin(a_)*axis
     root=base+r*.12;tip=base+r*rad+droop*rad*.28*(1-math.cos(a_)*.3)
     i=len(V);V+=[root-side*w*.3,root+side*w*.3,base+r*rad*.62+side*w+droop*rad*.07,base+r*rad*.62-side*w+droop*rad*.07,tip]
     F+=[[i,i+1,i+2],[i,i+2,i+3],[i+3,i+2,i+4]];F+=[[f[0],f[2],f[1]] for f in F[-3:]]
    put(trimesh.Trimesh(np.array(V),F,process=False),'vegetation',material)
   crown=np.array([x,y,z+h])
   for k in range(36):
    ang=k*math.tau/36*2.4+rng.uniform(-.12,.12);elev=math.radians(rng.uniform(-25,70));d=np.array([math.cos(ang)*math.cos(elev),math.sin(ang)*math.cos(elev),math.sin(elev)])
    base=crown+d*rng.uniform(1.1,1.6);cyl(crown,base,.035,'vegetation','frond',5)
    side=np.array([-math.sin(ang),math.cos(ang),0.])
    fan_leaf(base,d,side,rng.uniform(1.1,1.45))
   # the skirt: dead leaves hanging flat against the trunk under the crown
   for k in range(14):
    ang=k*math.tau/14+rng.uniform(-.1,.1);out=np.array([math.cos(ang),math.sin(ang),0.]);side=np.array([-out[1],out[0],0.])
    base=crown+out*(radius+.08)+np.array([0,0,-rng.uniform(.1,.6)]);down=np.array([0,0,-1.])*.97+out*.24
    fan_leaf(base,down/np.linalg.norm(down),side,rng.uniform(.8,1.2),'thatch')
  elif palm:
   for zh in np.arange(.3,h-.3,.27):cyl([x,y,z+zh],[x,y,z+zh+.045],radius+.015,'vegetation','bark',10)
   for k in range(14):
    ang=k*math.tau/14+rng.uniform(-.1,.1);d=np.array([math.cos(ang),math.sin(ang),0]);side=np.array([-d[1],d[0],0]);root=np.array([x,y,z+h]);length=rng.uniform(2.9,4.)
    previous=root
    for s in np.linspace(.08,1,18):
     mid=root+d*length*s+np.array([0,0,1.5*math.sin(s*math.pi)-1.0*s]);cyl(previous,mid,.025*(1-s)+.012,'vegetation','frond',5);previous=mid
     for sign in [-1,1]:
      tip=mid+side*sign*(.8*math.sin(s*math.pi)+.08)+d*.13+np.array([0,0,-.16]);blade(mid,tip,.065,'frond')
  else:
   for k in range(7):
    ang=k*math.tau/7;d=np.array([math.cos(ang),math.sin(ang),0]);root=np.array([x,y,z+h*.6]);tip=root+d*rng.uniform(1.3,2.3)+[0,0,rng.uniform(1.,2.6)];cyl(root,tip,.05,'vegetation','bark',6)
   for k in range(270):
    d=rng.normal(size=3);d/=np.linalg.norm(d);r=rng.random()**.3;pos=np.array([x,y,z+h])+d*r*np.array([2.4,2.4,2.8]);ang=rng.uniform(0,math.tau);end=pos+np.array([math.cos(ang),math.sin(ang),rng.uniform(-.4,.4)])*.42;blade(pos,end,.15,'leaf'+str(1+k%3))
  count+=1
 ctx['counts']['trees']=count
