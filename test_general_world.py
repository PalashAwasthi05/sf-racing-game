"""Identity-free adversarial fixtures. Checks cannot substitute for image validation."""
import sys,json,copy
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,box,LineString,Point
from shapely.affinity import rotate,translate
sys.path.insert(0,str(Path(__file__).parent))
from general_world import geometry,evidence,objects,scheduler
from general_world.policy import Policy
p=Policy();passed=[]
# A courtyard stays empty even though the shell must be a closed positive solid.
g=Polygon([(0,0),(20,0),(20,16),(0,16)],holes=[[(7,5),(13,5),(13,11),(7,11)]])
flat=lambda x,y:np.zeros(np.broadcast_arrays(x,y)[0].shape)
m,r=geometry.reconstruct_building(g,np.empty((0,3)),flat,p,12)
assert m.is_volume and m.is_watertight and abs(m.volume-g.area*12.3)<1e-4
# No roof or floor triangle crosses the footprint's courtyard void.
for t,n in zip(m.triangles,m.face_normals):
 if abs(n[2])>.99:assert not Polygon(t[:,:2]).contains(Point(10,8))
passed.append('closed shell retains courtyard void; no invisible cap across gap')
# Frames cannot join the separated arms of a notched frontage.
notch=Polygon([(0,0),(8,0),(8,6),(14,6),(14,0),(22,0),(22,18),(0,18)])
frames=evidence.contiguous_frames(notch)
assert all(evidence.validate_frame(f,notch) for f in frames)
assert not evidence.validate_frame({'a':[0,0],'b':[22,0]},notch)
assert not evidence.validate_frame({'a':[0,-4],'b':[22,-4]},notch)
passed.append('disconnected and displaced facade frames rejected')
assert evidence.unique_links([('x','y'),('y','x')])==[('x','y')];passed.append('reciprocal joins processed once')
# Equivalent geometry at an arbitrary pose remains congruent; identities are not inputs.
angle=37;off=np.array([71.3,-23.7]);gg=translate(rotate(g,angle,origin=(0,0)),*off)
mm,rr=geometry.reconstruct_building(gg,np.empty((0,3)),flat,p,12)
assert abs(mm.volume-m.volume)<.002 and rr['authoredProfilesUsed']==0
passed.append('translation/rotation replay with no identity input')
# Supported roof islands must never dilate across an unmeasured channel.
xy=np.array([(x,y) for x in np.arange(.12,22,.3) for y in np.arange(.12,18,.3)]);z=np.where((xy[:,0]<8)|(xy[:,0]>14),14,8);pts=np.c_[xy,z]
layers,report=geometry.supported_roof_layers(box(0,0,22,18),pts,flat,Policy(roof_cell_m=1,minimum_roof_samples=3))
assert layers and report['unsupportedCellsFilled']==0
assert all(not region.contains(Point(11,9)) for region,height in layers)
passed.append('roof islands cannot bridge low or unobserved channel')
# The same no-image policy applies to formerly privileged and anonymous metadata.
record={'label':'window','confidence':.99,'producer':{'mode':'automatic','inputHash':'abc'},'sourceIds':[]}
assert not evidence.inspect_record(record,{},p,'2026-01-01')['accepted']
assert not evidence.inspect_record({**record,'manualOverride':True},{},p,'2026-01-01')['accepted']
assert evidence.contains_authored_override({'producer':{'extra':{'manualOverride':True}}})
passed.append('no-image and nested authored overrides cannot invent windows')
# Monotone steps pass; midflight obstruction must not become a floating/buried staircase.
line=LineString([(0,0),(4,0)]);slope=lambda x,y:np.asarray(x)*.2
steps,s=geometry.supported_stairs(line,2,8,slope,rail=True);assert s['accepted'] and sum(m.metadata.get('component')=='tread' for m in steps)==8 and np.allclose(np.array(s['handrailDatums'])-s['treadTops'],1)
bump=lambda x,y:np.asarray(x)*.2+np.where((np.asarray(x)>1)&(np.asarray(x)<3),2,0)
steps,s=geometry.supported_stairs(line,2,8,bump,rail=True);assert not s['accepted'] and not steps
passed.append('stair profile conflict rejected; rail references tread levels')
# Control nodes do not assert physical poles; missing sign artwork cannot become another sign.
a={'observedKind':'sign','position':[0,0,2],'orientation':[1,0,0],'supportKey':'pole','sourceIds':['s'],'positionUncertaintyM':.1}
assert not objects.admit_object(a)['accepted']
assert not objects.admit_object({**a,'artwork':'captured','semanticRole':'control-point'})['accepted']
b={**a,'position':[6,0,2]};assert not objects.same_physical_object(a,b)
c={**a,'supportKey':'opposite-curb','position':[.1,0,2]};assert not objects.same_physical_object(a,c)
passed.append('no substitute sign content; distinct physical signs retained')
assert objects.leaf_is_supported([.5,0,2],[0,0,2],[1,0,2],lambda x,y:0,[])
assert not objects.leaf_is_supported([.5,1,2],[0,0,2],[1,0,2],lambda x,y:0,[])
passed.append('detached foliage rejected by support graph')
# Previous attention cannot bypass first-pass coverage, even with enormous salience.
ss=[{'key':'old','attempts':3,'visibleArea':10000},{'key':'new','attempts':0,'visibleArea':1}];cc=[{'key':'a','covers':['old'],'routeOrder':0},{'key':'b','covers':['new'],'routeOrder':1}]
chosen,state=scheduler.allocate(ss,cc,1);assert chosen[0]['key']=='b'
renamed=copy.deepcopy(ss);renamed[0]['name']='Landmark';renamed[1]['name']='Ordinary';assert scheduler.allocate(renamed,cc,1)[0]==chosen
passed.append('unserved surfaces precede extra attention; names do not change schedule')
report={'passed':passed,'testCount':len(passed),'independentPhotographicAccuracy':False,'noLocationSpecificFixtures':True}
Path('outputs').mkdir(exist_ok=True)
Path('outputs/general-world-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
