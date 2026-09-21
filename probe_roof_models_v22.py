"""Compare whole-building measured plane fits against the existing tier approach."""
import json,time
from pathlib import Path
import numpy as np
from shapely.geometry import shape,mapping
from general_world.roof_planes import fit_roof,fit_patches,RoofPolicy
root=Path('work/building-v22-inputs');data=json.loads((root/'building-roof-inputs.json').read_text());cloud=np.load(root/'building-roof-points.npz');out=Path('work/building-v22-models');out.mkdir(exist_ok=True);reports=[]
policy=RoofPolicy(cell_m=1.5)
for b in data['buildings']:
 start=time.time();g=shape(b['geometry']);points=cloud[b['id']];model,report=fit_roof(g,points,policy);patches=[]
 if model:patches=[{'geometry':g,'model':model,'heldOutP95M':report['heldOutP95M']}]
 else:patches,pr=fit_patches(g,points,policy);report['localFits']=pr
 serial=[{k:mapping(v) if hasattr(v,'__geo_interface__') else v for k,v in p.items()} for p in patches]
 item={'id':b['id'],'areaM2':g.area,'patchAreaM2':sum(p['geometry'].area for p in patches),'patches':serial,'report':report,'seconds':time.time()-start}
 (out/(b['id']+'.json')).write_text(json.dumps(item));reports.append({k:v for k,v in item.items() if k!='patches'});print(b['id'],len(patches),round(item['patchAreaM2']/g.area,2),round(item['seconds'],2),flush=True)
(out/'report.json').write_text(json.dumps(reports,indent=2))
