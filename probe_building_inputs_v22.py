"""Export measured inputs from the active adapter for building-level roof validation."""
from pathlib import Path
import sys
source=Path(__file__).with_name('generate_detailed_v14.py')
namespace={'__name__':'__building_input_probe__','__file__':str(source)}
code=source.read_text().split('for c in ([] if _args.streets_only else candidates):',1)[0]
exec(compile(code,str(source),'exec'),namespace)
import numpy as np,json
from shapely.geometry import mapping
out=namespace['OUT'];records=[];arrays={}
for c in namespace['candidates']:
 g=c['g'];p=namespace['get_points'](g.buffer(-.65));z0=float(namespace['elev'](g.centroid.x,g.centroid.y));p=p[p[:,2]>z0+3]
 arrays[c['id']]=p
 records.append({'id':c['id'],'geometry':mapping(g),'groundZ':z0,'pointCount':len(p),'tags':c['osm']['tags'] if c['osm'] else {}})
np.savez_compressed(out/'building-roof-points.npz',**arrays)
(out/'building-roof-inputs.json').write_text(json.dumps({'buildings':records,'origin':{'lat':namespace['lat'],'lon':namespace['lon']},'verticalOrigin':namespace['base']},indent=2))
print('Saved',len(records),'building inputs',flush=True)
