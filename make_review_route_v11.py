from pathlib import Path
import json
from shapely.geometry import shape,Point
from shapely.ops import transform,unary_union
from pyproj import Transformer
from general_world.review_route import review_routes
w=Path(__file__).resolve().parent;d=json.loads((w/'normalized-v9/input.json').read_text());f=d['coordinateFrame'];t=Transformer.from_crs('EPSG:4326',f['projectedCrs'],always_xy=True);ox,oy=f['originProjectedM']
def project(x,y,z=None):
 import numpy as np
 a,b=t.transform(x,y);return np.asarray(a)-ox,np.asarray(b)-oy
lines=[transform(project,shape(r['line'])) for r in json.loads((w/'data/streets.json').read_text()) if r.get('active') is True]
footprints=unary_union([shape(b['geometry']).buffer(.7) for b in d['buildings']])
routes=review_routes(lines,Point(0,0).buffer(195),footprints)
(w/'king-street/public/review-routes.json').write_text(json.dumps({'routes':routes,'source':'active mapped road geometry','cameraHeightM':1.25,'speedMps':10,'coverage':'road centerline traversal; not complete facade accuracy verification'},separators=(',',':')))
print({'routes':len(routes),'points':sum(map(len,routes))})
