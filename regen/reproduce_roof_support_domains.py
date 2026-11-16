"""Read-only reproduction of exported native roof Dirichlet ownership.

The review massing coordinates are rounded1mm, so reproduction is accepted only
when every original exported structural plane agrees within2cm over its bounds.
"""
import types
from pathlib import Path
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from pipeline import roof_shell,spatial_rules
from regen.rebuild_roof_support_review import parameters


def reproduce(mass, context, observations, profiles, exported_audit):
    footprint=Polygon(mass['outline'][0],mass['outline'][1:]);raw=np.asarray(context)
    raw=raw[contains_xy(footprint,raw[:,0],raw[:,1])&(raw[:,2]>mass['z0']+3)]
    a,b,c,d=footprint.bounds;nx=max(2,int(np.ceil((c-a)/1.5)));ny=max(2,int(np.ceil((d-b)/1.5)))
    inner=raw[contains_xy(footprint.buffer(-.65),raw[:,0],raw[:,1])]
    _,keep=spatial_rules.surface_raster(inner,a,d,1.5,nx,ny);inner=inner[keep]
    _,keep=spatial_rules.surface_raster(raw,a,d,1.5,nx,ny);boundary=raw[keep]
    boundary=boundary[np.array([footprint.boundary.distance(Point(p))<=1 for p in boundary[:,:2]])]
    raw=np.unique(np.vstack([inner,boundary]),axis=0)
    module=types.ModuleType('pipeline._roof_support_export_reproduction');module.__package__='pipeline'
    exec(compile(Path(roof_shell.__file__).read_text(),str(roof_shell.__file__),'exec'),module.__dict__)
    module._retain_owned_components=lambda mask,*args,**kwargs:(np.asarray(mask,bool),[])
    regions,fit=module._fit_structural_regions(footprint,raw,.5)
    constraints,host=parameters(observations,profiles,mass['id'],mass['z0'])
    regions,_=module._recover_front_planes(footprint,raw,host,constraints,regions,.5)
    regions,_,_=module._recover_supported_ridges(footprint,regions,.5)
    original=exported_audit['roofShell']['structuralPlaneSurface'].get('planes',[])
    design=np.array([[x,y,1] for x in (a,c) for y in (b,d)])
    errors=[min((float(abs(design@(np.asarray(p)-r['plane'])).max()) for r in regions),default=float('inf')) for p in original]
    verified=bool(original) and bool(errors) and max(errors)<=.02
    geometry=unary_union([r['geometry'] for r in regions]) if verified else None
    return geometry,{'status':'exported-plane-domain-reproduced' if verified else 'unresolved-plane-reproduction',
        'originalPlanes':original,'reproducedPlanes':[r['plane'].tolist() for r in regions],
        'maximumPlaneDifferenceAcrossFootprintM':max(errors,default=None),'planeJoinToleranceM':.02,
        'regions': [{'plane':r['plane'].tolist(),'geometryWkt':r['geometry'].wkt} for r in regions] if verified else [],
        'qualification':'Exact existing native fit/crease ownership, using isolated module without the newer raw-cell rejection/island gate. Rounded massing boundary is independently checked through plane reproduction.'}
