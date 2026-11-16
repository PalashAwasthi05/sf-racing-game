from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, box
from pipeline.roof_envelope import continuous_patches
from pipeline.roof_observations import validate
from general_world.roof_planes import roof_solid


def test_fine_sampling_recovers_measured_slopes_without_relaxing_validation():
    data = np.load(Path(__file__).parent/'tests/fixtures/small_compound_roof.npz')
    for offset in [np.zeros(3), np.array([200., -300., 45.])]:
        p = data['points'] + offset
        g = Polygon(data['outline'] + offset[:2])
        before, _ = continuous_patches(g,p,1.5,refine=False)
        after, audit = continuous_patches(g,p,1.5,refine=True)
        assert not before and after and audit['refinedSampling']
        assert all(r['heldOutP95M'] <= .30 and r['explainedInterior'] >= .90 for r in audit['patches'] if 'rejected' not in r)
        assert sum(patch['geometry'].area for patch in after) > .4*g.area
        for patch in after:
            mesh=roof_solid(patch['geometry'],patch['model'],float(p[:,2].min())-1)
            assert mesh.is_watertight and mesh.is_volume


def test_flat_measurements_reject_invented_mansard():
    x,y=np.meshgrid(np.arange(.1,12,.2),np.arange(.1,16,.2))
    p=np.c_[x.ravel(),y.ravel(),np.full(x.size,10.)]
    patches,audit=validate(box(0,0,12,16),p,{'kind':'mansard','wallTopM':8.,'riseM':3.,'pitchDeg':72},0.)
    assert patches is None and audit['status']=='contradicted'
    slopes,_=continuous_patches(box(0,0,12,16),p,1.5)
    assert not slopes


def test_observed_dimensions_can_be_supported_by_independent_surface():
    x,y=np.meshgrid(np.arange(.1,12,.2),np.arange(.1,16,.2))
    z=8.+np.minimum(np.minimum(x,12-x),np.minimum(y,16-y))*.5
    p=np.c_[x.ravel(),y.ravel(),z.ravel()]
    patches,audit=validate(box(0,0,12,16),p,{'kind':'hip','wallTopM':8.,'riseM':3.,'overhangM':0},0.)
    assert patches and audit['status']=='supported'


def test_exterior_slope_lowers_wall_base_but_interior_equipment_does_not():
    from pipeline.roof_envelope import supported_eave_base
    g=box(0,0,10,16)
    slope={'geometry':g,'model':{'planes':[[.5,0,8.],[-.5,0,13.]],'operation':'min'}}
    assert supported_eave_base(g,[slope],11.) == 8.
    slope['geometry']=box(3,3,7,12)
    assert supported_eave_base(g,[slope],11.) == 11.
