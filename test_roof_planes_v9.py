import numpy as np
from shapely.geometry import box,Polygon
from general_world.roof_planes import fit_roof,roof_solid,evaluate

def cloud(fn):
    x,y=np.meshgrid(np.arange(.1,20,.3),np.arange(.1,14,.3))
    return np.c_[x.ravel(),y.ravel(),fn(x,y).ravel()]

def test_slope_has_no_steps():
    g=box(0,0,20,14);p=cloud(lambda x,y:10+.3*x)
    model,report=fit_roof(g,p)
    assert report['accepted'] and report['sloped']
    mesh=roof_solid(g,model,5)
    top=mesh.triangles_center[mesh.face_normals[:,2]>.1]
    assert np.max(np.abs(top[:,2]-(10+.3*top[:,0])))<1e-6
    assert mesh.is_volume and mesh.is_watertight

def test_shared_gable_ridge():
    g=box(0,0,20,14);p=cloud(lambda x,y:14-.4*np.abs(x-10))
    model,report=fit_roof(g,p);mesh=roof_solid(g,model,5)
    assert report['accepted'] and report['planes']==2
    assert mesh.is_watertight and mesh.is_volume
    assert abs(mesh.bounds[1,2]-14)<.05
    assert abs(mesh.volume-1960)<5

def test_step_is_not_misread_as_slope():
    g=box(0,0,20,14);p=cloud(lambda x,y:np.where(x<10,10.,15.))
    model,report=fit_roof(g,p)
    assert model is None and not report['accepted']

def test_courtyard_not_capped():
    g=Polygon([(0,0),(20,0),(20,14),(0,14)],holes=[[(8,4),(8,10),(12,10),(12,4)]])
    p=cloud(lambda x,y:10+.3*x);model,report=fit_roof(g,p);mesh=roof_solid(g,model,5)
    assert report['accepted'] and mesh.is_watertight
    assert not any(Polygon(t[:,:2]).intersection(box(8.001,4.001,11.999,9.999)).area>1e-6 for t,n in zip(mesh.triangles,mesh.face_normals) if n[2]>.1)

def test_sparse_and_curved_roofs_unresolved():
    g=box(0,0,20,14);p=cloud(lambda x,y:10+.3*x);model,r=fit_roof(g,p[p[:,0]<5]);assert model is None
    p=cloud(lambda x,y:10+.10*(x-10)**2+.1*(y-7)**2);model,r=fit_roof(g,p);assert model is None

def test_input_order_does_not_change_roof():
    g=box(0,0,20,14);p=cloud(lambda x,y:10+.3*x)
    a,_=fit_roof(g,p);b,_=fit_roof(g,p[::-1]);assert a==b

def test_main_reconstruction_uses_continuous_roof():
    from general_world.geometry import reconstruct_building
    from general_world.policy import Policy
    g=box(0,0,20,14);p=cloud(lambda x,y:10+.3*x)
    m,r=reconstruct_building(g,p,lambda x,y:np.zeros_like(x),Policy())
    assert r['roofStatus']=='continuous-plane-envelope' and m.is_volume
    tops=m.triangles_center[m.face_normals[:,2]>.1]
    assert np.max(np.abs(tops[:,2]-(10+.3*tops[:,0])))<1e-5

if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_')]
    for test in tests:test()
    print({'roofTestsPassed':len(tests)})
