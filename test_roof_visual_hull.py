import numpy as np
import trimesh
from regen.propose_roof_visual_hull import ceiling,protected_vertices


def views(angle=0, same_image=False):
    result=[]
    for i,x in enumerate((-1,1)):
        result.append({'image':'same' if same_image else str(i),'sourceSha256':'a'*64,'index':i,
            'projection':{'cameraXYZ':[x,-10,0],'basisRightDownForward':[[1,0,0],[0,0,-1],[0,1,0]],
                'focalPx':100,'sizePx':[100,100]},
            'clearance':np.maximum(35-np.arange(100)[:,None],0).repeat(100,axis=1).astype(np.float32),
            'eligible':np.ones((100,100),np.float32),'angleMarginDeg':angle})
    return result


def test_supported_pitched_roof_preserved_while_unknown_spike_gets_ceiling():
    xyz=np.array([[0,0,1],[1,0,1.2],[1,1,1.2],[0,1,1],[.5,.5,2.5]])
    mesh=trimesh.Trimesh(vertices=xyz,faces=[[0,1,4],[1,2,4],[2,3,4],[3,0,4]],process=False)
    audit={'roofShell':{'structuralPlaneSurface':{'planes':[[.2,0,1]],'method':'harmonic-graph-interpolation-with-Dirichlet-plane-and-eave-anchors'}}}
    protected,_=protected_vertices(mesh,audit,[],[],{'id':'synthetic','baseRoofZ':.5})
    assert protected.tolist()==[True,True,True,True,False]
    proposed,evidence=ceiling(mesh.vertices[4],views(),0,0,1)
    assert proposed is not None and 0<evidence['loweringM']<1
    candidate=xyz.copy();candidate[4]=proposed
    assert np.array_equal(candidate[:4],xyz[:4])
    assert evidence['independentImages']==2
    assert all(p['after']['skyClearancePx']<=p['after']['allowancePx'] for p in evidence['sources'])


def test_camera_misregistration_margin_abstains_instead_of_lowering_roof():
    proposed,evidence=ceiling(np.array([.5,.5,2.5]),views(angle=10),.5,.25,1)
    assert proposed is None
    assert evidence['status']=='fewer-than-two-independent-margin-robust-contradictions'


def test_same_panorama_in_two_crops_is_not_independent_evidence():
    proposed,evidence=ceiling(np.array([.5,.5,2.5]),views(same_image=True),0,0,1)
    assert proposed is None and evidence['independentImages']==1
