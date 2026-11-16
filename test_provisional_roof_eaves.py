import numpy as np
from shapely.geometry import box
from regen.provisional_roof_eaves import solve_unknown_apron,source_priors,solve_planar_front_cap


def grid():
    xy=np.array([[x,y] for y in range(5) for x in range(5)],float)
    faces=[]
    for y in range(4):
        for x in range(4):
            i=y*5+x;faces.extend([[i,i+1,i+6],[i,i+6,i+5]])
    return xy,np.array(faces)


def test_only_unknown_front_apron_changes_and_keeps_measured_levels():
    xy,faces=grid();known=xy[:,1]>=2;unknown=~known;z=8+.3*xy[:,1];z[unknown]=10
    prior={'side':'source','profileId':'source','normal':np.array([0,-1]),'tangent':np.array([1,0]),'planeV':0.,'frameU':[0,4],
        'levelZ':8.1,'heightIntervalM':[7.7,8.1],'confidence':'medium','eaveConfidence':'low','uncertaintyM':.4,'wallTopM':8.,'sourceReading':{}}
    after,_,_,audit=solve_unknown_apron(xy,faces,z,known,unknown,[prior],box(0,0,4,4))
    assert audit['appliedBoundaryVertices']==5
    assert np.all(after[xy[:,1]==0]==8.1)
    assert np.array_equal(after[known],z[known])
    assert np.all(after<=z) and audit['measuredAnchorMaximumChangeM']==0


def test_source_protected_domain_and_no_roof_semantics_abstain():
    xy,faces=grid();known=xy[:,1]>=2;z=8+.3*xy[:,1];z[~known]=10
    prior={'side':'source','profileId':'source','normal':np.array([0,-1]),'tangent':np.array([1,0]),'planeV':0.,'frameU':[0,4],
        'levelZ':8.1,'heightIntervalM':[7.7,8.1],'confidence':'medium','eaveConfidence':'low','uncertaintyM':.4,'wallTopM':8.,'sourceReading':{}}
    protected=box(1.5,-.1,2.5,3)
    after,_,_,audit=solve_unknown_apron(xy,faces,z,known,~known,[prior],box(0,0,4,4),protected_domains=[protected])
    assert np.array_equal(after[xy[:,0]==2],z[xy[:,0]==2])
    obs={'side':'s','confidence':'high','hostRoofKind':'unknown','eaveM':8.,'eaveEvidence':{'junctionHeightRangeM':[7.7,8.1]}}
    profile={'id':'regen-s','match':{'partId':'building'}}
    admitted,rejected=source_priors([obs],[profile],'building',0.,7.)
    assert admitted==[] and rejected[0]['reason']=='no-confident-non-gable-pitched-host-observation'


def test_full_observed_frontage_eave_does_not_leave_unknown_endpoint_fins():
    xy,faces=grid();known=(xy[:,1]>=2)&(xy[:,0]>=1)&(xy[:,0]<=3);z=8+.3*xy[:,1];z[~known]=10
    prior={'side':'source','profileId':'source','normal':np.array([0,-1]),'tangent':np.array([1,0]),'planeV':0.,'frameU':[0,4],
        'levelZ':8.1,'heightIntervalM':[7.7,8.1],'confidence':'medium','eaveConfidence':'low','uncertaintyM':.4,'wallTopM':8.,'sourceReading':{}}
    after,_,_,audit=solve_unknown_apron(xy,faces,z,known,~known,[prior],box(0,0,4,4),support_mode='observed-frontage')
    assert audit['appliedBoundaryVertices']==5
    np.testing.assert_array_equal(after[known],z[known])
    np.testing.assert_array_equal(after[xy[:,1]==0],np.full(5,8.1))
    assert max(x['supportLateralOffsetM'] for x in audit['records'][0]['interiorSupports'])==1


def test_planar_front_cap_replaces_unknown_sag_without_moving_roof_sides():
    xy,faces=grid();known=xy[:,1]>=2;z=8+np.minimum(xy[:,0],4-xy[:,0]);z[~known]=7
    prior={'side':'source','profileId':'source','normal':np.array([0,-1]),'tangent':np.array([1,0]),'planeV':0.,'frameU':[0,4],
        'levelZ':8.1,'heightIntervalM':[7.9,8.1],'confidence':'medium','eaveConfidence':'low','uncertaintyM':.2,'wallTopM':8.,'sourceReading':{}}
    crease={'model':{'planes':[[1,0,8],[-1,0,12]],'operation':'min'},'zone':box(1.5,1.5,2.5,4),'audit':{'maximumDirectSupportContinuationM':.7}}
    solved,audit=solve_planar_front_cap(xy,faces,z,known,~known,[prior],box(0,0,4,4),[crease])
    assert solved is not None
    after=solved[0];np.testing.assert_array_equal(after[known],z[known])
    assert audit['maximumRaisingOfUnmeasuredFillM']>1
    assert after[(xy[:,0]==2)&(xy[:,1]==1)][0]>8
    assert audit['measuredAnchorMaximumChangeM']==0
    absent,rejected=solve_planar_front_cap(xy,faces,z,known,~known,[prior],box(0,0,4,4),[])
    assert absent is None and rejected['status']=='no-admissible-planar-front-cap'
