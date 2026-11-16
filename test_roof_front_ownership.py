import copy
import numpy as np
import pytest
from shapely.geometry import Polygon,box,MultiPoint,Point
from pipeline.roof_front_ownership import (
    prepare_front_prior,solve_front_ownership,
    continue_supported_plane_to_observed_perimeter,merge_preserved_components)


def source(frame=(0,4),span=None):
    obs={'side':'joined-source','confidence':'medium','hostRoofKind':'hip','eaveM':8.,'anchorUncertaintyM':.2,
        'eaveConfidence':'low','eaveEvidence':{'junctionHeightRangeM':[7.9,8.1]}}
    if span is not None:obs['sourceFrontageU']=span
    profile={'id':'joined-profile','normal':[0,-1],'planeV':0,'frameU':list(frame),
        'reading':{'wallTopM':8.,'confidence':'medium'},'readingProvenance':{'status':'accepted-approximate'}}
    return obs,profile


def grid():
    xy=np.array([[x,y] for y in range(5) for x in range(5)],float);faces=[]
    for y in range(4):
        for x in range(4):
            i=y*5+x;faces.extend([[i,i+1,i+6],[i,i+6,i+5]])
    return xy,np.array(faces)


def test_absolute_source_span_is_intersected_without_changing_inputs_or_confidence():
    obs,profile=source((10,20),(8,14));original=copy.deepcopy((obs,profile))
    prior,audit=prepare_front_prior(obs,profile,-5,2)
    assert prior['frameU']==[10,14] and prior['profileFrameU']==[10,20]
    assert prior['confidence']=='medium' and prior['eaveConfidence']=='low'
    np.testing.assert_allclose(prior['eaveZRange'],[2.9,3.1])
    assert audit['measuredAnchor'] is False and audit['confidenceUpgraded'] is False
    assert (obs,profile)==original
    obs.pop('sourceFrontageU');legacy,_=prepare_front_prior(obs,profile,-5,2)
    assert legacy['frameU']==[10,20]


@pytest.mark.parametrize('span',[None,[14,11],[0,3],[True,14],[11,float('nan')]])
def test_explicit_invalid_or_empty_span_abstains_instead_of_using_full_frontage(span):
    obs,profile=source((10,20));obs['sourceFrontageU']=span
    prior,audit=prepare_front_prior(obs,profile,0,7)
    assert prior is None and audit['status']=='unresolved'


@pytest.mark.parametrize('evidence',[[], 'bad', {'junctionHeightRangeM':[8]}, {'junctionHeightRangeM':[9,7]}])
def test_invalid_explicit_eave_evidence_is_local_unresolved(evidence):
    obs,profile=source();obs['eaveEvidence']=evidence
    prior,audit=prepare_front_prior(obs,profile,0,7)
    assert prior is None and audit['status']=='unresolved'


def test_planar_cap_uses_separate_provisional_mask_and_keeps_measured_roof():
    xy,faces=grid();measured=xy[:,1]>=2;z=8+np.minimum(xy[:,0],4-xy[:,0]);z[~measured]=7
    obs,profile=source();prior,_=prepare_front_prior(obs,profile,0,7)
    crease={'model':{'planes':[[1,0,8],[-1,0,12]],'operation':'min'},'zone':box(1.5,1.5,2.5,4),
        'audit':{'maximumDirectSupportContinuationM':.7}}
    result=solve_front_ownership(xy,faces,z,measured,~measured,[],[crease],box(0,0,4,4),[prior])
    np.testing.assert_array_equal(result['measured_anchor_mask'],measured)
    np.testing.assert_array_equal(result['z'][measured],z[measured])
    assert result['provisional_boundary_mask'].any()
    assert not (result['provisional_boundary_mask']&measured).any()
    assert result['audit']['provisionalConstraintsAreNotMeasuredAnchors']


def flank_fixture(remote=False):
    xy,faces=grid();measured=xy[:,1]>=1;z=8+xy[:,1];z[~measured]=9
    obs,profile=source(span=[1,3]);prior,_=prepare_front_prior(obs,profile,0,7)
    points=np.array([[x,y] for x in (.5,1,2,3,3.5) for y in ((2.5,3.5) if remote else (.5,1.5))])
    region={'geometry':MultiPoint(points).convex_hull,'plane':np.array([0,1,8.]),'inlierXY':points,
        'validation':{'status':'heldout-validated','heldOutP95M':.05}}
    return xy,faces,z,measured,prior,region


def test_supported_plane_continues_only_to_its_observed_perimeter_span():
    xy,faces,z,measured,prior,region=flank_fixture()
    result=continue_supported_plane_to_observed_perimeter(xy,faces,z,measured,~measured,[region],box(0,0,4,4),[prior])
    selected=(xy[:,1]==0)&(xy[:,0]>=1)&(xy[:,0]<=3)
    np.testing.assert_array_equal(result['z'][selected],np.full(3,8.))
    np.testing.assert_array_equal(result['z'][measured],z[measured])
    np.testing.assert_array_equal(result['z'][~selected],z[~selected])
    assert result['audit']['records'][0]['maximumContinuationM']<=np.sqrt(2)*.5
    absent=continue_supported_plane_to_observed_perimeter(xy,faces,z,measured,~measured,[region],box(0,0,4,4),[])
    np.testing.assert_array_equal(absent['z'],z)


def test_remote_or_unvalidated_support_never_authorizes_a_flank():
    xy,faces,z,measured,prior,region=flank_fixture(remote=True)
    result=continue_supported_plane_to_observed_perimeter(xy,faces,z,measured,~measured,[region],box(0,0,4,4),[prior])
    np.testing.assert_array_equal(result['z'],z)
    region['validation']={}
    result=continue_supported_plane_to_observed_perimeter(xy,faces,z,measured,~measured,[region],box(0,0,4,4),[prior])
    np.testing.assert_array_equal(result['z'],z)


def test_continuation_does_not_bridge_a_courtyard_to_reach_support():
    xy,faces,z,measured,prior,region=flank_fixture()
    footprint=Polygon([(0,0),(4,0),(4,4),(0,4)],holes=[[(1.5,.25),(2.5,.25),(2.5,1.25),(1.5,1.25)]])
    region['inlierXY']=np.array([p for p in region['inlierXY'] if footprint.covers(Point(p))])
    result=continue_supported_plane_to_observed_perimeter(xy,faces,z,measured,~measured,[region],footprint,[prior])
    index=np.flatnonzero((xy[:,0]==2)&(xy[:,1]==0))[0]
    assert result['z'][index]==z[index]
    assert result['audit']['footprintUnchanged']


def test_existing_source_component_never_moves_with_revised_host():
    old={'kind':'dormer','xy':[2.,3.],'facing':[0,1],'widthM':2.,'bottomM':8.8,'ridgeM':10.4,
        'frontEvidence':{'coordinateSpace':'normalized-front','apertures':[]}}
    shifted={**old,'xy':[2.,4.]};new={**old,'xy':[5.,4.]}
    evidence={'side':'source','profileId':'joined','featureKind':'dormer','confidence':'medium'}
    result,audit=merge_preserved_components([old],{'status':'resolved','accepted':[evidence]},
        [shifted,new],{'accepted':[evidence,evidence],'rejected':[]})
    assert result[0] is old and result[0]['xy']==[2.,3.]
    assert len(result)==2 and result[1] is new
    assert audit['existingSourceParametersUnchanged'] and audit['confidenceUpgraded'] is False
