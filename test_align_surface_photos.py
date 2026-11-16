import cv2
import numpy as np

from regen.align_surface_photos import align, project


def surface():
    image=np.full((500,600,3),185,dtype='uint8')
    rng=np.random.default_rng(123)
    for _ in range(180):
        x,y=rng.integers([15,15],[585,485]);radius=int(rng.integers(2,12))
        color=tuple(int(v) for v in rng.integers(20,235,3))
        cv2.circle(image,(int(x),int(y)),radius,color,-1)
    return image


def test_known_planar_camera_warp_recovers_independent_landmarks():
    a=surface();expected=np.array([[.97,.035,12.],[-.02,1.015,11.],[.00004,.00008,1.]])
    b=cv2.warpPerspective(a,expected,(600,500),borderValue=(185,185,185))
    b[180:260,230:310]=90  # an occluder is not a transferable wall feature
    result=align(a,b)
    assert result['accepted'],result
    checks=np.array([[80,70],[510,85],[120,410],[480,390]],float)
    actual=project(checks,np.asarray(result['homographySourceToTarget']))
    assert np.max(np.linalg.norm(actual-project(checks,expected),axis=1))<1.
    assert result['holdoutMatches']>=8
    assert result['absolutePositionValidated'] is False
    assert result['physicalPlaneValidated'] is False


def test_featureless_wall_cannot_establish_registration():
    image=np.full((500,600,3),170,dtype='uint8')
    result=align(image,image.copy())
    assert not result['accepted']
    assert result['reason']=='insufficient-descriptors'


def test_all_features_masked_out_does_not_fall_back_to_occluders():
    image=surface();mask=np.zeros(image.shape[:2],dtype=bool)
    result=align(image,image.copy(),mask,mask)
    assert not result['accepted']
    assert result['sourceKeypoints']==result['targetKeypoints']==0
