import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from regen import register_cameras as rc
import auto_read_frontages_v24 as reader


def test_skyline_fit_alone_cannot_move_street_crop():
    camera={'xy':np.array([1.,2.]),'R':np.eye(3)}
    rec={'accepted':True,'xy':[8,9],'dx':7,'dy':7,'dyawDeg':0,'cameraHeightM':3,
         'heldout':{'passed':True},'streetRegistrationVerified':False}
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'poses.json';p.write_text(json.dumps({'a':rec}))
        result=rc.apply_registration({'a':camera},p,bias_fallback=False,require_heldout=True,require_street=True)
        assert result['a']['registered'] is False
        np.testing.assert_array_equal(result['a']['xy'],camera['xy'])
        rec['streetRegistrationVerified']=True;p.write_text(json.dumps({'a':rec}))
        result=rc.apply_registration({'a':camera},p,bias_fallback=False,require_heldout=True,require_street=True)
        np.testing.assert_array_equal(result['a']['xy'],[8,9])


def test_rectifier_applies_camera_height_correction_once():
    wall={'midpoint':[0,5],'normal':[0,-1],'lengthM':2,'heightM':3}
    camera={'xy':np.array([0.,0.]),'R':np.eye(3),'groundZ':2.,'heightOffsetM':.6}
    rays=[]
    def capture(d,R):
        rays.append(d.copy());return np.zeros(d.shape[:2]),np.zeros(d.shape[:2])
    with patch.object(reader.fc,'world_dirs_to_pano_uv',capture):
        reader.rectify(wall,camera,2.,np.zeros((8,16,3),dtype=np.uint8))
        first = np.concatenate(rays, axis=0); rays.clear()
        reader.rectify(wall,{**camera,'heightOffsetM':0},2.,np.zeros((8,16,3),dtype=np.uint8))
    np.testing.assert_allclose(first[...,2]-np.concatenate(rays,axis=0)[...,2],-.6)


def test_heldout_skyline_rejects_a_fit_that_only_matches_training_sectors():
    camera={'xy':np.array([0.,0.])}
    obs=np.full(len(rc.AZ),np.radians(4.))
    rec={'xy':[1.,0.],'dyawDeg':0.,'tiltDeg':[0.,0.], 'cameraHeightM':3.,'accepted':True,'why':'ok'}
    def fit(cam, training, *args):
        # Half of the bearings must be absent from parameter fitting.
        assert np.isfinite(training).sum()==len(rc.AZ)//2
        return dict(rec),(training,None,None,0.)
    def heights(S,o,xy):
        return np.full((1,len(rc.AZ),1), float(xy[0,0])),np.array([1.])
    def sky(h,r,z):
        # Nominal is off by 2 degrees. The proposed fit is off by 4 on the
        # held-out sectors despite reporting success on its training sectors.
        return np.full((1,len(rc.AZ)),np.radians(8. if h[0,0,0] else 6.))
    with patch.object(rc,'register',fit),patch.object(rc,'predicted_heights',heights),patch.object(rc,'skyline_of',sky),patch.object(rc,'lookup',lambda *args: np.array([0.])):
        result,_=rc.register_heldout(camera,obs,None,None,None)
    assert not result['accepted']
    assert not result['heldout']['passed']
    assert not result['streetRegistrationVerified']
