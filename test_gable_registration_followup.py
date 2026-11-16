import unittest
import numpy as np
from shapely.geometry import box
from test_staged_gable_ownership import StagedGableTests
from test_provisional_gable_ownership import SourceOutlineTests
from regen.staged_gable_ownership import roof_gable_ownership as helper

class RegistrationFollowupTests(unittest.TestCase):
 def profile(self,with_registration=True,offset=.003):
  p=StagedGableTests().profile();p['frameU']=[offset,10+offset];p['planeV']=.01
  if with_registration:p['sourceFrameRegistration']={'status':'exact-source-joined','side':'example__s0','image':'source-image','sourceReadingSha256':'a'*64,'metadataPath':'sides.json','metadataSha256':'b'*64,'pixelsPerMetre':45,'normal':p['normal'],'frameU':p['frameU'],'planeV':p['planeV'],'heightM':p['heightRefM']}
  return p
 def test_half_pixel_corner_registration_keeps_original_observation(self):
  p=self.profile();prior,_=helper.prepare_gable_prior(p,0);xy,f=SourceOutlineTests().grid();z=np.full(len(xy),10.4);known=xy[:,1]>=2;out,photo,changed,audit=helper.solve(xy,f,z,known,~known,[prior],box(0,0,10,6));origin=np.argmin(np.linalg.norm(xy,axis=1));self.assertTrue(photo[origin]);self.assertAlmostEqual(out[origin],7.1);self.assertAlmostEqual(prior['traceUZ'][0,0],.003);knots,_=helper.outline_mesh_points([prior],box(0,0,10,6));self.assertTrue(np.array_equal(knots[0],[0,0]));self.assertTrue(np.array_equal(out[known],z[known]));self.assertFalse(photo[xy[:,0]>8.91].any())
 def test_no_metadata_or_large_gap_does_not_expand_span(self):
  xy,f=SourceOutlineTests().grid();z=np.full(len(xy),10.4);known=xy[:,1]>=2
  for p in [self.profile(False),self.profile(True,.1)]:
   prior,_=helper.prepare_gable_prior(p,0);out,photo,changed,audit=helper.solve(xy,f,z,known,~known,[prior],box(0,0,10,6));self.assertFalse(photo[0])
 def test_mismatched_registration_identity_abstains(self):
  p=self.profile();p['sourceFrameRegistration']['image']='different';self.assertIsNone(helper.prepare_gable_prior(p,0)[0])
 def test_cleaned_semantic_change_can_have_no_issue(self):
  from pipeline.facade_lint import clean
  p=StagedGableTests().profile();p.pop('gableOutline');p['parapets']=[];p['roofs']=[{'kind':'gable','x':[0,1],'z':.71,'riseM':2}];q,issues=clean(p);self.assertEqual(issues,[]);self.assertIsNotNone(q.get('gableOutline'));self.assertIsNone(p.get('gableOutline'))
