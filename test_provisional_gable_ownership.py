import unittest
import numpy as np
from shapely.geometry import box
from regen.provisional_gable_ownership import source_priors,solve

class SourceOutlineTests(unittest.TestCase):
 def profile(self):
  return {'id':'regen-example__s0','match':{'partId':'example'},'normal':[0,-1],'planeV':0,'frameU':[0,10],'heightRefM':10,'reading':{'confidence':'medium','wallTopM':7.1,'notes':'Explicit stepped gable front; adjoining wall beyond source span.'},'parapets':[{'top':[[0,.71],[.1,.71],[.4,1.04],[.7,.72],[.8,.71],[.89,.71]]}]}
 def grid(self):
  xy=np.array([(x,y) for y in np.arange(0,6.1,.5) for x in np.arange(0,10.1,.5)]);faces=[];w=21
  for y in range(12):
   for x in range(20):
    a=y*w+x;faces.extend([[a,a+1,a+w],[a+1,a+w+1,a+w]])
  return xy,np.array(faces)
 def test_partial_span_preserves_measured_rear_and_neighbor(self):
  priors,_=source_priors([self.profile()],'example','example',0);xy,f=self.grid();known=xy[:,1]>=2;z=np.full(len(xy),10.4);out,photo,changed,audit=solve(xy,f,z,known,~known,priors,box(0,0,10,6));self.assertGreater(photo.sum(),10);self.assertTrue(np.array_equal(out[known],z[known]));self.assertTrue(np.array_equal(out[xy[:,0]>8.9],z[xy[:,0]>8.9]));self.assertTrue(np.isfinite(out).all());__import__('json').dumps(audit);self.assertLess(out[(xy[:,0]==0)&(xy[:,1]==0)][0],8);self.assertGreater(out[(xy[:,0]==4)&(xy[:,1]==0)][0],10)
 def test_second_multivertex_shape_and_protected_source_domain(self):
  p=self.profile();p['parapets'][0]['top']=[[0,.71],[.25,.71],[.5,.95],[.75,.71],[1,.71]];priors,_=source_priors([p],'example','example',0);xy,f=self.grid();known=xy[:,1]>=2;z=np.full(len(xy),10.4);protected=box(3,-.01,5,2);out,photo,changed,audit=solve(xy,f,z,known,~known,priors,box(0,0,10,6),[protected]);mask=(xy[:,0]>=3)&(xy[:,0]<=5)&(xy[:,1]<=2);self.assertTrue(np.array_equal(out[mask],z[mask]));self.assertEqual(audit['protectedMaximumChangeM'],0)
 def test_reject_decorative_or_nonmonotone_outline(self):
  p=self.profile();p['reading']['notes']='Flat cornice and ornamental pediment';self.assertEqual(source_priors([p],'example','example',0)[0],[])
  p=self.profile();p['parapets'][0]['top'][2][0]=.05;self.assertEqual(source_priors([p],'example','example',0)[0],[])
 def test_cleaner_preserves_trace_as_gable_outline_not_parapet(self):
  from pipeline.facade_lint import clean
  p=self.profile();q,issues=clean(p);self.assertEqual(q['parapets'],[]);self.assertEqual(q['gableOutline'],p['parapets'][0]['top']);self.assertTrue(any(i['rule']=='parapets' for i in issues))
 def test_unreferenced_grid_node_does_not_disconnect_solve(self):
  priors,_=source_priors([self.profile()],'example','example',0);xy,f=self.grid();xy=np.vstack([xy,[2.1,.1]]);known=xy[:,1]>=2;z=np.full(len(xy),10.4);out,photo,changed,audit=solve(xy,f,z,known,~known,priors,box(0,0,10,6));self.assertTrue(np.isfinite(out).all());self.assertFalse(changed[-1]);self.assertGreater(photo.sum(),0)
 def test_no_interior_support_abstains(self):
  priors,_=source_priors([self.profile()],'example','example',0);xy,f=self.grid();z=np.full(len(xy),10.4);out,photo,changed,audit=solve(xy,f,z,np.zeros(len(z),bool),np.ones(len(z),bool),priors,box(0,0,10,6));self.assertFalse(photo.any());self.assertTrue(np.array_equal(z,out))
