import unittest,types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,box
import pipeline
from regen.staged_gable_ownership import roof_gable_continuation as helper
from test_gable_registration_followup import RegistrationFollowupTests
from test_provisional_gable_ownership import SourceOutlineTests

class ContinuationTests(unittest.TestCase):
 def module(self):
  path=Path('regen/staged_gable_ownership/roof_shell-continuation.py');m=types.ModuleType('pipeline._source_continuation_test');m.__package__='pipeline';exec(compile(path.read_text(),str(path),'exec'),m.__dict__);pipeline.roof_gable_ownership=helper;return m
 def test_clipped_corner_unknowns_continue_to_native_support_without_sliver(self):
  mod=self.module();foot=Polygon([(0,0),(-.0001,.02433),(-.01,6),(10,6),(10,0)]);xy=np.array([(x,y) for x in np.arange(.25,9.75,.25) for y in np.arange(2,5.75,.25)]);points=np.c_[xy,np.full(len(xy),10.4)];native=mod.build(foot,points,3);prior,_=helper.prepare_gable_prior(RegistrationFollowupTests().profile(),0);knots,_=helper.outline_mesh_points([prior],foot);result=mod.build(foot,points,3,gable_priors=[prior],source_outline_points=knots);audit=result['audit']['sourceGableOwnership'];self.assertGreater(audit['appliedBoundaryVertices'],0);self.assertEqual(audit['measuredAnchorMaximumChangeM'],0);self.assertEqual(audit['transition']['status'],'provisional-bounded-native-supported-transition');self.assertGreater(audit['transition']['additionalUnmeasuredVertices'],0)
  mesh=result['surface'];d=mesh.vertices[mesh.edges_unique[:,0]]-mesh.vertices[mesh.edges_unique[:,1]];grade=abs(d[:,2])/np.maximum(np.linalg.norm(d[:,:2],axis=1),1e-9);self.assertLess(grade.max(),2.5001);q=native['nativeSupportState'];distance,index=cKDTree(mesh.vertices[:,:2]).query(q['xy'][q['known']]);self.assertLess(distance.max(),1e-6);self.assertLess(abs(mesh.vertices[index,2]-q['z'][q['known']]).max(),1e-6);tri=mesh.triangles;u=tri[:,1,:2]-tri[:,0,:2];v=tri[:,2,:2]-tri[:,0,:2];self.assertAlmostEqual(abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]).sum()/2,foot.area,5)
 def test_irreconcilable_real_anchor_rejects_component(self):
  xy=np.array([[0,0],[1,0],[0,.02],[1,.02],[0,1],[1,1]],float);f=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4]]);z=np.full(6,10.4);initial=z.copy();initial[:2]=7.1;known=np.array([0,0,1,1,1,1],bool);photo=~known;zone=photo.copy();out,free,audit=helper._continue_unknown_transition(xy,f,z,initial,known,~known,photo,zone,np.zeros(6,bool),[(0,2),(1,3)]);self.assertIsNone(out);self.assertEqual(audit['status'],'rejected-source-native-anchor-conflict')
 def test_protected_native_step_is_never_smoothed(self):
  xy,f=SourceOutlineTests().grid();z=np.full(len(xy),10.4);z[xy[:,0]>=7]=13.0;known=xy[:,1]>=2;protected=(xy[:,0]>=7);prior,_=helper.prepare_gable_prior(RegistrationFollowupTests().profile(),0);out,photo,changed,audit=helper.solve(xy,f,z,known,~known,[prior],box(0,0,10,6),[box(7,-.1,10,6)]);self.assertTrue(np.array_equal(out[known|protected],z[known|protected]));self.assertEqual(audit.get('protectedMaximumChangeM',0),0)
