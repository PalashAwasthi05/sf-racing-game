import unittest,json,types
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box
import pipeline
from regen.staged_gable_ownership import roof_gable_ownership as gable
from pipeline.facade_lint import clean
from test_provisional_gable_ownership import SourceOutlineTests

class StagedGableTests(unittest.TestCase):
 def profile(self):
  p=SourceOutlineTests().profile();p['sourceIds']=['mapillary:source-image'];p['readingProvenance']={'status':'ok','accepted':True,'registeredCropSource':True,'sourceReading':{'path':'source.json','sha256':'a'*64},'image':'source-image'};return clean(p)[0]
 def test_lint_outline_prepared_with_partial_source_span(self):
  p=self.profile();prior,audit=gable.prepare_gable_prior(p,0);self.assertIsNotNone(prior);self.assertEqual(audit['status'],'prepared-provisional-source-outline');self.assertAlmostEqual(prior['traceUZ'][-1,0],8.9);self.assertIsNone(prior['uncertaintyM']);self.assertEqual(p['parapets'],[])
 def test_bad_provenance_and_step_abstain(self):
  p=self.profile();p['readingProvenance']['accepted']=False;self.assertIsNone(gable.prepare_gable_prior(p,0)[0]);p=self.profile();p['gableOutline'][1][0]=p['gableOutline'][0][0];self.assertIn('riser-topology',gable.prepare_gable_prior(p,0)[1]['reason']);self.assertIsNone(gable.prepare_gable_prior(None,0)[0])
 def test_conflicting_source_outlines_abstain(self):
  p=self.profile();a,_=gable.prepare_gable_prior(p,0);b=dict(a);b['traceUZ']=a['traceUZ']+[0,.2];xy,f=SourceOutlineTests().grid();z=np.full(len(xy),10.4);known=xy[:,1]>=2;out,photo,changed,audit=gable.solve(xy,f,z,known,~known,[a,b],box(0,0,10,6));self.assertFalse(changed.any());self.assertIn('conflicting',audit['status']);self.assertTrue(np.array_equal(out,z))
 def test_topology_never_snaps_to_courtyard(self):
  p=self.profile();prior,_=gable.prepare_gable_prior(p,0);prior['planeV']=-3;prior['traceUZ'][:,0]=[3,3.1,4,5,5.8,6];foot=box(0,0,10,10).difference(box(3,3,6,6));points,audit=gable.outline_mesh_points([prior],foot);self.assertEqual(points,[])
 def test_solver_never_owns_unobserved_courtyard_ring(self):
  from shapely.geometry import Polygon
  prior,_=gable.prepare_gable_prior(self.profile(),0);prior['planeV']=-3;prior['traceUZ'][:,0]=[3,3.1,4,5,5.8,6];xy,f=SourceOutlineTests().grid();foot=box(0,0,10,6).difference(box(3,3,6,5));f=np.array([q for q in f if foot.covers(Polygon(xy[q]))]);z=np.full(len(xy),10.4);known=xy[:,1]>=5.5;out,photo,changed,audit=gable.solve(xy,f,z,known,~known,[prior],foot);self.assertFalse(photo.any());self.assertTrue(np.array_equal(out,z))
 def test_mixed_geometry_keeps_polygon_and_discards_line_remnant(self):
  from shapely.geometry import GeometryCollection,LineString
  prior,_=gable.prepare_gable_prior(self.profile(),0);xy,f=SourceOutlineTests().grid();z=np.full(len(xy),10.4);known=xy[:,1]>=2;foot=GeometryCollection([box(0,0,10,6),LineString([(12,0),(12,6)])]);out,photo,changed,audit=gable.solve(xy,f,z,known,~known,[prior],foot);self.assertGreater(photo.sum(),0);self.assertTrue(np.isfinite(out).all());points,_=gable.outline_mesh_points([prior],foot);self.assertGreater(len(points),0)
 def test_staged_roof_build_closes_source_outline_preserves_native_support(self):
  source=Path('regen/staged_gable_ownership/roof_shell.py');module=types.ModuleType('pipeline._staged_gable_test');module.__package__='pipeline';exec(compile(source.read_text(),str(source),'exec'),module.__dict__);pipeline.roof_gable_ownership=gable
  xy=np.array([(x,y) for y in np.arange(2,9.75,.25) for x in np.arange(.25,9.75,.25)]);points=np.c_[xy,np.full(len(xy),10.4)];foot=box(0,0,10,10);native=module.build(foot,points,3);prior,_=gable.prepare_gable_prior(self.profile(),0);knots,_=gable.outline_mesh_points([prior],foot);after=module.build(foot,points,3,gable_priors=[prior],source_outline_points=knots)
  self.assertGreater(after['audit']['sourceGableOwnership']['appliedBoundaryVertices'],0);self.assertEqual(after['audit']['sourceGableOwnership']['measuredAnchorMaximumChangeM'],0)
  q=native['nativeSupportState'];distance,index=cKDTree(after['surface'].vertices[:,:2]).query(q['xy'][q['known']]);self.assertLess(distance.max(),1e-6);self.assertLess(np.max(abs(after['surface'].vertices[index,2]-q['z'][q['known']])),1e-6)
  tri=after['surface'].triangles[:,:,:2];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];area=abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()/2;self.assertAlmostEqual(area,foot.area,6);self.assertLess(after['surface'].vertices[np.argmin(np.linalg.norm(after['surface'].vertices[:,:2],axis=1)),2],8)
  json.dumps(after['audit'])
