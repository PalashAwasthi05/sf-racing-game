import unittest
import numpy as np
from shapely.geometry import box,Polygon
from pipeline.roof_envelope import prepare_grid,owned_samples

class RoofEnvelopeTests(unittest.TestCase):
 def test_outside_rectangle_does_not_lower_l_shaped_roof(self):
  p=Polygon([(0,0),(6,0),(6,2),(2,2),(2,6),(0,6)])
  raw=np.zeros((6,6));raw[:,:]=30
  from shapely import contains_xy
  yy,xx=np.indices(raw.shape);inside=contains_xy(p,xx+.5,6-yy-.5)
  raw[~inside]=-50
  grid,known,xy,audit=prepare_grid(raw,p,0,6,1)
  self.assertTrue(np.all(grid[known]==30))
  self.assertEqual(audit['measuredCells'],20)
 def test_unmeasured_cells_do_not_become_tier_evidence(self):
  raw=np.full((6,6),20.);raw[2:4,2:4]=np.nan
  grid,known,xy,audit=prepare_grid(raw,box(0,0,6,6),0,6,1)
  self.assertEqual(audit['unknownInsideCells'],4)
  self.assertFalse(known[2,2]);self.assertTrue(np.isfinite(grid).all())
 def test_walls_sample_exact_generated_levels(self):
  samples=owned_samples(box(0,0,10,10),None,10,[{'geometry':box(0,0,5,10),'top':2.5}],[],[(box(8,0,10,10),9)])
  self.assertTrue(np.all(samples[samples[:,0]<5,2]==12.5))
  self.assertTrue(np.all(samples[samples[:,0]>8,2]==9))
if __name__=='__main__':unittest.main()
