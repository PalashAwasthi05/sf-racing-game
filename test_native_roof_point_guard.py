import unittest
import numpy as np,trimesh
from regen.check_native_roof_points import heights
class NativePointGuardTests(unittest.TestCase):
 def test_bounds_overlap_does_not_admit_triangle_outside_point(self):
  mesh=trimesh.Trimesh(vertices=[[0,0,10],[1,0,10],[0,1,10],[0,1,0],[1,0,0],[1,1,0]],faces=[[0,1,2],[3,4,5]],process=False)
  z,_=heights(mesh,np.array([[.1,.1],[.75,.75]]),np.array([0.,10.]));self.assertTrue(np.allclose(z,[10.,0.]))
