import numpy as np
from shapely.geometry import LineString,Point
from general_world.review_route import review_routes

def test_continuous_and_all_roads_covered():
 lines=[LineString([(-10,0),(10,0)]),LineString([(0,-10),(0,10)])]
 route=review_routes(lines,Point(0,0).buffer(12),spacing_m=2)[0]
 assert max(np.linalg.norm(np.diff(route,axis=0),axis=1))<=2.001
 for p in [(-10,0),(10,0),(0,10),(0,-10)]:assert np.min(np.linalg.norm(np.asarray(route)-p,axis=1))<.001
 assert np.allclose(route[0],route[-1])

def test_excludes_building_and_does_not_teleport():
 routes=review_routes([LineString([(-10,0),(10,0)])],Point(0,0).buffer(12),Point(0,0).buffer(2))
 assert len(routes)==2
 for route in routes:
  assert all(np.linalg.norm(p)>=1.999 for p in route)
  assert max(np.linalg.norm(np.diff(route,axis=0),axis=1))<=3.001
