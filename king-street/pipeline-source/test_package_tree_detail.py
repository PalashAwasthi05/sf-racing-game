import struct
import unittest
import numpy as np
from package_tree_detail import extract_cell
from test_neighborhood_canopies import fixture


class TreeDetailTests(unittest.TestCase):
    def test_source_samples_bark_and_transforms_survive_tree_only_extraction(self):
        source,plan,expected=fixture()
        source.model['nodes']=[dict(node,matrix=matrix.ravel(order='F').tolist())for node,matrix in plan.nodes]
        source.model['scenes']=[{'nodes':list(range(len(plan.nodes)))}]
        payload,entries,stats,bounds,classes=extract_cell(source)
        self.assertEqual(classes,['bark','leaf1'])
        self.assertEqual(stats,{'sourceSamples':1000,'leafSamples':1000,'frondTriangles':0,'barkTriangles':24})
        self.assertEqual(entries[0]['kind'],'leaves')
        points=np.frombuffer(payload,dtype='<f4',count=3000,offset=entries[0]['offset']).reshape(-1,3)
        np.testing.assert_allclose(points,expected,atol=2e-6)
        self.assertEqual(len(payload)%4,0)
        self.assertLessEqual(bounds[0],float(points[:,0].min()))
        self.assertGreaterEqual(bounds[2],float(points[:,0].max()))
        # The same projected positions shifted to an arbitrary map location
        # retain sample count and byte layout; there is no special origin zone.
        for node in source.model['nodes']:
            matrix=np.array(node['matrix']).reshape(4,4,order='F');matrix[:3,3]+=[3000,7,-1000]
            node['matrix']=matrix.ravel(order='F').tolist()
        shifted,other,counts,_,_=extract_cell(source)
        self.assertEqual(stats,counts);self.assertEqual(entries,other)
        points2=np.frombuffer(shifted,dtype='<f4',count=3000,offset=other[0]['offset']).reshape(-1,3)
        np.testing.assert_allclose(points2,points+[3000,7,-1000],atol=.0002)


if __name__=='__main__':
    unittest.main()
